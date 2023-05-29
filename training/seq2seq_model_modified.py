###########################################
# Using Waymo Open Dataset to achive the human trajectory predition tasks
# Based on the 3d bounding box and 3d pose keypoints to predict the future trajctory
#
# Final Project of ROAS6000H HKUST-GZ
# Written by Yukai ZHANG and Mandan CHAO
#
######################################
"""Modify the original seq2seq_model_fn.py to satisfy our model and dataset"""

import sys
import numpy as np
import json
import sys
import os
import argparse
import time
from abc import abstractmethod
import tqdm

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath + "/../")

import utils.utils as utils
import utils.WarmUpScheduler as warm_up_scheduler
import data.H36MDataset_v2 as h36mdataset_fn
import data.MyWaymoDataset as waymodataset_fn
import visualize.viz as viz
import models.seq2seq_model as seq2seq_model


# Choose the device the torch runs on
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Min threshold for mean average precision in metters and set to 10cm
_MAP_TRESH = 0.10


class ModelFn(object):
    def __init__(
        self,
        params,
        train_dataset_fn=None,
        eval_dataset_fn=None,
        pose_encoder_fn=None,
        pose_decoder_fn=None,
    ):
        """Initialize the model function with given params"""
        self._params = params
        self._train_dataset_fn = train_dataset_fn
        self._eval_dataset_fn = eval_dataset_fn
        self._visualize = False  # unused param

        thisname = self.__class__.__name__
        self.init_model(pose_encoder_fn, pose_decoder_fn)
        self._loss_fn = self.loss_mse  # set the model loss function as mse
        self._model.to(_DEVICE)  # set the torch device as gpu/cpu
        self._optimizer_fn = self.select_optimizer()  # set the model optimizer function

        self.select_lr_fn()  # select the learning rate function
        self.finetune_init()  # initialize the fine tune function

        self._lr_db_curve = []

        # determine the learning rate type step/epoch
        lr_type = (
            "stepwise" if self._params["learning_rate_fn"] == "beatles" else "epochwise"
        )
        self._params["lr_schedule_type"] = lr_type

        self.evaluate_fn = (
            self.evaluate_waymo
        )  # TODO: write the evaluate function for waymo dataset

        # choose the written path
        self._writer = SummaryWriter(
            os.path.join(self._params["model_prefix"], "tf_logs")
        )

        # Read and print the params info
        m_params = filter(lambda p: p.requires_grad, self._model.parameters())
        nparams = sum([np.prod(p.size()) for p in m_params])
        print("[INFO] ({}) This module has {} parameters!".format(thisname, nparams))
        print("[INFO] ({}) Intializing ModelFn with params".format(thisname))
        for k, v in self._params.items():
            print("[INFO] ({}) {}: {}".format(thisname, k, v))

    def finetune_init(self):
        """Finetune the model from given ckpt"""
        if self._params["finetuning_ckpt"] is not None:
            print(
                "[INFO] (finetune_model) Finetuning from:",
                self._params["finetuning_ckpt"],
            )
            self._model.load_state_dict(
                torch.load(self._params["finetuning_ckpt"], map_location=_DEVICE)
            )

    def select_lr_fn(self):
        """Calls the selection of learning rate function"""
        self._lr_scheduler = self.get_lr_fn()
        lr_fn = self._params["learning_rate_fn"]
        if self._params["warmup_epochs"] > 0 and lr_fn != "beatles":
            # set this lr schedule to gradually warm-up(increasing) learning rate in optimizer.
            self._lr_scheduler = warm_up_scheduler.GradualWarmupScheduler(
                self._optimizer_fn,
                multiplier=1,
                total_epoch=self._params["warmup_epochs"],
                after_scheduler=self._lr_scheduler,
            )

    def get_lr_fn(self):
        """Creates the function to be used to generate the learning rate."""
        # torch.optim.lr_scheduler: provides the methods to adjust the learning rate based on the number of epochs
        if self._params["learning_rate_fn"] == "step":
            return torch.optim.lr_scheduler.StepLR(
                self._optimizer_fn, step_size=self._params["lr_step_size"], gamma=0.1
            )
        elif self._params["learning_rate_fn"] == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(
                self._optimizer_fn, gamma=0.95
            )
        elif self._params["learning_rate_fn"] == "linear":
            lr0, T = self._params["learning_rate"], self._params["max_epochs"]
            lrT = lr0 * 0.5
            m = (lrT - 1) / T
            lambda_fn = lambda epoch: m * epoch + 1.0
            return torch.optim.lr_scheduler.LinearLR()
        elif self._params["learning_rate_fn"] == "beatles":
            D = float(self._params["model_dim"])
            warmup = self._params["warmup_epochs"]
            lambda_fn = lambda e: (D ** (-0.5)) * min(
                (e + 1.0) ** (-0.5), (e + 1.0) * warmup ** (-1.5)
            )
            return torch.optim.lr_scheduler.LambdaLR(
                self._optimizer_fn, lr_lambda=lambda_fn
            )
        else:
            raise ValueError(
                "Unknown learning rate function: {}".format(
                    self._params["learning_rate_fn"]
                )
            )

    @abstractmethod
    def init_model(self, pose_encoder_fn, pose_decoder_fn):
        pass

    @abstractmethod
    def select_optimizer(self):
        pass

    @abstractmethod
    def compute_loss(self, inputs=None, target=None, preds=None):
        # Without using the inputs info
        return self._loss_fn(preds, target)

    def loss_mse(self, decoder_pred, decoder_gt):
        """Computes the L2 loss between predictions and ground truth."""
        step_loss = (decoder_pred - decoder_gt) ** 2
        step_loss = step_loss.mean()
        return step_loss

    def print_logs(self, step_loss, current_step):
        selection_logs = ""
        # if self._params['query_selection']:
        #     selection_logs = ' selection loss {:.4f}'.format(selection_loss)

        print(
            "[INFO] global {3:06d}; step {0:04d}; step_loss: {1:.4f}; lr: {2:.2e} {4:s}".format(
                current_step,
                step_loss,
                self._params["learning_rate"],
                self._global_step,
                selection_logs,
            )
        )

    # TODO: !!!define the loss: only pose loss
    def train_one_epoch(self, epoch):
        """Trains for a number of steps before evaluation"""
        epoch_loss = 0
        act_loss = 0
        sel_loss = 0
        N = len(self._train_dataset_fn)  # num of samples
        for current_step, sample in enumerate(self._train_dataset_fn):
            self._optimizer_fn.zero_grad()
            # for k in sample.keys():
            #     if k == '':
            #         sample[k] = sample[k].to(_DEVICE)

            decoder_pred = self._model(
                sample["encoder_inputs"]
            )  # TODO: maybe need modification to input

            step_loss = self.compute_loss(
                inputs=sample["encoder_inputs"],
                target=sample["decoder_outputs"],
                preds=decoder_pred,
            )
            epoch_loss += step_loss.item()

            step_loss.backward()

            if current_step % 10 == 0:
                step_loss = step_loss.cpu().data.numpy()
                self.print_logs(step_loss, current_step)
            self.update_learning_rate(self._global_step, mode="stepwise")
            self._global_step += 1

        return epoch_loss / N

    def train(self):
        """Main training loop"""
        self._params["learning_rate"] = self._lr_scheduler.get_last_lr()[0]
        self._global_step = 1
        thisname = self.__class__.__name__

        # train this model for all epoch as asked
        for e in range(self._params["max_epochs"]):
            self._scalars = {}
            self._model.train()
            start_time = time.time()
            epoch_loss = self.train_one_epoch(e)

            # act_log = ''
            self._scalars["epoch_loss"] = epoch_loss
            print("epoch {0:04d}; epoch_loss: {1:.4f}".format(e, epoch_loss))
            self.flush_extras(e, "train")

            _time = time.time() - start_time
            self._model.eval()
            eval_loss = self.evaluate_fn(e, _time)

            self._scalars["eval_loss"] = eval_loss
            print(
                "[INFO] ({}) Epoch {:04d}; eval_loss: {:.4f}; lr: {:.2e}".format(
                    thisname, e, eval_loss, self._params["learning_rate"]
                )
            )

            self.write_summary(e)
            model_path = os.path.join(
                self._params["model_prefix"], "models", "ckpt_epoch_%04d.pt" % e
            )
            if (e + 1) % 100 == 0:
                torch.save(self._model.state_dict(), model_path)

            self.update_learning_rate(e, mode="epochwise")
            self.flush_extras(e, "eval")

        # save the last one
        model_path = os.path.join(
            self._params["model_prefix"], "models", "ckpt_epoch_%04d.pt" % e
        )
        torch.save(self._model.state_dict().model_path)

    # Ignore this part. NO SUMMARY WILL BE WRITTEN!
    def write_summary(self, epoch):
        # for action_, ms_errors_ in ms_eval_loss.items():
        self._writer.add_scalars(
            "loss/recon_loss",
            {"train": self._scalars["epoch_loss"], "eval": self._scalars["eval_loss"]},
            epoch,
        )

    def update_learning_rate(self, epoch_step, mode="stepwise"):
        """Update learning rate handler updating only when the mode matches."""
        if self._params["lr_schedule_type"] == mode:
            self._lr_scheduler.step(epoch_step)
            self._writer.add_scalar(
                "learning_rate/lr", self._params["learning_rate"], epoch_step
            )
            self._lr_db_curve.append([self._params["learning_rate"], epoch_step])
            self._params["learning_rate"] = self._lr_scheduler.get_last_lr()[0]

    @abstractmethod
    def flush_extras(self, epoch, phase):
        pass

    # TODO: modify this evaluate function!!!
    @torch.no_grad()
    def evaluate_waymo(self, current_step, step_time):
        """Evaluation loop."""
        eval_loss = 0

        sample = next(iter(self._eval_dataset_fn))
        # for j, sample in enumerate(self._eval_dataset_fn):
        # for k in sample.keys():
        #     if (k=='decoder_outputs_euler') or (k=='actions'):
        #         continue
        #     sample[k] = sample[k].squeeze().to(_DEVICE)
        input_length = 30
        output_length = 20
        box_seq, hkp_seq = sample
        box_inputs = box_seq[:, :input_length, :, :2]
        box_outputs = box_seq[:, input_length:, :, :2]
        hkp_inputs = hkp_seq[:, :input_length, :, :]
        hkp_outputs = hkp_seq[:, input_length, :, :]

        decoder_pred = self._model(box_inputs)

        srnn_loss = self.compute_loss(
            inputs=box_inputs,
            target=box_outputs,
            preds=decoder_pred,
        )

        # [batch_size, sequence_length, pose_dim]
        decoder_pred = decoder_pred[0][-1]
        # [batch_size, sequence_length, pose_dim]
        msre_ = decoder_pred - sample["decoder_outputs"]
        # [batch_size, sequence_length]
        msre_ = torch.sqrt(torch.sum(msre_ * msre_, dim=-1))
        msre_ = msre_.mean().item()

        eval_loss = srnn_loss
        # run validation on different ranges
        mean_eval_error_dict = self.validation_srnn_ms(sample, decoder_pred)

        self._scalars["ms_eval_loss"] = mean_eval_error_dict
        self._scalars["msre"] = msre_

        return eval_loss

    # TODO: match the dataset_factory with waymodataset!
    def dataset_factory(params):
        """Defines the datasets that will be used for training and validation."""

        train_dataset = h36mdataset_fn.H36MDataset(params, mode="train")
        train_dataset_fn = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            num_workers=4,
            collate_fn=h36mdataset_fn.collate_fn,
            drop_last=True,
        )

        eval_dataset = h36mdataset_fn.H36MDataset(
            params, mode="eval", norm_stats=train_dataset._norm_stats
        )
        eval_dataset_fn = torch.utils.data.DataLoader(
            eval_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=True
        )

        return train_dataset_fn, eval_dataset_fn

    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_path", type=str, default=None)
        parser.add_argument("--action", type=str, default=None)
        parser.add_argument("--use_one_hot", action="store_true")
        parser.add_argument("--source_seq_len", type=int, default=50)
        parser.add_argument("--target_seq_len", type=int, default=25)
        # parser.add_argument('--input_size', type=int, default=55)
        parser.add_argument("--batch_size", type=int, default=16)
        parser.add_argument("--max_epochs", type=int, default=3000)
        parser.add_argument("--steps_per_epoch", type=int, default=200)
        parser.add_argument("--learning_rate", type=float, default=0.005)
        parser.add_argument("--optimizer_fn", type=str, default="adam")
        parser.add_argument("--warmup_epochs", type=int, default=30)
        parser.add_argument("--remove_low_std", action="store_true")
        args = parser.parse_args()

        params = vars(args)
        print(params)
        params["action_subset"] = [args.action]
        params["virtual_dataset_size"] = args.steps_per_epoch * args.batch_size

        train_dataset_fn, eval_dataset_fn = dataset_factory(params)

        model_fn = Seq2SeqModelFn(
            params, train_dataset_fn=train_dataset_fn, eval_dataset_fn=eval_dataset_fn
        )

        model_fn.train()
