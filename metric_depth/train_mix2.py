# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat
import uuid
import torch.distributed as dist
from datetime import datetime as dt
from zoedepth.utils.misc import count_parameters, parallelize
from zoedepth.utils.config import get_config
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.trainers.builder import get_trainer
from zoedepth.models.builder import build_model
from zoedepth.data.data_mono import MixedNYUKITTI, TransMixDataloader, DepthDataLoader
import torch.utils.data.distributed
import torch.multiprocessing as mp
from zoedepth.trainers.zoedepth_trainer import Trainer as zoedepth_trainer
from zoedepth.trainers.loss import TrimmedMaeLoss, SILogLoss
import torch.nn as nn
import wandb
import torch
import numpy as np
from pprint import pprint
import argparse
import os
import torch.cuda.amp as amp
from tqdm import tqdm
from zoedepth.utils.config import DATASETS_CONFIG
from zoedepth.utils.misc import RunningAverageDict, colorize, colors
from zoedepth.utils.config import flatten
import copy
import torch.optim as optim

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["WANDB_START_METHOD"] = "thread"

from zoedepth.trainers.loss import GradL1Loss

from zoedepth.models.model_io import load_wts

def fix_random_seed(seed: int):
    """
    Fix random seed for reproducibility

    Args:
        seed (int): random seed
    """
    import random

    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_ckpt(config, model, checkpoint_dir="./checkpoints", ckpt_type="best"):
    import glob
    import os

    from zoedepth.models.model_io import load_wts

    if hasattr(config, "checkpoint"):
        checkpoint = config.checkpoint
    elif hasattr(config, "ckpt_pattern"):
        pattern = config.ckpt_pattern
        matches = glob.glob(os.path.join(
            checkpoint_dir, f"*{pattern}*{ckpt_type}*"))
        if not (len(matches) > 0):
            raise ValueError(f"No matches found for the pattern {pattern}")

        checkpoint = matches[0]

    else:
        return model
    model = load_wts(model, checkpoint)
    print("Loaded weights from {0}".format(checkpoint))
    return model

def is_rank_zero(args):
    return args.rank == 0



class MixedTrainer(zoedepth_trainer):
    def __init__(self, config, model, train_loader1, train_loader2, device=None):
        self.device=device
        self.config=config
        self.model=model
        self.train_loader1=train_loader1
        self.train_loader2=train_loader2
        self.optimizer =self.init_optimizer()
        self.scheduler = self.init_scheduler()
        self.silog_loss=SILogLoss()
        self.grad_loss=GradL1Loss()
        self.ssi_loss=TrimmedMaeLoss()
        self.scaler = amp.GradScaler(enabled=self.config.use_amp)
    
    @property
    def iters_per_epoch(self):
        return len(self.train_loader2)+len(self.train_loader1)

    def init_scheduler(self):
        
        lrs = [l['lr'] for l in self.optimizer.param_groups]
        return optim.lr_scheduler.OneCycleLR(self.optimizer, lrs, epochs=self.config.epochs, steps_per_epoch=len(self.train_loader1)+len(self.train_loader2),
                                             cycle_momentum=self.config.cycle_momentum,
                                             base_momentum=0.85, max_momentum=0.95, div_factor=self.config.div_factor, final_div_factor=self.config.final_div_factor, pct_start=self.config.pct_start, three_phase=self.config.three_phase)


    
    def train_on_batch(self, batch, train_step):
        """
        Expects a batch of images and depth as input
        batch["image"].shape : batch_size, c, h, w
        batch["depth"].shape : batch_size, 1, h, w
        """

        images, depths_gt = batch['image'].to(
            self.device), batch['depth'].to(self.device)
        dataset = batch['dataset'][0]

        b, c, h, w = images.size()
        mask = batch["mask"].to(self.device).to(torch.bool)

        losses = {}

        with amp.autocast(enabled=self.config.use_amp):

            output = self.model(images)
            pred_depths = output['metric_depth']
    
            l_si, pred = self.silog_loss(
                pred_depths, depths_gt, mask=mask, interpolate=True, return_interpolated=True)
            loss = self.config.w_si * l_si
            losses[self.silog_loss.name] = l_si

            if self.config.w_grad > 0:
                # print(depths_gt.shape, mask.shape, pred.shape)
                l_grad = self.grad_loss(pred, depths_gt, mask=mask)
                
                loss = loss + self.config.w_grad * l_grad
                losses[self.grad_loss.name] = l_grad
            else:
                l_grad = torch.Tensor([0])

        self.scaler.scale(loss).backward()

        if self.config.clip_grad > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.clip_grad)

        self.scaler.step(self.optimizer)

        if self.should_log and (self.step % int(self.config.log_images_every * self.iters_per_epoch)) == 0:
            # -99 is treated as invalid depth in the log_images function and is colored grey.
            depths_gt[torch.logical_not(mask)] = -99

            self.log_images(rgb={"Input": images[0, ...]}, depth={"GT": depths_gt[0], "PredictedMono": pred[0]}, prefix="Train",
                            min_depth=DATASETS_CONFIG[dataset]['min_depth'], max_depth=DATASETS_CONFIG[dataset]['max_depth'])

            if self.config.get("log_rel", False):
                self.log_images(
                    scalar_field={"RelPred": output["relative_depth"][0]}, prefix="TrainRel")

        self.scaler.update()
        self.optimizer.zero_grad()

        return losses
    
    def train_on_batch_ssi(self, batch, train_step):
        """
        Expects a batch of images and depth as input
        batch["image"].shape : batch_size, c, h, w
        batch["depth"].shape : batch_size, 1, h, w
        """

        images, depths_gt = batch['image'].to(
            self.device), batch['depth'].to(self.device)
        dataset = batch['dataset'][0]

        b, c, h, w = images.size()
        mask = batch["mask"].to(self.device).to(torch.bool)

        losses = {}

        with amp.autocast(enabled=self.config.use_amp):

            output = self.model(images)
            pred_depths = output['metric_depth']
    
            l_si, pred = self.ssi_loss(
                pred_depths, depths_gt, mask=mask, interpolate=True, return_interpolated=True)
            loss = self.config.w_si * l_si
            losses[self.ssi_loss.name] = l_si

            if self.config.w_grad > 0:
                # print(depths_gt.shape, mask.shape, pred.shape)
                l_grad = self.grad_loss(pred, depths_gt, mask=mask)
                
                loss = loss + self.config.w_grad * l_grad
                losses[self.grad_loss.name] = l_grad
            else:
                l_grad = torch.Tensor([0])

        self.scaler.scale(loss).backward()

        if self.config.clip_grad > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.clip_grad)

        self.scaler.step(self.optimizer)

        if self.should_log and (self.step % int(self.config.log_images_every * self.iters_per_epoch)) == 0:
            # -99 is treated as invalid depth in the log_images function and is colored grey.
            depths_gt[torch.logical_not(mask)] = -99

            self.log_images(rgb={"Input": images[0, ...]}, depth={"GT": depths_gt[0], "PredictedMono": pred[0]}, prefix="Train",
                            min_depth=DATASETS_CONFIG[dataset]['min_depth'], max_depth=DATASETS_CONFIG[dataset]['max_depth'])

            if self.config.get("log_rel", False):
                self.log_images(
                    scalar_field={"RelPred": output["relative_depth"][0]}, prefix="TrainRel")

        self.scaler.update()
        self.optimizer.zero_grad()

        return losses

    def train(self):
        print(f"Training {self.config.name}")
        if self.config.uid is None:
            self.config.uid = str(uuid.uuid4()).split('-')[-1]
        run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-{self.config.uid}"
        self.config.run_id = run_id
        self.config.experiment_id = f"{self.config.name}{self.config.version_name}_{run_id}"
        self.should_write = ((not self.config.distributed)
                             or self.config.rank == 0)
        self.should_log = self.should_write  # and logging
        if self.should_log:
            tags = self.config.tags.split(
                ',') if self.config.tags != '' else None
            wandb.init(project=self.config.project, name=self.config.experiment_id, config=flatten(self.config), dir=self.config.root,
                       tags=tags, notes=self.config.notes, settings=wandb.Settings(start_method="fork"))

        self.model.train()
        self.step = 0

        if self.config.prefetch:
            for i, batch in tqdm(enumerate(self.train_loader), desc=f"Prefetching...",
                                 total=self.iters_per_epoch) if is_rank_zero(self.config) else enumerate(self.train_loader):
                pass

        losses = {}
        def stringify_losses(L): return "; ".join(map(
            lambda kv: f"{colors.fg.purple}{kv[0]}{colors.reset}: {round(kv[1].item(),3):.4e}", L.items()))
        for epoch in range(self.config.epochs):
            if self.should_early_stop():
                break
            self.epoch = epoch
            ################################# Train loop ##########################################################
            if self.should_log:
                wandb.log({"Epoch": epoch}, step=self.step)

            pbar2 = tqdm(enumerate(self.train_loader2), desc=f"Epoch: {epoch + 1}/{self.config.epochs}. Loop: Train",
                        total=self.iters_per_epoch) if is_rank_zero(self.config) else enumerate(self.train_loader2)
            for i, batch in pbar2:
                try:
                    if self.should_early_stop():
                        print("Early stopping")
                        break
                    losses = self.train_on_batch(batch, i)
                    self.raise_if_nan(losses)
                    if is_rank_zero(self.config) and self.config.print_losses:
                        pbar2.set_description(
                            f"Epoch: {epoch + 1}/{self.config.epochs}. Loop: Train. Losses: {stringify_losses(losses)}")
                    self.scheduler.step()

                    if self.should_log and self.step % 50 == 0:
                        wandb.log({f"Train/{name}": loss.item()
                                for name, loss in losses.items()}, step=self.step)
                    self.step += 1 
                    del losses, batch
                    torch.cuda.empty_cache()             
                except Exception as e:
                    del losses, batch
                    torch.cuda.empty_cache()  
                    print(f"Error: {e}")
                except:
                    del losses, batch
                    torch.cuda.empty_cache()  
                    print("Unknown error")    

            if epoch % 20 != 1:
                continue
            pbar1 = tqdm(enumerate(self.train_loader1), desc=f"Epoch: {epoch + 1}/{self.config.epochs}. Loop: Train",
                        total=self.iters_per_epoch) if is_rank_zero(self.config) else enumerate(self.train_loader1)
            for i, batch in pbar1:
                try:
                    if self.should_early_stop():
                        print("Early stopping")
                        break
                    losses = self.train_on_batch_ssi(batch, i)
                    self.raise_if_nan(losses)
                    if is_rank_zero(self.config) and self.config.print_losses:
                        pbar1.set_description(
                            f"Epoch: {epoch + 1}/{self.config.epochs}. Loop: Train. Losses: {stringify_losses(losses)}")
                    self.scheduler.step()

                    if self.should_log and self.step % 50 == 0:
                        wandb.log({f"Train/{name}": loss.item()
                                for name, loss in losses.items()}, step=self.step)
                    self.step += 1     
                    del losses, batch
                    torch.cuda.empty_cache()       
                except Exception as e:

                    print(f"Error: {e}")
                except: 
                    print("Unknown error")
            
            self.save_checkpoint("scale_opaque_ssi_transparent.pt")
        self.step += 1

def main_worker(gpu, ngpus_per_node, config):
    try:
        fix_random_seed(43)

        config.gpu = gpu
        config.aug=True

        model = build_model(config)
        

        model = load_ckpt(config, model)
        pretrained= "./depth_anything_finetune/mixed.pt"
        model = load_wts(model, pretrained)
        model = parallelize(config, model)

        total_params = f"{round(count_parameters(model)/1e6,2)}M"
        config.total_params = total_params
        print(f"Total parameters : {total_params}")
        print(config.dataset)
        opaque_config= copy.deepcopy(config)
        
        opaque_config.dataset="nyu"
        print(opaque_config.dataset)
        train_loader_transparent = TransMixDataloader(config, "train").data
        train_loader_opaque = DepthDataLoader(opaque_config, "train").data

        trainer = MixedTrainer(config, model,train_loader_transparent, train_loader_opaque,  device=gpu)
        trainer.train()

        
    finally:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    mp.set_start_method('forkserver')

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="synunet")
    parser.add_argument("-d", "--dataset", type=str, default='mix')
    parser.add_argument("--trainer", type=str, default=None)

    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    overwrite_kwargs["model"] = args.model
    if args.trainer is not None:
        overwrite_kwargs["trainer"] = args.trainer

    config = get_config(args.model, "train", args.dataset, **overwrite_kwargs)
    # git_commit()
    if config.use_shared_dict:
        shared_dict = mp.Manager().dict()
    else:
        shared_dict = None
    config.shared_dict = shared_dict
    config.use_lora = True
    config.batch_size = config.bs
    config.mode = 'train'
    if config.root != "." and not os.path.isdir(config.root):
        os.makedirs(config.root)

    try:
        node_str = os.environ['SLURM_JOB_NODELIST'].replace(
            '[', '').replace(']', '')
        nodes = node_str.split(',')

        config.world_size = len(nodes)
        config.rank = int(os.environ['SLURM_PROCID'])
        # config.save_dir = "/ibex/scratch/bhatsf/videodepth/checkpoints"

    except KeyError as e:
        # We are NOT using SLURM
        config.world_size = 1
        config.rank = 0
        nodes = ["127.0.0.1"]

    if config.distributed:

        print(config.rank)
        port = np.random.randint(15000, 15025)
        config.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        print(config.dist_url)
        config.dist_backend = 'nccl'
        config.gpu = None

    ngpus_per_node = torch.cuda.device_count()
    config.num_workers = config.workers
    config.distributed=False
    config.nproc_per_node = 1
    config.ngpus_per_node = ngpus_per_node
    print("Config:")
    pprint(config)
    if config.distributed:
        config.world_size = ngpus_per_node * config.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, config))
    else:
        if ngpus_per_node == 1:
            config.gpu = 0
        main_worker(config.gpu, ngpus_per_node, config)
