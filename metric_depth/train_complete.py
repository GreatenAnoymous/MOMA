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

from zoedepth.utils.misc import count_parameters, parallelize
from zoedepth.utils.config import get_config
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.trainers.builder import get_trainer
from zoedepth.models.builder import build_model
from zoedepth.data.data_mono import DepthDataLoader
import torch.utils.data.distributed
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import numpy as np
import torch.cuda.amp as amp
from pprint import pprint
import argparse
from torchvision import transforms
import os
from zoedepth.utils.misc import compute_metrics
from zoedepth.trainers.zoedepth_trainer import  Trainer
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["WANDB_START_METHOD"] = "thread"

from zoedepth.utils.config import DATASETS_CONFIG
class DepthCompleteTrainer(Trainer):
    def __init__(self, config, model, train_loader, test_loader=None, device=None):
        super().__init__(config, model, train_loader,
                         test_loader=test_loader, device=device)

    def train_on_batch(self, batch, train_step):
        """
        Expects a batch of images and depth as input
        batch["image"].shape : batch_size, c, h, w
        batch["depth"].shape : batch_size, 1, h, w
        """

        images, depths_gt, depths_raw = batch['image'].to(
            self.device), batch['depth'].to(self.device), batch["depth_raw"].to(self.device)
        dataset = batch['dataset'][0]
        images=images.permute(0,3,1,2)
        depths_raw=depths_raw.permute(0,3,1,2)
        depths_gt=depths_gt.permute(0,3,1,2)
        b, c, h, w = images.size()
        mask = batch["mask"].to(self.device).to(torch.bool)

        losses = {}

        with amp.autocast(enabled=self.config.use_amp):

            output = self.model(images, depths_raw)
            pred_depths = output['metric_depth']
            if mask.any()==False:
                # Code to be executed if mask has True values
                # ...
                raise ValueError("mask has no True values")

            if torch.isnan(pred_depths).any():
                raise ValueError("pred_depths contain NaN values")
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
                            min_depth=torch.min(pred_depths[0]).item(), max_depth=torch.max(pred_depths[0]).item())

            if self.config.get("log_rel", False):
                self.log_images(
                    scalar_field={"RelPred": output["relative_depth"][0]}, prefix="TrainRel")

        self.scaler.update()
        self.optimizer.zero_grad()

        return losses
    
    @torch.no_grad()
    def eval_infer(self, x, y):
        with amp.autocast(enabled=self.config.use_amp):
           
            m = self.model.module if self.config.multigpu else self.model
            pred_depths = m(x, y)['metric_depth']
        return pred_depths




    def validate_on_batch(self, batch, val_step):
        # print(batch["image"].shape, batch["depth"].shape, batch["mask"].shape)
        images = batch['image'].to(self.device)
        depths_gt = batch['depth'].to(self.device)
        depths_raw=batch["depth_raw"].to(self.device)
        dataset = batch['dataset'][0]
        mask = batch["mask"].to(self.device)
        # if 'has_valid_depth' in batch:
        #     if not batch['has_valid_depth']:
        #         # print("no valid depth")
        #         return None, None
        images=images.permute(0,3,1,2)
        depths_raw=depths_raw.permute(0,3,1,2)
        depths_gt=depths_gt.permute(0,3,1,2)
     


        if dataset == 'nyu':
            pred_depths = self.crop_aware_infer(images)
        else:
            pred_depths = self.eval_infer(images, depths_raw)

  

        if not torch.any(mask):
            raise ValueError("validation mask has no True values")

        with amp.autocast(enabled=self.config.use_amp):
            l_depth = self.silog_loss(
                pred_depths, depths_gt, mask=mask.to(torch.bool), interpolate=True)
    
        metrics = compute_metrics(depths_gt, pred_depths, **self.config)
    
        losses = {f"{self.silog_loss.name}": l_depth.item()}
    

        if val_step == 1 and self.should_log:
            depths_gt[torch.logical_not(mask)] = -99
            
            # print("predicted depth shape", pred_depths[0].shape, pred_depths[0].max(), pred_depths[0].min(), pred_depths[0])
            self.log_images(rgb={"Input": images[0]}, depth={"GT": depths_gt[0], "PredictedMono": pred_depths[0]}, prefix="Test",
                            min_depth=torch.min(pred_depths[0]).item(), max_depth=torch.max(pred_depths[0]).item())

        return metrics, losses





def fix_random_seed(seed: int):
    import random

    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


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


def main_worker(gpu, ngpus_per_node, config):
    try:
        seed = config.seed if 'seed' in config and config.seed else 43
        fix_random_seed(seed)

        config.gpu = gpu
        config.train_midas=True

        model = build_model(config)

            
        model = load_ckpt(config, model)
        model = parallelize(config, model)

        if next(model.parameters()).is_cuda:
            print("Model is in CUDA")
  
        total_params = f"{round(count_parameters(model)/1e6,2)}M"
        config.total_params = total_params
        print(f"Total parameters : {total_params}", config.gpu)
        

        train_loader = DepthDataLoader(config, "train").data
        # train_loader = DepthDataLoader(config, "online_eval").data
        test_loader = DepthDataLoader(config, "online_eval").data

        # trainer = get_trainer(config)(
        #     config, model, train_loader, test_loader, device=config.gpu)
        trainer=DepthCompleteTrainer(config, model, train_loader, test_loader, device=config.gpu)

        trainer.train()
        # torch.save(model.state_dict(), 'model_weights.pth')
    finally:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    mp.set_start_method('forkserver')

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="damc")
    parser.add_argument("-d", "--dataset", type=str, default='nyu')
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
    config.dataset="depth_complete"
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
    config.ngpus_per_node = ngpus_per_node
    config.nproc_per_node = 1
    print("Config:")
    pprint(config)
    if config.distributed:
        config.world_size = ngpus_per_node * config.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, config))
    else:
        if ngpus_per_node == 1:
            config.gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        main_worker(config.gpu, ngpus_per_node, config)
