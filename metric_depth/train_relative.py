from zoedepth.utils.misc import count_parameters, parallelize, compute_metrics, compute_ssi_metrics
from zoedepth.trainers.base_trainer import BaseTrainer
from zoedepth.trainers.loss import ScaleAndShiftInvariantLoss, TrimmedMaeLoss
from zoedepth.data.data_mono import DepthDataLoader
from zoedepth.utils.config import get_config
import torch.utils.data.distributed
from zoedepth.utils.arg_utils import parse_unknown
import torch.multiprocessing as mp
import torch
import numpy as np
import torch.cuda.amp as amp
import argparse
import os
from pprint import pprint
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from zoedepth.data.preprocess import get_black_border
from zoedepth.utils.config import DATASETS_CONFIG
from zoedepth.models.base_models.depth_anything_lora import DepthAnythingLoraCore
from zoedepth.trainers.loss import GradL1Loss

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


class RelativeTrainer(BaseTrainer):
    def __init__(self, config, model, train_loader, test_loader=None, device=None):
        super().__init__(config, model, train_loader,
                         test_loader=test_loader, device=device)
        self.device = device
        self.model=model.to(device)
        self.ssi_loss = ScaleAndShiftInvariantLoss()
        self.scaler = amp.GradScaler(enabled=self.config.use_amp)
        self.grad_loss=GradL1Loss()
        self.epoch=0
        self.should_log=False
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

        # assert next(self.model.parameters()).is_cuda
        # assert images.device == self.device, f"Image device {images.device} does not match model device {self.device}"
        losses = {}

        with amp.autocast(enabled=self.config.use_amp):
        
        
            pred_depths= self.model(images, denorm=False, return_rel_depth=True)
            pred_depths=pred_depths.unsqueeze(1)
            pred_depths = nn.functional.interpolate(pred_depths, size=(h, w), mode='bilinear', align_corners=False)

            loss=self.config.w_si * self.ssi_loss(pred_depths, depths_gt, mask=mask)
            losses[self.ssi_loss.name] = loss

            if self.config.w_grad > 0:
                l_grad = self.grad_loss(pred_depths, depths_gt, mask=mask)
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
        # if self.should_log and (self.step % int(self.config.log_images_every * self.iters_per_epoch)) == 0:
        #     # -99 is treated as invalid depth in the log_images function and is colored grey.
        #     depths_gt[torch.logical_not(mask)] = -99

        #     self.log_images(rgb={"Input": images[0, ...]}, depth={"GT": depths_gt[0], "PredictedMono": pred_depths[0]}, prefix="Train",
        #                     min_depth=DATASETS_CONFIG[dataset]['min_depth'], max_depth=DATASETS_CONFIG[dataset]['max_depth'])

        #     if self.config.get("log_rel", False):
        #         self.log_images(
        #             scalar_field={"RelPred": output["relative_depth"][0]}, prefix="TrainRel")
        self.scaler.update()
        self.optimizer.zero_grad()

        return losses
    
    @torch.no_grad()
    def eval_infer(self, x):
        with amp.autocast(enabled=self.config.use_amp):
            b, c, h, w =x.size()
            # m = self.model.module if self.config.multigpu else self.model
            # pred_depths = m(x)['metric_depth']
            pred_depths= self.model(x, denorm=False, return_rel_depth=True)
            pred_depths=pred_depths.unsqueeze(1)
            pred_depths = nn.functional.interpolate(pred_depths, size=(h, w), mode='bilinear', align_corners=False)

        return pred_depths

    @torch.no_grad()
    def crop_aware_infer(self, x):
        print("Cropping the image to avoid black border")
        # if we are not avoiding the black border, we can just use the normal inference
        if not self.config.get("avoid_boundary", False):
            return self.eval_infer(x)
        
        # otherwise, we need to crop the image to avoid the black border
        # For now, this may be a bit slow due to converting to numpy and back
        # We assume no normalization is done on the input image

        # get the black border
        assert x.shape[0] == 1, "Only batch size 1 is supported for now"
        x_pil = transforms.ToPILImage()(x[0].cpu())
        x_np = np.array(x_pil, dtype=np.uint8)
        black_border_params = get_black_border(x_np)
        top, bottom, left, right = black_border_params.top, black_border_params.bottom, black_border_params.left, black_border_params.right
        x_np_cropped = x_np[top:bottom, left:right, :]
        x_cropped = transforms.ToTensor()(Image.fromarray(x_np_cropped))

        # run inference on the cropped image
        pred_depths_cropped = self.eval_infer(x_cropped.unsqueeze(0).to(self.device))

        # resize the prediction to x_np_cropped's size
        pred_depths_cropped = nn.functional.interpolate(
            pred_depths_cropped, size=(x_np_cropped.shape[0], x_np_cropped.shape[1]), mode="bilinear", align_corners=False)
        

        # pad the prediction back to the original size
        pred_depths = torch.zeros((1, 1, x_np.shape[0], x_np.shape[1]), device=pred_depths_cropped.device, dtype=pred_depths_cropped.dtype)
        pred_depths[:, :, top:bottom, left:right] = pred_depths_cropped

        return pred_depths



    def validate_on_batch(self, batch, val_step):
        images = batch['image'].to(self.device)
        depths_gt = batch['depth'].to(self.device)
        dataset = batch['dataset'][0]
        mask = batch["mask"].to(self.device)
        if 'has_valid_depth' in batch:
            if not batch['has_valid_depth']:
                return None, None

        depths_gt = depths_gt.squeeze().unsqueeze(0).unsqueeze(0)
        mask = mask.squeeze().unsqueeze(0).unsqueeze(0)
        # if dataset == 'nyu':
        #     pred_depths = self.crop_aware_infer(images)
        # else:
        pred_depths = self.eval_infer(images)
        pred_depths = pred_depths.squeeze().unsqueeze(0).unsqueeze(0)
        
        with amp.autocast(enabled=self.config.use_amp):
            l_depth = self.ssi_loss(
                pred_depths, depths_gt, mask=mask.to(torch.bool))

        metrics = compute_ssi_metrics(depths_gt, pred_depths, **self.config)
        losses = {f"{self.ssi_loss.name}": l_depth.item()}

        if val_step == 1 and self.should_log:
            depths_gt[torch.logical_not(mask)] = -99
            self.log_images(rgb={"Input": images[0]}, depth={"GT": depths_gt[0], "PredictedMono": pred_depths[0]}, prefix="Test",
                            min_depth=DATASETS_CONFIG[dataset]['min_depth'], max_depth=DATASETS_CONFIG[dataset]['max_depth'])
        print(metrics)
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


def main_worker(gpu,config):
    try:
        seed = config.seed if 'seed' in config and config.seed else 43
        fix_random_seed(seed)
        config.gpu=gpu
        model=DepthAnythingLoraCore.build(**config)
        model=parallelize(config,model)
        print(len([p for p in model.parameters() if p.requires_grad]), "Total Learnable Parameters")
        
        total_params = f"{round(count_parameters(model)/1e6,2)}M"
        config.total_params = total_params
        print(f"Total parameters : {total_params}", config.gpu)
        
        train_loader = DepthDataLoader(config, "train").data
        test_loader = DepthDataLoader(config, "online_eval").data

        trainer = RelativeTrainer(config, model, train_loader, test_loader=test_loader, device=config.gpu)
        # trainer.train()
        trainer.validate()
        
    finally:
        import wandb
        wandb.finish()


def test_dam():
    from PIL import Image
    import cv2
    import torch.nn.functional as F

    image= cv2.imread("../000.jpg")
    # depth=np.array(Image.open("../../object_dataset/object_dataset_14/17_gt_depth.png"))
    image=torch.tensor(np.array(image))
    image=image.permute(2,0,1)
    image=image.unsqueeze(0).float()
    image=image/255.0
    b,c,w,h=image.shape
    print(image.shape,"000 shape")
    model=DepthAnythingLoraCore.build()
    
    output=model(image, denorm=False, return_rel_depth=True)
    output=1/output
    print("output shape",output.shape)
    depth=F.interpolate(output.unsqueeze(0), size=(w, h), mode='bilinear', align_corners=False)
    print(output.shape, depth.shape)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        
    depth = depth.cpu().numpy().astype(np.uint8)
    depth=depth.squeeze()
    depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    print(depth.shape)
    cv2.imwrite("output_colored.png", depth)


def main_func():
    mp.set_start_method('forkserver')

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="synunet")
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

    config.batch_size = config.bs
    config.mode = 'train'
    config.multigpu=False
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
    config.distributed=False
    print("Config:")
    pprint(config)
    if config.distributed:
        config.world_size = ngpus_per_node * config.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, config))
    else:
        if ngpus_per_node == 1:
            print("Using single GPU")
            config.gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # config.gpu=torch.device("cpu")
        print(config.gpu,"the gpu")   
        main_worker(config.gpu,  config)


if __name__ == '__main__':
    # test_dam()
    main_func()
    
