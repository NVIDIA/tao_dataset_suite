# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MAL/blob/main/LICENSE

"""MAL model."""
import itertools
import json
import os

import cv2
import numpy as np

from pycocotools.coco import COCO
from pycocotools.mask import encode
from pycocotools.cocoeval import COCOeval

from mmcv.cnn import ConvModule

import torchmetrics
import pytorch_lightning as pl

from fairscale.nn import auto_wrap

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nvidia_tao_ds.auto_label.datasets.data_aug import Denormalize
from nvidia_tao_ds.auto_label.models import vit_builder
from nvidia_tao_ds.auto_label.utils.optimizers.adamw import AdamWwStep
from nvidia_tao_ds.auto_label.utils.lr_schedulers.cosine_lr import adjust_learning_rate


class MeanField(nn.Module):
    """Mean Field approximation to refine mask."""

    def __init__(self, cfg=None):
        """Init."""
        super().__init__()
        self.kernel_size = cfg.train.crf_kernel_size
        assert self.kernel_size % 2 == 1
        self.zeta = cfg.train.crf_zeta
        self.num_iter = cfg.train.crf_num_iter
        self.high_thres = cfg.train.crf_value_high_thres
        self.low_thres = cfg.train.crf_value_low_thres
        self.cfg = cfg

    def trunc(self, seg):
        """Clamp mask values by crf_value_(low/high)_thres."""
        seg = torch.clamp(seg, min=self.low_thres, max=self.high_thres)
        return seg

    @torch.no_grad()
    def forward(self, feature_map, seg, targets=None):
        """Forward pass with num_iter."""
        feature_map = feature_map.float()
        kernel_size = self.kernel_size
        B, H, W = seg.shape
        C = feature_map.shape[1]

        self.unfold = torch.nn.Unfold(kernel_size, stride=1, padding=self.kernel_size // 2)
        # feature_map [B, C, H, W]
        feature_map = feature_map + 10
        # unfold_feature_map [B, C, kernel_size ** 2, H*W]
        unfold_feature_map = self.unfold(feature_map).reshape(B, C, kernel_size**2, H * W)
        # B, kernel_size**2, H*W
        kernel = torch.exp(-(((unfold_feature_map - feature_map.reshape(B, C, 1, H * W)) ** 2) / (2 * self.zeta ** 2)).sum(1))

        if targets is not None:
            t = targets.reshape(B, H, W)
            seg = seg * t
        else:
            t = None

        seg = self.trunc(seg)

        for it in range(self.num_iter):
            seg = self.single_forward(seg, kernel, t, B, H, W, it)

        return (seg > 0.5).float()

    def single_forward(self, x, kernel, targets, B, H, W, it):
        """Forward pass."""
        x = x[:, None]
        # x [B 2 H W]
        B, _, H, W = x.shape
        x = torch.cat([1 - x, x], 1)
        kernel_size = self.kernel_size
        # unfold_x [B, 2, kernel_size**2, H * W]
        # kernel   [B,    kennel_size**2, H * W]
        unfold_x = self.unfold(-torch.log(x)).reshape(B, 2, kernel_size ** 2, H * W)
        # aggre, x [B, 2, H * W]
        aggre = (unfold_x * kernel[:, None]).sum(2)
        aggre = torch.exp(-aggre)
        if targets is not None:
            aggre[:, 1:] = aggre[:, 1:] * targets.reshape(B, 1, H * W)
        out = aggre
        out = out / (1e-6 + out.sum(1, keepdim=True))
        out = self.trunc(out)
        return out[:, 1].reshape(B, H, W)


class MaskHead(nn.Module):
    """Mask Head."""

    def __init__(self, in_channels=2048, cfg=None):
        """Init."""
        super().__init__()
        self.num_convs = cfg.model.mask_head_num_convs
        self.in_channels = in_channels
        self.mask_head_hidden_channel = cfg.model.mask_head_hidden_channel
        self.mask_head_out_channel = cfg.model.mask_head_out_channel
        self.mask_scale_ratio = cfg.model.mask_scale_ratio

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = self.in_channels if i == 0 else self.mask_head_hidden_channel
            out_channels = self.mask_head_hidden_channel if i < self.num_convs - 1 else self.mask_head_out_channel
            self.convs.append(ConvModule(in_channels, out_channels, 3, padding=1))

    def forward(self, x):
        """Forward pass."""
        for idx, conv in enumerate(self.convs):
            if idx == 3:
                h, w = x.shape[2:]
                th, tw = int(h * self.mask_scale_ratio), int(w * self.mask_scale_ratio)
                x = F.interpolate(x, (th, tw), mode='bilinear', align_corners=False)
            x = conv(x)
        return x


class RoIHead(nn.Module):
    """RoI Head."""

    def __init__(self, in_channels=2048, cfg=None):
        """Init."""
        super().__init__()
        self.mlp1 = nn.Linear(in_channels, cfg.model.mask_head_out_channel)
        self.relu = nn.ReLU()
        self.mlp2 = nn.Linear(cfg.model.mask_head_out_channel, cfg.model.mask_head_out_channel)

    def forward(self, x, boxmask=None):
        """Forward pass."""
        x = x.mean((2, 3))
        x = self.mlp2(self.relu(self.mlp1(x)))
        return x


class MALStudentNetwork(pl.LightningModule):
    """MAL student model."""

    def __init__(self, in_channels=2048, cfg=None):
        """Init."""
        super().__init__()
        self.cfg = cfg
        self.backbone = vit_builder.build_model(cfg=cfg)
        # Load pretrained weights
        if cfg.checkpoint:
            print('Loading pretrained weights.....')
            state_dict = torch.load(cfg.checkpoint)
            if 'state_dict' in state_dict.keys():
                state_dict = state_dict['state_dict']
            if 'model' in state_dict.keys():
                state_dict = state_dict['model']
            self.backbone.load_state_dict(state_dict, strict=False)
        # K head
        self.roi_head = RoIHead(in_channels, cfg=cfg)
        # V head
        self.mask_head = MaskHead(in_channels, cfg=cfg)
        # make student sharded on multiple gpus
        self.configure_sharded_model()

    def configure_sharded_model(self):
        """Sharded backbone."""
        self.backbone = auto_wrap(self.backbone)

    def forward(self, x, boxmask, bboxes):
        """Forward pass."""
        if self.cfg.train.use_amp:
            x = x.half()
        feat = self.backbone.base_forward(x)
        spatial_feat_ori = self.backbone.get_spatial_feat(feat)
        h, w = spatial_feat_ori.shape[2:]
        mask_scale_ratio_pre = int(self.cfg.model.mask_scale_ratio_pre)
        if not self.cfg.model.not_adjust_scale:
            spatial_feat_list = []
            masking_list = []
            areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
            for idx, (scale_low, scale_high) in enumerate([(0, 32**2), (32**2, 96**2), (96**2, 1e5**2)]):
                masking = (areas < scale_high) * (areas > scale_low)
                if masking.sum() > 0:
                    spatial_feat = F.interpolate(
                        spatial_feat_ori[masking],
                        size=(int(h * 2**(idx - 1)), int(w * 2**(idx - 1))),
                        mode='bilinear', align_corners=False)
                    boxmask = None
                else:
                    spatial_feat = None
                    boxmask = None
                spatial_feat_list.append(spatial_feat)
                masking_list.append(masking)
            roi_feat = self.roi_head(spatial_feat_ori)
            n, maxh, maxw = roi_feat.shape[0], h * 4, w * 4

            seg_all = torch.zeros(n, 1, maxh, maxw).to(roi_feat)
            for idx, (spatial_feat, masking) in enumerate(zip(spatial_feat_list, masking_list)):
                if masking.sum() > 0:
                    mn = masking.sum()
                    mh, mw = int(h * mask_scale_ratio_pre * 2**(idx - 1)), int(w * mask_scale_ratio_pre * 2**(idx - 1))
                    seg_feat = self.mask_head(spatial_feat)
                    c = seg_feat.shape[1]
                    masked_roi_feat = roi_feat[masking]
                    seg = (masked_roi_feat[:, None, :] @ seg_feat.reshape(mn, c, mh * mw * 4)).reshape(mn, 1, mh * 2, mw * 2)
                    seg = F.interpolate(seg, size=(maxh, maxw), mode='bilinear', align_corners=False)
                    seg_all[masking] = seg
            ret_vals = {'feat': feat, 'seg': seg_all, 'spatial_feat': spatial_feat_ori, 'masking_list': masking_list}
        else:
            spatial_feat = F.interpolate(
                spatial_feat_ori, size=(int(h * mask_scale_ratio_pre), int(w * mask_scale_ratio_pre)),
                mode='bilinear', align_corners=False)
            boxmask = F.interpolate(boxmask, size=spatial_feat.shape[2:], mode='bilinear', align_corners=False)
            seg_feat = self.mask_head(spatial_feat)
            roi_feat = self.roi_head(spatial_feat_ori, boxmask)
            n, c, h, w = seg_feat.shape
            seg = (roi_feat[:, None, :] @ seg_feat.reshape(n, c, h * w)).reshape(n, 1, h, w)
            seg = F.interpolate(seg, (h * 4, w * 4), mode='bilinear', align_corners=False)
            ret_vals = {'feat': feat, 'seg': seg, 'spatial_feat': spatial_feat_ori}
        return ret_vals


class MALTeacherNetwork(MALStudentNetwork):
    """MAL teacher model."""

    def __init__(self, in_channels, cfg=None):
        """Init."""
        super().__init__(in_channels, cfg=cfg)
        self.eval()
        self.momentum = cfg.model.teacher_momentum

    @torch.no_grad()
    def update(self, student):
        """Update EMA teacher model."""
        for param_student, param_teacher in zip(student.parameters(), self.parameters()):
            param_teacher.data = param_teacher.data * self.momentum + param_student.data * (1 - self.momentum)


class MIoUMetrics(torchmetrics.Metric):
    """MIoU Metrics."""

    def __init__(self, dist_sync_on_step=True, num_classes=20):
        """Init."""
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("cnt", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, label, iou):
        """Update."""
        self.cnt[label - 1] += 1
        self.total[label - 1] += iou

    def update_with_ious(self, labels, ious):
        """Update with IOUs."""
        for iou, label in zip(ious, labels):
            self.cnt[label - 1] += 1
            self.total[label - 1] += float(iou)
        return ious

    def cal_intersection(self, seg, gt):
        """Calcuate mask intersection."""
        B = seg.shape[0]
        inter_cnt = (seg * gt).reshape(B, -1).sum(1)
        return inter_cnt

    def cal_union(self, seg, gt, inter_cnt=None):
        """Calculate mask union."""
        B = seg.shape[0]
        if inter_cnt is None:
            inter_cnt = self.cal_intersection(seg, gt)
        union_cnt = seg.reshape(B, -1).sum(1) + gt.reshape(B, -1).sum(1) - inter_cnt
        return union_cnt

    def cal_iou(self, seg, gt):
        """Calculate mask IOU."""
        inter_cnt = self.cal_intersection(seg, gt)
        union_cnt = self.cal_union(seg, gt, inter_cnt)
        return 1.0 * inter_cnt / (union_cnt + 1e-6)

    def compute(self):
        """Compute mIOU."""
        mIoUs = self.total / (1e-6 + self.cnt)
        mIoU = mIoUs.sum() / (self.cnt > 0).sum()
        return mIoU

    def compute_with_ids(self, ids=None):
        """Compute mIOU with IDs."""
        if ids is not None:
            total = self.total[torch.tensor(np.array(ids)).long()]
            cnt = self.cnt[torch.tensor(np.array(ids)).long()]
        else:
            total = self.total
            cnt = self.cnt
        mIoUs = total / (1e-6 + cnt)
        mIoU = mIoUs.sum() / (cnt > 0).sum()
        return mIoU


class MAL(pl.LightningModule):
    """Base MAL model."""

    def __init__(self, cfg=None, num_iter_per_epoch=None, categories=None):
        """Init."""
        super().__init__()
        # loss term hyper parameters
        self.num_convs = cfg.model.mask_head_num_convs
        self.loss_mil_weight = cfg.train.loss_mil_weight
        self.loss_crf_weight = cfg.train.loss_crf_weight
        self.loss_crf_step = cfg.train.loss_crf_step
        self.cfg = cfg
        self.mask_thres = cfg.train.mask_thres
        self.num_classes = len(categories) + 1

        self.mIoUMetric = MIoUMetrics(num_classes=self.num_classes)
        self.areaMIoUMetrics = nn.ModuleList([MIoUMetrics(num_classes=self.num_classes) for _ in range(3)])
        if self.cfg.evaluate.comp_clustering:
            self.clusteringScoreMetrics = torchmetrics.MeanMetric()

        backbone_type = cfg.model.arch  # TODO(@yuw): arch options?
        self.categories = categories

        if 'tiny' in backbone_type.lower():
            in_channel = 192
        if 'small' in backbone_type.lower():
            in_channel = 384
        elif 'base' in backbone_type.lower():
            in_channel = 768
        elif 'large' in backbone_type.lower():
            in_channel = 1024
        elif 'huge' in backbone_type.lower():
            in_channel = 1280
        elif 'fan' in backbone_type.lower():
            in_channel = 448

        self.mean_field = MeanField(cfg=self.cfg)
        self.student = MALStudentNetwork(in_channel, cfg=cfg)
        self.teacher = MALTeacherNetwork(in_channel, cfg=cfg)
        self.denormalize = Denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self._optim_type = cfg.train.optim_type
        self._lr = cfg.train.lr
        self._wd = cfg.train.wd
        self._momentum = cfg.train.optim_momentum
        if num_iter_per_epoch is not None:
            self._num_iter_per_epoch = num_iter_per_epoch // len(self.cfg.gpu_ids)
        self.cfg = cfg
        self.vis_cnt = 0
        self.local_step = 0
        # Enable manual optimization
        self.automatic_optimization = False

    def configure_optimizers(self):
        """Configure optimizers."""
        optimizer = AdamWwStep(
            self.parameters(), eps=self.cfg.train.optim_eps,
            betas=self.cfg.train.optim_betas,
            lr=self._lr, weight_decay=self._wd)
        return optimizer

    def crf_loss(self, img, seg, tseg, boxmask):
        """CRF loss."""
        refined_mask = self.mean_field(img, tseg, targets=boxmask)
        return self.dice_loss(seg, refined_mask).mean(), refined_mask

    def dice_loss(self, pred, target):
        """DICE loss.

        replace cross-entropy like loss in the original paper:
        (https://papers.nips.cc/paper/2019/file/e6e713296627dff6475085cc6a224464-Paper.pdf).

        Args:
            pred (torch.Tensor): [B, embed_dim]
            target (torch.Tensor): [B, embed_dim]
        Return:
            loss (torch.Tensor): [B]
        """
        pred = pred.contiguous().view(pred.size()[0], -1).float()
        target = target.contiguous().view(target.size()[0], -1).float()

        a = torch.sum(pred * target, 1)
        b = torch.sum(pred * pred, 1) + 0.001
        c = torch.sum(target * target, 1) + 0.001
        d = (2 * a) / (b + c)
        return 1 - d

    def mil_loss(self, pred, target):
        """Multi-instance loss.

        Args:
            pred (torch.Tensor): size of [batch_size, 128, 128], where 128 is input_size // 4
            target (torch.Tensor): size of [batch_size, 128, 128], where 128 is input_size // 4
        Return:
            loss (torch.Tensor): size of [batch_size]
        """
        row_labels = target.max(1)[0]
        column_labels = target.max(2)[0]
        row_input = pred.max(1)[0]
        column_input = pred.max(2)[0]

        loss = self.dice_loss(column_input, column_labels)
        loss += self.dice_loss(row_input, row_labels)

        return loss

    def training_step(self, x):
        """training step."""
        optimizer = self.optimizers()
        loss = {}
        image = x['image']

        local_step = self.local_step
        self.local_step += 1

        if 'timage' in x.keys():
            timage = x['timage']
        else:
            timage = image
        student_output = self.student(image, x['mask'], x['bbox'])
        teacher_output = self.teacher(timage, x['mask'], x['bbox'])
        B, oh, ow = student_output['seg'].shape[0], student_output['seg'].shape[2], student_output['seg'].shape[3]
        mask = F.interpolate(x['mask'], size=(oh, ow), mode='bilinear', align_corners=False).reshape(-1, oh, ow)

        if 'image' in x:
            student_seg_sigmoid = torch.sigmoid(student_output['seg'])[:, 0].float()
            teacher_seg_sigmoid = torch.sigmoid(teacher_output['seg'])[:, 0].float()

            # Multiple instance learning Loss
            loss_mil = self.mil_loss(student_seg_sigmoid, mask)
            # Warmup loss weight for multiple instance learning loss
            if self.current_epoch > 0:
                step_mil_loss_weight = 1
            else:
                step_mil_loss_weight = min(1, 1. * local_step / self.cfg.train.loss_mil_step)
            loss_mil *= step_mil_loss_weight
            loss_mil = loss_mil.sum() / (loss_mil.numel() + 1e-4) * self.loss_mil_weight
            loss.update({'mil': loss_mil})
            # Tensorboard logs
            self.log("train/loss_mil", loss_mil, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

            # Conditional Random Fields Loss
            th, tw = oh * self.cfg.train.crf_size_ratio, ow * self.cfg.train.crf_size_ratio
            # resize image
            scaled_img = F.interpolate(image, size=(th, tw), mode='bilinear', align_corners=False).reshape(B, -1, th, tw)
            # resize student segmentation
            scaled_stu_seg = F.interpolate(student_seg_sigmoid[None, ...], size=(th, tw), mode='bilinear', align_corners=False).reshape(B, th, tw)
            # resize teacher segmentation
            scaled_tea_seg = F.interpolate(teacher_seg_sigmoid[None, ...], size=(th, tw), mode='bilinear', align_corners=False).reshape(B, th, tw)
            # resize mask
            scaled_mask = F.interpolate(x['mask'], size=(th, tw), mode='bilinear', align_corners=False).reshape(B, th, tw)
            # loss_crf, pseudo_label
            loss_crf, _ = self.crf_loss(scaled_img, scaled_stu_seg, (scaled_stu_seg + scaled_tea_seg) / 2, scaled_mask)
            if self.current_epoch > 0:
                step_crf_loss_weight = 1
            else:
                step_crf_loss_weight = min(1. * local_step / self.loss_crf_step, 1.)
            loss_crf *= self.loss_crf_weight * step_crf_loss_weight
            loss.update({'crf': loss_crf})
            self.log("train/loss_crf", loss_crf, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        else:
            raise NotImplementedError

        total_loss = sum(loss.values())
        self.log("train/loss", total_loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("lr", optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        self.log("train/bs", image.shape[0], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        optimizer.zero_grad()
        self.manual_backward(total_loss)
        optimizer.step()
        if self._optim_type == 'adamw':
            adjust_learning_rate(optimizer, 1. * local_step / self._num_iter_per_epoch + self.current_epoch, self.cfg)
        self.teacher.update(self.student)

    def training_epoch_end(self, outputs):
        """On training epoch end."""
        self.local_step = 0

    def validation_step(self, batch, batch_idx, return_mask=False):
        """Validation step."""
        if self.cfg.dataset.load_mask:
            imgs, gt_masks, masks, labels, ids, boxmasks, boxes, ext_boxes, ext_hs, ext_ws =\
                batch['image'], batch['gtmask'], batch['mask'], batch['compact_category_id'], \
                batch['id'], batch['boxmask'], batch['bbox'], batch['ext_boxes'], batch['ext_h'], batch['ext_w']
        else:
            imgs, gt_masks, masks, labels, ids, boxmasks, boxes, ext_boxes, ext_hs, ext_ws =\
                batch['image'], batch['boxmask'], batch['boxmask'], batch['compact_category_id'], \
                batch['id'], batch['boxmask'], batch['bbox'], batch['ext_boxes'], batch['ext_h'], batch['ext_w']

        _, _, H, W = imgs.shape  # B, C, H, W
        denormalized_images = self.denormalize(imgs.cpu().numpy().transpose(0, 2, 3, 1)).astype(np.uint8)
        labels = labels.cpu().numpy()

        if self.cfg.evaluate.use_mixed_model_test:
            s_outputs = self.student(imgs, batch['boxmask'], batch['bbox'])
            t_outputs = self.teacher(imgs, batch['boxmask'], batch['bbox'])
            segs = (s_outputs['seg'] + t_outputs['seg']) / 2
        else:
            if self.cfg.evaluate.use_teacher_test:
                outputs = self.teacher(imgs, batch['boxmask'], batch['bbox'])
            else:
                outputs = self.student(imgs, batch['boxmask'], batch['bbox'])
            segs = outputs['seg']

        if self.cfg.evaluate.use_flip_test:
            if self.cfg.evaluate.use_mixed_model_test:
                s_outputs = self.student(torch.flip(imgs, [3]), batch['boxmask'], batch['bbox'])
                t_outputs = self.teacher(torch.flip(imgs, [3]), batch['boxmask'], batch['bbox'])
                flipped_segs = torch.flip((s_outputs['seg'] + t_outputs['seg']) / 2, [3])
                segs = (flipped_segs + segs) / 2
            else:
                if self.cfg.evaluate.use_teacher_test:
                    flip_outputs = self.teacher(torch.flip(imgs, [3]), batch['boxmask'], batch['bbox'])
                else:
                    flip_outputs = self.student(torch.flip(imgs, [3]), batch['boxmask'], batch['bbox'])
                segs = (segs + torch.flip(flip_outputs['seg'], [3])) / 2

        segs = F.interpolate(segs, (H, W), align_corners=False, mode='bilinear')
        segs = segs.sigmoid()
        thres_list = [0, 32**2, 96 ** 2, 1e5**2]

        segs = segs * boxmasks
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        binseg = segs.clone()
        for idx, (lth, hth) in enumerate(zip(thres_list[:-1], thres_list[1:])):
            obj_ids = ((lth < areas) * (areas <= hth)).cpu().numpy()
            if obj_ids.sum() > 0:
                binseg[obj_ids] = (binseg[obj_ids] > self.mask_thres[idx]).float()

        tb_logger = self.logger.experiment
        epoch_count = self.current_epoch

        batch_ious = []

        img_pred_masks = []

        for idx, (img_h, img_w, ext_h, ext_w, ext_box, seg, gt_mask, area, label) in enumerate(zip(batch['height'], batch['width'], ext_hs, ext_ws, ext_boxes, segs, gt_masks, areas, labels)):
            roi_pred_mask = F.interpolate(seg[None, ...], (ext_h, ext_w), mode='bilinear', align_corners=False)[0][0]
            h, w = int(img_h), int(img_w)
            img_pred_mask_shape = h, w

            img_pred_mask = np.zeros(img_pred_mask_shape).astype(float)

            img_pred_mask[max(ext_box[1], 0):min(ext_box[3], h), max(ext_box[0], 0):min(ext_box[2], w)] = \
                roi_pred_mask[max(0 - ext_box[1], 0):ext_h + min(0, h - ext_box[3]), max(0 - ext_box[0], 0):ext_w + min(0, w - ext_box[2])].cpu().numpy()

            for idx, (lth, hth) in enumerate(zip(thres_list[:-1], thres_list[1:])):
                if lth < area <= hth:
                    img_pred_mask = (img_pred_mask > self.mask_thres[idx]).astype(float)

            img_pred_masks.append(img_pred_mask[None, ...])
            if self.cfg.dataset.load_mask:
                iou = self.mIoUMetric.cal_iou(img_pred_mask[np.newaxis, ...], gt_mask.data[np.newaxis, ...])
                # overall mask IoU
                self.mIoUMetric.update(int(label), iou[0])
                batch_ious.extend(iou)
                # Small/Medium/Large IoU
                for jdx, (lth, hth) in enumerate(zip(thres_list[:-1], thres_list[1:])):
                    obj_ids = ((lth < area) * (area <= hth)).cpu().numpy()
                    if obj_ids.sum() > 0:
                        self.areaMIoUMetrics[jdx].update_with_ious(labels[obj_ids], iou[obj_ids])

        # Tensorboard vis
        if self.cfg.dataset.load_mask:
            for idx, batch_iou, img, seg, label, gt_mask, mask, _, area in zip(ids, batch_ious, denormalized_images, segs, labels, gt_masks, masks, boxes, areas):
                if area > 64**2 and batch_iou < 0.78 and self.vis_cnt <= 100:
                    seg = seg.cpu().numpy().astype(np.float32)[0]
                    mask = mask.data

                    seg = cv2.resize(seg, (W, H), interpolation=cv2.INTER_LINEAR)
                    seg = (seg * 255).astype(np.uint8)
                    seg = cv2.applyColorMap(seg, cv2.COLORMAP_JET)
                    tseg = cv2.applyColorMap((mask[0] > 0.5).cpu().numpy().astype(np.uint8) * 255, cv2.COLORMAP_JET)

                    vis = cv2.addWeighted(img, 0.5, seg, 0.5, 0)
                    tvis = cv2.addWeighted(img, 0.5, tseg, 0.5, 0)

                    tb_logger.add_image(f'val/vis_{int(idx)}', vis, epoch_count, dataformats="HWC")
                    tb_logger.add_image(f'valgt/vis_{int(idx)}', tvis, epoch_count, dataformats="HWC")
                self.vis_cnt += 1

        ret_dict = {}
        if return_mask:
            ret_dict['img_pred_masks'] = img_pred_masks
        if self.cfg.dataset.load_mask:
            ret_dict['ious'] = batch_ious
        return ret_dict

    def get_parameter_groups(self, print_fn=print):
        """Get parameter groups."""
        groups = ([], [], [], [])

        for name, value in self.named_parameters():
            # pretrained weights
            if 'backbone' in name:
                if 'weight' in name:
                    # print_fn(f'pretrained weights : {name}')
                    groups[0].append(value)
                else:
                    # print_fn(f'pretrained bias : {name}')
                    groups[1].append(value)

            # scracthed weights
            else:
                if 'weight' in name:
                    if print_fn is not None:
                        print_fn(f'scratched weights : {name}')
                    groups[2].append(value)
                else:
                    if print_fn is not None:
                        print_fn(f'scratched bias : {name}')
                    groups[3].append(value)
        return groups

    def validation_epoch_end(self, outputs):
        """On validation epoch end."""
        mIoU = self.mIoUMetric.compute()
        self.log("val/mIoU", mIoU, on_epoch=True, prog_bar=True, sync_dist=True)
        if dist.get_rank() == 0:
            print(f"val/mIoU: {mIoU}")
        if "coco" in self.cfg.dataset.type:
            # cat_kv = dict([(cat["name"], cat["id"]) for cat in self.categories])
            if self.cfg.evaluate.comp_clustering:
                clustering_score = self.clusteringScoreMetrics.compute()
                self.log("val/cluster_score", clustering_score, on_epoch=True, prog_bar=True, sync_dist=True)
            if dist.get_rank() == 0:
                if self.cfg.evaluate.comp_clustering:
                    print("val/cluster_score", clustering_score)
        else:
            raise NotImplementedError
        self.mIoUMetric.reset()
        self.vis_cnt = 0

        for i, name in zip(range(len(self.areaMIoUMetrics)), ["small", "medium", "large"]):
            area_mIoU = self.areaMIoUMetrics[i].compute()
            self.log(f"val/mIoU_{name}", area_mIoU, on_epoch=True, sync_dist=True)
            if dist.get_rank() == 0:
                print(f"val/mIoU_{name}: {area_mIoU}")
            self.areaMIoUMetrics[i].reset()


class MALPseudoLabels(MAL):
    """MAL model for pseudo label generation."""

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__(*args, **kwargs)
        self.box_inputs = None

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        pred_dict = super().validation_step(batch, batch_idx, return_mask=True)
        pred_seg = pred_dict['img_pred_masks']
        if self.cfg.dataset.load_mask:
            ious = pred_dict['ious']

        ret = []
        cnt = 0
        # t = time.time()
        for seg, (x0, y0, x1, y1), idx, image_id, category_id in zip(pred_seg, batch['bbox'], batch['id'], batch.get('image_id', batch.get('video_id', None)), batch['category_id']):
            encoded_mask = encode(np.asfortranarray(seg[0].astype(np.uint8)))
            encoded_mask['counts'] = encoded_mask['counts'].decode('ascii')
            labels = {
                "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                "id": int(idx),
                "category_id": int(category_id),
                "segmentation": encoded_mask,
                "iscrowd": 0,
                "area": float(x1 - x0) * float(y1 - y0),
                "image_id": int(image_id)
            }
            if 'score' in batch.keys():
                labels['score'] = float(batch['score'][cnt].cpu().numpy())
            if self.cfg.dataset.load_mask:
                labels['iou'] = float(ious[cnt])
            cnt += 1
            ret.append(labels)

        if batch.get('ytvis_idx', None) is not None:
            for ytvis_idx, labels in zip(batch['ytvis_idx'], ret):
                labels['ytvis_idx'] = list(map(int, ytvis_idx))

        return ret

    def validation_epoch_end(self, outputs):
        """On validation epoch end."""
        super().validation_epoch_end(outputs)
        ret = list(itertools.chain.from_iterable(outputs))
        if self.trainer.strategy.root_device.index > 0:
            with open(f"{self.cfg.inference.label_dump_path}.part{self.trainer.strategy.root_device.index}", "w", encoding='utf-8') as f:
                json.dump(ret, f)
            torch.distributed.barrier()
        else:
            val_ann_path = self.cfg.inference.ann_path
            with open(val_ann_path, "r", encoding='utf-8') as f:
                anns = json.load(f)
            torch.distributed.barrier()
            for i in range(1, len(self.cfg.gpu_ids)):
                with open(f"{self.cfg.inference.label_dump_path}.part{i}", "r", encoding='utf-8') as f:
                    obj = json.load(f)
                ret.extend(obj)
                os.remove(f"{self.cfg.inference.label_dump_path}.part{i}")

            if ret[0].get('ytvis_idx', None) is None:
                # for COCO format
                _ret = []
                _ret_set = set()
                for ann in ret:
                    if ann['id'] not in _ret_set:
                        _ret_set.add(ann['id'])
                        _ret.append(ann)
                anns['annotations'] = _ret
            else:
                # for YouTubeVIS format
                for inst_ann in anns['annotations']:
                    len_video = len(inst_ann['bboxes'])
                    inst_ann['segmentations'] = [None for _ in range(len_video)]

                for seg_ann in ret:
                    inst_idx, frame_idx = seg_ann['ytvis_idx']
                    anns['annotations'][inst_idx]['segmentations'][frame_idx] = seg_ann['segmentation']

            with open(self.cfg.inference.label_dump_path, "w", encoding='utf-8') as f:
                json.dump(anns, f)

            if self.box_inputs is not None:
                print("Start evaluating the results...")
                cocoGt = COCO(self.cfg.val_ann_path)
                cocoDt = cocoGt.loadRes(self.cfg.label_dump_path + ".result")

                for iou_type in ['bbox', 'segm']:
                    cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
                    cocoEval.evaluate()
                    cocoEval.accumulate()
                    cocoEval.summarize()
