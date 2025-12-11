# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from detectron2.layers import ShapeSpec, cat
from detectron2.utils.registry import Registry
from fvcore.nn import smooth_l1_loss
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from typing import Dict, Union
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals

import pdb


from math import nan, inf

ROI_H_HEAD_REGISTRY = Registry("ROI_H_HEAD")

count = 0


@ROI_H_HEAD_REGISTRY.register()
class FastRCNNFCHead(nn.Module):
    """
    A head for hand state, hand side and association vector prediction
    """

    @configurable
    def __init__(
        self,
        input_shape: int,
        *,
        box2box_transform,
        num_classes: int,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 100,
        cls_agnostic_bbox_reg: bool = False,
        smooth_l1_beta: float = 0.0,
        box_reg_loss_type: str = "smooth_l1",
        loss_weight: Union[float, Dict[str, float]] = 1.0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)

        # input_size = 2089     #input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # input_size = 2050
        input_size = 1024
        self.relation_layer = nn.Sequential(nn.Linear(input_size, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 2))
        # self.relation_loss = nn.CrossEntropyLoss()

        # weight = torch.zeros(2)
        # weight[0] = 1
        # weight[1] = 100
        self.relation_loss = nn.CrossEntropyLoss(ignore_index=2)
        # self.relation_loss = nn.CrossEntropyLoss(ignore_index=2)

        # self.relation_loss = nn.L1Loss()

        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.relation_layer[0], 0, 0.01)
        normal_init(self.relation_layer[2], 0, 0.01)
        normal_init(self.relation_layer[4], 0, 0.01)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type"     : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight"           : {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
            # fmt: on
        }

    def forward(self, z_feature):
        return self.relation_layer(z_feature)

    def losses(self, relation_pred, label):
        
 
        
        # temp = relation_pred -  1e+33
        # temp = temp.half()
       
        # loss_hand_side = self.relation_loss(temp.float(), label) 

        

        losses = {
                        #"loss_hand_side": loss_hand_side
                        "loss_hand_side": 0.1 * self.relation_loss(relation_pred, label)
                }
       
    
        # else:
        #     losses = {
        #         "loss_relation": torch.zeros(1, device = label.device, dtype = label.dtype)
        #     }
        return losses

    # def hand_contactstate(self, contactstate_pred, proposals):
    #     contactstate_loss = 0
    #     # The loss is only defined on hand proposals
    #     # gt_hands = (cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0))
    #     gt_labels = (cat([p.gt_contactstate for p in proposals], dim=0) if len(proposals) else torch.empty(0))
    #     # sel_gt_labels = gt_labels[gt_hands==0].clone().detach()
    #     # sel_contactstate_pred = contactstate_pred[gt_hands==0].clone().detach()
    #     contactstate_loss = 0.1 * self.hand_contactstate_loss(contactstate_pred, gt_labels)
    #     return contactstate_loss


def build_h_head(cfg, input_shape):
    name = "FastRCNNFCHead"  # TODO: name = cfg.MODEL.ROI_Z_HEAD.NAME
    return ROI_H_HEAD_REGISTRY.get(name)(cfg, input_shape)
