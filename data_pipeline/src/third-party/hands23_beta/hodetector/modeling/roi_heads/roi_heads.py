# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from os import O_NONBLOCK
from collections import defaultdict
from typing import Dict
import torch
import numpy as np
from detectron2.layers import ShapeSpec, cat
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.roi_heads import StandardROIHeads, select_foreground_proposals
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou


# from hodetector.modeling.roi_heads.z_head import build_z_head
from hodetector.modeling.roi_heads.mlp_layer import build_z_head
from hodetector.modeling.roi_heads.hand_side import build_h_head
from hodetector.modeling.roi_heads.tool import build_t_head
from hodetector.modeling.roi_heads.grasp import build_g_head

from hodetector.utils.positional_encoding import *
from typing import Dict, List, Optional, Tuple
from torch.nn import functional as F

import cv2
import pdb
import math
from math import nan
import time

from datetime import datetime


def write_error_mess(mess = ''):
    f = open("/home/evacheng/hand_detector.d2-master/seg_Model_2023_obj_detec/error_mess.txt", "a")
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")

    f.write("Error with " + mess +" at " + str(current_time) +'\n')
    f.close()




@ROI_HEADS_REGISTRY.register()
class hoRCNNROIHeads(StandardROIHeads):
    """
    The ROI specific heads for ho R-CNN
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__(cfg, input_shape)
        self.z_head = build_z_head(cfg, 2089)#self.box_head.output_shape)
        self.h_head = build_h_head(cfg, 2089)
        self.t_head = build_t_head(cfg, 2089)
        self.g_head = build_g_head(cfg, 2089)

        self.cfg = cfg
        
        # If MODEL.VIS_MINIBATCH is True we store minibatch targets
        # for visualization purposes
        # self._vis = cfg.MODEL.VIS_MINIBATCH
        self._misc = {}
        # self._vis_dir = cfg.OUTPUT_DIR

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
       

        # #debug //Feb 10 2023
        # print("************In Forward**************")

        # pdb.set_trace()

        # del images
        if self.training:
            assert targets, "'targets' argument is required during training"

            # try:
            #     temp = proposals[0].get("gt_interaction")
            # except:
            #     pdb.set_trace()
            #     print("error before calling label_and_sample_proposals")

            proposals = self.label_and_sample_proposals(proposals, targets)
           
           

        # del targets

        if self.training:
            #   #debug //Feb 10 2023
            # print("************Going to Forward Box**************")
            losses = self._forward_box(features, proposals, images)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.

            # pdb.set_trace()
            del images
            del targets
            
            #  #debug //Feb 10 2023
            # print("************After Forward Box**************")
            # losses.update(self._forward_mask(features, proposals))
            # losses.update(self._forward_keypoint(features, proposals))

            #  #debug //Feb 10 2023
            # print("***********Update Loss**************")

            # pdb.set_trace()
            
            return proposals, losses
        else:

            # pdb.set_trace()

            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            # pred_instances_1 = self.forward_with_given_boxes(features, pred_instances)
            
            return pred_instances, {}


    def _forward_box(self, features_org: Dict[str, torch.Tensor], proposals: List[Instances], images = None):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        
        features = [features_org[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)


        if self.training:
            # select hand features
            fore_proposals, _ = select_foreground_proposals(proposals, 3)
            # hand_proposals, hand_idx = select_foreground_proposals(fore_proposals, 1)
            # object_proposals, object_idx = select_foreground_proposals(fore_proposals, 0)
            # hand_idx = torch.from_numpy(np.argwhere(np.asarray(hand_idx[0].cpu())).flatten())
            # object_idx = torch.from_numpy(np.argwhere(np.asarray(object_idx[0].cpu())).flatten())
        # else:
            # hand_proposals = proposals
        # TODO: enlarge box

        del box_features

      

        if self.training:


            losses_z, losses_h, losses_t, losses_g= self._forward_z(features, fore_proposals, images)

            losses = self.box_predictor.losses(predictions, proposals)

            losses.update(self._forward_mask(features_org, proposals))

         

            if (losses_z is None) or (torch.isnan(losses_z["loss_relation"])) or (torch.isinf(losses_z["loss_relation"])):
                write_error_mess("z_head")
            else:
                 #if losses_z is not None:
                losses.update(losses_z)


            if (losses_h is None) or (torch.isnan(losses_h["loss_hand_side"])) or (torch.isinf(losses_h["loss_hand_side"])):
                write_error_mess("h_head")
            else:
                losses.update(losses_h)

            if (losses_t is None) or (torch.isnan(losses_t["loss_touch"]) or (torch.isinf(losses_t["loss_touch"]))):
                write_error_mess("t_head")
            else:
                losses.update(losses_t)

            if (losses_g is  None) or (torch.isnan(losses_g["loss_grasp"]) or (torch.isinf(losses_g["loss_grasp"]))):
                write_error_mess("g_head")
            else:
                losses.update(losses_g)

            # pdb.set_trace()
            # proposals is modified in-place below, so losses must be computed first.
            # if self.train_on_pred_boxes:
            #     with torch.no_grad():
            #         pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
            #             predictions, proposals
            #         )
            #         for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
            #             proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            # # debug
            # for value in losses.values():
            #     if torch.isnan(value):
            #         print("bug!!!!")

            # copied from detectron2 roi_heads.py // Feb 21 2023
            # if self.mask_on:
            #     proposals, fg_selection_masks = select_foreground_proposals(
            #         proposals, self.num_classes
            #     )
            #     # Since the ROI feature transform is shared between boxes and masks,
            #     # we don't need to recompute features. The mask loss is only defined
            #     # on foreground proposals, so we need to select out the foreground
            #     # features.
            #     mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
            #     del box_features
            #     losses.update(self.mask_head(mask_features, proposals))

           
            for val in losses.values():
             
                assert torch.isnan(val)== False and torch.isfinite(val)

            # pdb.set_trace()
            return losses
        else:
            
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            z_instances, pred_instances = self._inference_z(features, pred_instances)
            #new code for adding mask /// Jan 2023
            # pred_instances = self._forward_mask(features_org, pred_instances)

            # pdb.set_trace()
            try:
                pred_instances = self.forward_with_given_boxes(features_org, pred_instances)
            except:
                
               pred_instances[0].set("pred_masks", torch.zeros((0,1,0,0), dtype=z_instances[0].dtype, device=z_instances[0].device))
               #pdb.set_trace()
               
            
            
            # if z_instances ==  "None":
            #     return pred_instances
            
            #pred_instances = self.forward_with_given_boxes(features, pred_instances)
            
            for z, instances in zip(z_instances, pred_instances):
                instances.pred_dz = z
            
            # if len(pred_instances[0].pred_dz[0]) == 9:
            #     pdb.set_trace()

            
            return pred_instances



    def get_PF(self, z_feature, bbox, handId, objectId, secondObjectId):

       

        N,F = z_feature.shape

        PF = torch.zeros((N,N,2*F), device= z_feature.device, dtype=z_feature.dtype)

        v = torch.zeros((N,N,2), device= z_feature.device, dtype=z_feature.dtype )
        min_dist = torch.zeros((N,N, 2), device= z_feature.device, dtype=z_feature.dtype )
        max_dist = torch.zeros((N,N, 2), device= z_feature.device, dtype=z_feature.dtype )
        v_norm = torch.zeros((N,N,1), device= z_feature.device, dtype=z_feature.dtype)
        v_over_v_norm = torch.zeros((N,N,2), device= z_feature.device, dtype=z_feature.dtype)

        def sdf(hmin, hmax, val):
            if val < hmin:
                return hmin - val
            elif val > hmax:
                return val -hmax
            else:
                return min(hmin-val, val-hmax)
        
        diag = 0
        for i in range(N):
            # touch[i] <=4 => tools or containers
            if handId[i]>=0 or objectId[i]>=0:
                x1,y1,x2,y2 = bbox[i]
                x1 = x1.item()
                x2 = x2.item()
                y1 = y1.item()
                y2 = y2.item()
                diag = math.sqrt((x2-x1)**2+(y2-y1)**2)



            for j in range(N):
                PF[i,j,:F] = z_feature[i]
                PF[i,j, F:] = z_feature[j]

                #hand-versus-objects and objects-versus-second objects
                if (handId[i]>=0 and objectId[j]>=0) or (objectId[i]>=0 and secondObjectId[j]>=0):
                    X1,Y1,X2,Y2 = bbox[j]
                    X1 = X1.item()
                    X2 = X2.item()
                    Y1 = Y1.item()
                    Y2 = Y2.item()
                    v[i][j][0] = (X1+X2 - x1 - x2)/2
                    v[i][j][1] = (Y1+Y2 - y1 - y2)/2
                    v_norm[i][j][0] = math.sqrt( v[i][j][0]**2 + v[i][j][1]**2)
                    min_dist[i][j][0] =  min( sdf(x1,x2,X1), sdf(x1,x2,X2))/diag
                    min_dist[i][j][1] =  min( sdf(y1,y2,Y1), sdf(y1,y2,Y2))/diag
                    max_dist[i][j][0] =  max( sdf(x1,x2,X1), sdf(x1,x2,X2))/diag
                    max_dist[i][j][1] =  max( sdf(y1,y2,Y1), sdf(y1,y2,Y2))/diag
                    v_over_v_norm[i][j] = v[i][j]/v_norm[i][j]

        return torch.cat((PF,v,v_norm,min_dist, max_dist, v_over_v_norm), 2).cuda()


    def _inference_z(self, features, pred_instances):

        image_size = pred_instances[0].image_size

        # try:
        #     assert pred_instances[0].__len__() >0
        # except:
        #     pdb.set_trace()
        #     return "None", "None"
            
    
    
        pred_boxes_raw =  [x.pred_boxes for x in pred_instances][0].tensor
        pred_classes_raw =  [x.pred_classes for x in pred_instances][0]
        pred_scores = [x.scores for x in pred_instances][0].reshape(-1,1)

        pred_boxes = []
        pred_classes = []
        pred_score = []

        for i in range(pred_scores.shape[0]):
            #if (pred_classes_raw[i] == 0 and pred_scores[i] >= 0.8) or pred_classes_raw[i] == 2 or pred_scores[i] >=0.3:
            if (pred_classes_raw[i] == 1 and pred_scores[i] >= float(self.cfg.FIRSTOBJ)) or (pred_classes_raw[i] == 2 and  pred_scores[i] >= float(self.cfg.SECONDOBJ) ) or (pred_classes_raw[i] == 0 and  pred_scores[i] >= float(self.cfg.HAND)):

                pred_boxes.append([x for x in pred_boxes_raw[i]])
                pred_classes.append(pred_classes_raw[i].item())
                pred_score.append(pred_scores[i])
        

        
        pred_boxes = torch.tensor(pred_boxes).cuda()
        pred_boxes = [Boxes(pred_boxes)]
        pred_boxes_ = pred_boxes[0]
        

        pred_scores_ = torch.tensor(pred_score).cuda()
        

        pred_classes = torch.tensor(pred_classes).cuda()
        pred_classes_ = pred_classes.clone().detach().cuda()
       

       


        z_feature = self.box_pooler(features, pred_boxes)
        z_feature = self.box_head(z_feature)

        boxes = pred_boxes[0].tensor[range(len(z_feature))]
       

        try:
            Ids = F.one_hot(pred_classes, num_classes = 3)
        except:
            temp = torch.zeros(0,1).cuda()
            temp_g = torch.zeros(0,8).cuda()
            temp_t = torch.zeros(0,7).cuda()
            return [torch.cat((boxes,temp, temp,temp, temp, temp, temp_g, temp_t), dim=1)], [Instances(image_size = image_size, pred_boxes = pred_boxes_, pred_classes = pred_classes_, scores = pred_scores_)]
        
        handId = Ids[:,0]-1
        objectId = Ids[:,1]-1
        secondObjectId = Ids[:,2]-1
        
       
        PF = self.get_PF(z_feature, boxes, handId, objectId, secondObjectId)

        
        score_z = self.z_head(PF)
        score_h = self.h_head(z_feature)
        score_t = self.t_head(z_feature)
        score_g = self.g_head(z_feature)
    

        cls_prob = F.softmax(score_z, 2)
        hand_side = F.softmax(score_h, 1)
        touch_type = F.softmax(score_t, 1)
        grasp = F.softmax(score_g, 1)

        inter_prob = torch.zeros_like(cls_prob)
        inter_prob[:,:,0] = 1

        length = len(inter_prob)

       
        
        try:

            for i in range(length):
                if pred_classes[i] == 0:
                    for j in range(length):
                        if pred_classes[j] == 1:
                            inter_prob[i][j] = cls_prob[i][j]
                elif pred_classes[i] == 1:
                    for j in range(length):
                        if pred_classes[j] == 2:
                            inter_prob[i][j] = cls_prob[i][j]
        except:
            pdb.set_trace()
            print("error in assigning interaction probability")

        
    
        score = torch.zeros(score_z.shape[0], 1).cuda()
        interaction = -1 * torch.ones(score_z.shape[0], 1).cuda()
        contact_state = torch.zeros(score_z.shape[0], 1).cuda()

        h_side = torch.zeros(hand_side.shape[0],1).cuda()
        t_type = 100*torch.ones(hand_side.shape[0],1).cuda()
        g_type = 100*torch.ones(hand_side.shape[0],1).cuda()
        


        for i in range(len(z_feature)):
            try:
                if pred_classes[i] == 0:
                    h_side[i] = torch.argmax(hand_side[i]).item()
                    g_type[i] =  torch.argmax(grasp[i]).item()
                
                    cls_prob_i = 1 - inter_prob[i][:,0].reshape(-1)
                    max_val  = torch.max(cls_prob_i)
                    idx = torch.argmax(cls_prob_i)
                    interaction[i] = -1 if max_val.item() <= self.cfg.HAND_RELA else idx.item()
                    # pdb.set_trace()
                    contact_state[i] = 0 if interaction[i] == -1 else torch.argmax(inter_prob[i][int(idx.item())][1:]).item()+1
                    score[i] = max_val
                    
                elif pred_classes[i] == 1:
                    t_type[i] = torch.argmax(touch_type[i]).item()

                    

                    try:
                        assert t_type[i] <= 8
                    except:
                        pdb.set_trace()

                    cls_prob_i = 1 - inter_prob[i][:,0].reshape(-1)
                    max_val  = torch.max(cls_prob_i)
                    idx = torch.argmax(cls_prob_i)
                    interaction[i] = -1 if max_val.item() <= self.cfg.OBJ_RELA else idx.item()
                    score[i] = max_val
            except:
                pdb.set_trace()
                print("error in assign values")
            
        
               
        
        return [torch.cat((boxes, interaction, h_side,g_type, t_type, contact_state, score, grasp, touch_type,  1-inter_prob[:,:,0]), dim=1)], [Instances(image_size = image_size, pred_boxes = pred_boxes_, pred_classes = pred_classes_, scores = pred_scores_)]
        

    def _forward_z(self, features, proposals, images):



        z_feature = self.box_pooler(features,  [x.proposal_boxes for x in proposals])
        z_feature = self.box_head(z_feature)


        try:
            interaction = proposals[0].get('gt_interaction').clone().detach()
        except:
            pdb.set_trace()
            print("Error!")

        handId = proposals[0].get('gt_handId').clone().detach()
        objectId = proposals[0].get('gt_objectId').clone().detach()
        secondObjectId = proposals[0].get('gt_secondObjectId').clone().detach()
        bboxes = proposals[0].get('proposal_boxes').tensor.clone().detach()
        gt_boxes = proposals[0].get('gt_boxes').tensor.clone().detach()

        gt_handSide = proposals[0].get('gt_handSide').clone().detach()
        gt_contactState = proposals[0].get('gt_contactState').clone().detach()
        gt_touch = proposals[0].get('gt_touch').clone().detach()
        gt_grasp = proposals[0].get('gt_grasp').clone().detach()

        length = z_feature.shape[0]
   
        iou = []


        for i in range(length):
            x1,y1,x2,y2 = bboxes[i]
            X1,Y1,X2,Y2 = gt_boxes[i]
            x1 = x1.item()
            x2 = x2.item()
            y1 = y1.item()
            y2 = y2.item()
            X1 = X1.item()
            X2 = X2.item()
            Y1 = Y1.item()
            Y2 = Y2.item()

            intersect = (min(x2,X2)-max(x1,X1))*(min(y2,Y2)-max(y1,Y1))
            union = (x2-x1)*(y2-y1)+(X2-X1)*(Y2-Y1)
            if (intersect/union)>=0.5:
                iou.append(i)
        
        length = len(iou)

        PF = self.get_PF(z_feature[iou], bboxes[iou], handId[iou], objectId[iou], secondObjectId[iou])

       
      
        PL = 100 * torch.ones((length, length), device= interaction.device, dtype=interaction.dtype).cuda()

        try:
            PF.get_device()
            PL.get_device()
        except:
            pdb.set_trace()
       

        for i in range(length):
            if handId[iou[i]] >= 0 or objectId[iou[i]] >= 0:
                for j in range(length):
                    if (handId[iou[i]] >= 0 and objectId[iou[j]] >=0) :
                        PL[i][j] = gt_contactState[iou[i]] if interaction[iou[i]] == objectId[iou[j]] else 0
                    elif (objectId[iou[i]] >=0 and secondObjectId[iou[j]] >= 0):
                        PL[i][j] = gt_contactState[iou[i]] if interaction[iou[i]] == secondObjectId[iou[j]] else 0
                       
        
       
       

        pred_z = self.z_head(torch.flatten(PF, start_dim=0, end_dim=1))
        pred_h = self.h_head(z_feature[iou])
        pred_t = self.t_head(z_feature[iou])
        pred_g = self.g_head(z_feature[iou]) #uncomment when start training grasp
        
        

    
       
        # try:
        assert pred_z.shape[0] == torch.flatten(PL, start_dim=0, end_dim=1).shape[0]
        losses_inter = self.z_head.losses(pred_z, torch.flatten(PL, start_dim=0, end_dim=1))

    

        assert pred_h.shape[0] == gt_handSide[iou].shape[0]
        losses_side = self.h_head.losses(pred_h, gt_handSide[iou])
        assert pred_t.shape[0] == gt_touch[iou].shape[0]
        losses_touch = self.t_head.losses(pred_t, gt_touch[iou])

        assert pred_g.shape[0] == gt_touch[iou].shape[0] #uncomment when start training grasp
        losses_grasp = self.g_head.losses(pred_g, gt_grasp[iou])
           
      
       
        return losses_inter, losses_side, losses_touch, losses_grasp
