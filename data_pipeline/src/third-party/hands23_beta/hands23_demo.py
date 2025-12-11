# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from cgi import test
import os
import torch
import cv2
import random
import numpy as np
import pdb
import copy
import argparse
import json
import glob
import torch
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from hodetector.data import register_ho_pascal_voc, hoMapper
from hodetector.modeling import roi_heads
from vis_utils import vis_per_image

import sys
# sys.path.append('/home/dshan/workspace/hands23.beta')

def tell_grasp(grasp_type):
    # get names for grasp
    if grasp_type == 0:
        return "NP-Palm"
    elif grasp_type == 1:
        return  "NP-Fin"
    elif grasp_type == 2:
        return "Pow-Pris"
    elif grasp_type == 3:
        return "Pre-Pris"
    elif grasp_type == 4:
        return "Pow-Circ"
    elif grasp_type == 5:
        return "Pre-Circ"
    elif grasp_type == 6:
        return  "Later"
    elif grasp_type == 7:
        return "Other"
    else:
        pdb.set_trace()

def tell_contact(contact):
    # get names for contact
    if contact == 0:
        return "no_contact"
    elif contact == 1:
        return "other_person_contact"
    elif contact == 2:
        return "self_contact"
    elif contact == 3:
        return "object_contact"
    else:
        print("error!")
        return "obj_to_obj_contact"
        
                
def tell_touch(touch):
    # get names for touch
    if touch == 0:
        return "tool_,_touched"
    elif touch == 1:
        return "tool_,_held"
    elif touch == 2:
        return "tool_,_used"
    elif touch == 3:
        return "container_,_touched"
    elif touch == 4:
        return "container_,_held"
    elif touch == 5:
        return "neither_,_touched"
    elif touch == 6:
        return "neither_,_held"
    else:
        print("error!")
        pdb.set_trace()


def tell_grasp_clean(grasp_type):
    # get short names for grasps
    if grasp_type == 0:
        return "NP-P"
    elif grasp_type == 1:
        return  "NP-F"
    elif grasp_type == 2:
        return "Pow-P"
    elif grasp_type == 3:
        return "Pre-P"
    elif grasp_type == 4:
        return "Pow-C"
    elif grasp_type == 5:
        return "Pre-C"
    elif grasp_type == 6:
        return  "Lat"
    elif grasp_type == 7:
        return "Other"
    else:
        pdb.set_trace()


def tell_contact_clean(contact):
    # get short names for contact
    if contact == 0:
        return "no"
    elif contact == 1:
        return "other_p"
    elif contact == 2:
        return "self"
    elif contact == 3:
        return "obj"
    elif contact == 4:
        return "objs"
    else:
        pdb.set_trace()
        

def parse_grasp(grasp_scores):
    grasp_dict = {}
    for type, score in zip(["NP-Palm","NP-Fin", "Pow-Pris", "Pre-Pris", "Pow-Circ", "Pre-Circ", "Later","Other"], grasp_scores):
        grasp_dict[type] = str(round(score.item(),4))
    return grasp_dict


def parse_touch(touch_scores):
    touch_dict = {}
    for touch,score in zip(["tool_,_touched", "tool_,_held", "tool_,_used", "container_,_touched", "container_,_held","neither_,_touched", "neither_,_held"], touch_scores):
        touch_dict[touch] = str(round(score.item(),4))
    return touch_dict


class Hands:
    def __init__(self, hand_id, hand_bbox, hand_mask, contactState, hand_side, grasp, pred_score, grasp_scores = None):
        self.id = hand_id
        self.hand_bbox = hand_bbox
        self.contactState = tell_contact(contactState)
        self.contactState_clean = tell_grasp_clean(contactState)
        self.hand_side = "right_hand" if hand_side==1 else "left_hand"
        self.obj_bbox = None
        self.obj_touch = None
        self.obj_touch_score = None
        self.second_obj_bbox = None
        self.grasp = tell_grasp(grasp)
        self.grasp_clean = tell_grasp_clean(grasp)
        self.grasp_scores = grasp_scores
        self.hand_mask = hand_mask
        self.pred_score = round(pred_score,2)
        self.obj_bbox = None
        self.obj_touch = None
        self.obj_touch_clean = None
        self.obj_masks = None
        self.second_obj_bbox = None
        self.second_obj_masks = None
        self.has_first = False
        self.has_second = False
        self.obj_pred_score = None
        self.sec_obj_pred_score = None
    

    def set_first_obj(self, obj_bbox , obj_touch , obj_masks, pred_score, touch_scores = None):
        self.obj_bbox = obj_bbox
        self.obj_touch = tell_touch(obj_touch)
        self.obj_touch_clean = tell_touch(obj_touch)
        self.obj_masks = obj_masks
        self.obj_pred_score = round(pred_score,2)
        self.has_first  = True
        self.obj_touch_score = touch_scores
         

    def set_second_obj(self, obj_bbox, obj_masks, pred_score):
        self.second_obj_bbox = obj_bbox
        self.second_obj_masks = obj_masks
        self.sec_obj_pred_score = round(pred_score,2)
        self.has_second = True
    

    def save_masks(self, save_dir, im, img_id, mess = ''):
        ims = copy.deepcopy(im)
        ims[:,:,:] = 0
        ims[self.hand_mask, :] = 255

        # pdb.set_trace()
        save_dir = os.path.join(save_dir, "masks"+mess)
        img_id = img_id.strip('\n')
        os.makedirs(save_dir, exist_ok=True)

        cv2.imwrite(save_dir+'/2_'+str(self.id)+'_'+img_id.split('.')[0]+'.png', ims)

        if self.has_first:
            ims[:,:,:] = 0
            ims[self.obj_masks, :] = 255
            cv2.imwrite(save_dir+'/3_'+str(self.id)+'_'+img_id.split('.')[0]+'.png', ims)

            if self.has_second:
                ims[:,:,:] = 0
                ims[self.second_obj_masks, :] = 255
                cv2.imwrite(save_dir+'/5_'+str(self.id)+'_'+img_id.split('.')[0]+'.png', ims)


    def get_json(self):
        info = {}
        info['hand_id'] = self.id 
        info['hand_bbox'] = [str(x) for x in self.hand_bbox]
        info['contact_state'] = self.contactState
        info['hand_side'] = self.hand_side
        info['obj_bbox'] = [str(x) for x in self.obj_bbox] if self.obj_bbox is not None else None
        info['obj_touch'] = str(self.obj_touch)
        info['obj_touch_scores'] = parse_touch(self.obj_touch_score) if self.has_first else None
        info['second_obj_bbox'] = [str(x) for x in self.second_obj_bbox]  if self.second_obj_bbox is not None else None
        info['grasp'] = self.grasp
        info['grasp_scores'] = parse_grasp(self.grasp_scores) 
        info['hand_pred_score'] = str(self.pred_score)
        info['obj_pred_score'] = str(self.obj_pred_score)
        info['sec_obj_pred_score'] = str(self.sec_obj_pred_score)
        return info



def deal_output(im, predictor):
    outputs = predictor(im)

    pred_boxes = outputs["instances"].get("pred_boxes").tensor.to("cpu").detach().numpy()
    pred_dz = outputs["instances"].get("pred_dz").to("cpu").detach().numpy()
    pred_classes =  outputs["instances"].get("pred_classes").to("cpu").detach().numpy()
    pred_scores = outputs["instances"].get("scores").to("cpu").detach().numpy()
    pred_masks = outputs["instances"].get("pred_masks").to("cpu").detach().numpy()


    interaction = torch.tensor(pred_dz[:, 4])
    hand_side = torch.tensor(pred_dz[:, 5])
    grasp =  torch.tensor(pred_dz[:, 6])
    touch_type = torch.tensor(pred_dz[:, 7])
    contact_state = torch.tensor(pred_dz[:,8])

    scores = torch.tensor(pred_dz[:,9])
    grasp_scores = torch.tensor(pred_dz[:,10:18])
    touch_scores = torch.tensor(pred_dz[:,18:25])

    hand_list = []
    count = 0

    for i in range(len(pred_classes)):
        if pred_classes[i] == 0:
            curr_hand = Hands(hand_id= count, hand_bbox=pred_boxes[i], hand_mask=pred_masks[i], contactState=int(contact_state[i].item()),hand_side=hand_side[i].item(), grasp = grasp[i].item(), pred_score= pred_scores[i], grasp_scores= grasp_scores[i])
            count = count+1

            if interaction[i] >=0:
                obj_id = int(interaction[i])

                curr_hand.set_first_obj(obj_bbox=pred_boxes[obj_id], obj_touch= touch_type[obj_id].item(), obj_masks=pred_masks[obj_id], pred_score= pred_scores[obj_id], touch_scores=touch_scores[obj_id])

                if interaction[obj_id] >=0:
                    second_obj_id = int(interaction[obj_id])

                    curr_hand.set_second_obj(obj_bbox=pred_boxes[second_obj_id], obj_masks=pred_masks[second_obj_id], pred_score= pred_scores[second_obj_id])

            hand_list.append(curr_hand)

    return hand_list





def set_cfg_old(args):

    cfg = get_cfg()
    cfg.merge_from_file("src/third-party/hands23_beta/faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml")
    args.model_weights = f'./saved_models/hands23_model/final_on_blur_model_0399999.pth'

    if args.model_weights is not None:
        cfg.MODEL.WEIGHTS = args.model_weights
    else:
        cfg.MODEL.WEIGHTS = args.model_weights
    
    thresh = args.thresh if args.thresh is not None else 0.05
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(thresh)

    hand_thresh = args.hand_thresh if args.hand_thresh is not None else 0.8
    cfg.HAND = float(hand_thresh)

    first_obj_thresh = args.first_obj_thresh if args.first_obj_thresh is not None else 0.3
    cfg.FIRSTOBJ = float(first_obj_thresh)

    second_obj_thresh = args.second_obj_thresh if args.second_obj_thresh is not None else 0.05 #0.3
    cfg.SECONDOBJ = float(second_obj_thresh)

    cfg.HAND_RELA = 0.3
    cfg.OBJ_RELA  = 0.05 # 0.3

    cfg.freeze()

    return cfg


def set_cfg():

    cfg = get_cfg()
    cfg.merge_from_file("src/third-party/hands23_beta/faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml")
    model_weights = f'./saved_models/hands23_model/final_on_blur_model_0399999.pth'

    
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.HAND = 0.8
    cfg.FIRSTOBJ = 0.3
    cfg.SECONDOBJ = 0.05 

    cfg.HAND_RELA = 0.3
    cfg.OBJ_RELA  = 0.05 # 0.3

    cfg.freeze()

    return cfg




def init_hands23(args=None):
    # parser_hands23 = argparse.ArgumentParser()
    # parser_hands23.add_argument("--thresh") 
    # parser_hands23.add_argument("--hand_thresh")
    # parser_hands23.add_argument("--first_obj_thresh")
    # parser_hands23.add_argument("--second_obj_thresh")
    # parser_hands23.add_argument("--model_weights", default=f"./saved_models/hands23_model/final_on_blur_model_0399999.pth")
    # # parser_hands23.add_argument("--data_dir", default=f"./images")
    # args_hands23 = parser_hands23.parse_args()
    
    # set configuration
    cfg = set_cfg()
    predictor = DefaultPredictor(cfg)
    return predictor


# from mmengine.structures import BaseDataElement, InstanceData, PixelData 
# from mmdet.structures.det_data_sample import DetDataSample



def inference_hands23(det_model, im):
    '''
    4 classes of objects
        0: left_hand
        1: right_hand
        2: first_object
        3: second_object
    '''
    hand_lists = deal_output(im = im, predictor= det_model)
    return hand_lists
    
    # print(hand_lists)
    bboxes, labels, scores = [], [], []
    for hands in hand_lists:
        p = hands.get_json()
        # basic info
        h_bbox  = [ float(x) for x in p['hand_bbox']]
        h_score = float(p['hand_pred_score'])
        h_side    = p['hand_side']
        fo_bbox   = p['obj_bbox']
        so_bbox   = p['second_obj_bbox']

        # add
        bboxes.append(h_bbox)
        labels.append(0 if h_side=='left_hand' else 1)
        scores.append(h_score)
        
        if fo_bbox is not None:
            fo_bbox  = [ float(x) for x in fo_bbox]
            fo_score = float(p['obj_pred_score'])
            # add
            bboxes.append(fo_bbox)
            labels.append(2)
            scores.append(fo_score)

            if so_bbox is not None:
                so_bbox  = [ float(x) for x in so_bbox ]
                so_score = float(p['sec_obj_pred_score'])
                # add
                bboxes.append(so_bbox)
                labels.append(3)
                scores.append(so_score)


    result = DetDataSample()
    img_meta = dict(img_shape=(480, 640), pad_shape=(480, 640))
    pred_instances = InstanceData(metainfo=img_meta)
    if len(bboxes) == 0:
        # pred_instances.bboxes = None #torch.Tensor(bboxes).cuda()
        # pred_instances.labels = None #torch.Tensor(labels).cuda()
        # pred_instances.scores = None #torch.Tensor(scores).cuda()
        pred_instances.bboxes = torch.Tensor([[0, 0, 1, 1]]).float().cuda()
        pred_instances.labels = torch.Tensor([0]).int().cuda()
        pred_instances.scores = torch.Tensor([0]).float().cuda()
    else:
        pred_instances.bboxes = torch.Tensor(bboxes).float().cuda()
        pred_instances.labels = torch.Tensor(labels).int().cuda()
        pred_instances.scores = torch.Tensor(scores).float().cuda()
    result.pred_instances= pred_instances

    # print(result)
    # breakpoint()
    return result




def main():
    frame = '/home/dshan/workspace/datasets/our_data/videos_decode/srl_put_egg_1/color_000107.jpg'
    im = cv2.imread(frame)
    det_model = init_detector()
    result = inference_hands23(det_model,im)

def main_old():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresh") 
    parser.add_argument("--hand_thresh")
    parser.add_argument("--first_obj_thresh")
    parser.add_argument("--second_obj_thresh")
    parser.add_argument("--model_weights", default=f"./saved_models/hands23_model/final_on_blur_model_0399999.pth")
    parser.add_argument("--data_dir", default=f"./images")
    args = parser.parse_args()
    
    # set configuration
    cfg = set_cfg(args)
    predictor = DefaultPredictor(cfg)
    
    # inputs
    print(f' a folder of images...')
    images = glob.glob(f'{args.data_dir}/*')

    # outputs
    save_dir = f"{args.data_dir}_vis"
    save_mask_dir = f"{save_dir}/masks" 
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_mask_dir, exist_ok=True)

    # save results
    res = {}
    res["save_dir"] = save_dir
    res["images"] = []
    json_path = f"{args.data_dir}.json"

    # loop
    for test_img in tqdm(images):
        print(f'Processing: {test_img}')
        im = cv2.imread(test_img)
        im_name = os.path.split(test_img)[-1][:-4]
        
        # record img res
        img = {}
        img["file_name"] = test_img
        img["predictions"] = []

        #save masks and vis
        hand_lists = deal_output(im = im, predictor= predictor)
        for hands in hand_lists:
            hands.save_masks(save_dir, im, test_img.split('/')[-1])
            img['predictions'].append(hands.get_json())
    
        # vis and save
        im = vis_per_image(im, img['predictions'], im_name, save_mask_dir, use_simple=False)
        save_path = os.path.join(save_dir, im_name+'.png')
        im.save(save_path)

        res["images"].append(img)
      

    f = open(json_path, 'w')
    json.dump(res, f, indent=4)
    

if __name__ == '__main__':
    main()

