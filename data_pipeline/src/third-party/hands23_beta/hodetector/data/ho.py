# -*- coding: utf-8 -*-
# adapted from pascal_voc.py

import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union

import numpy as np
from parso import split_lines
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

import torch
import glob
import pdb
import random
import time

import cv2

__all__ = ["load_ho_voc_instances", "register_ho_pascal_voc"]

# fmt: off
CLASS_NAMES = (
    "hand", "targetobject"
)


# fmt: on

def get_mask(class_id, row_num, im_id, mask_dir, dirname):
    mask_d = mask_dir+class_id+"_"+ str(row_num) +'_'+im_id.replace('.jpg', '')+'.png'

    try:
        mask = cv2.imread(mask_d)
        L = len(mask)
    except:
        # pdb.set_trace()
        try:
            im = cv2.imread(dirname+im_id)
            size = im.shape
        except:
            pdb.set_trace()
            print("error")

        # print("********ERROR: Can't find mask: "+ mask_d +" !*************")
        return np.ones(shape = (size[0], size[1], 1)).astype(bool), mask_d

    mask = mask[:,:,0] > 128
    mask = mask.astype(bool)

    return mask, ''

def seg_to_poly(x,y,w,h,box_segments):

    if box_segments.any() == None:
        return False


    box = [x,y,w,h]
    # box_segments = torch.zeros((im_w,im_h,1)).bool().cpu()
    # box_segments[box[1]:box[1]+box[3], box[0]:box[0]+box[2], :] = True
    contours, hierarchy = cv2.findContours(box_segments.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []

    for contour in contours:
        if contour.size >=6:
            segmentation.append(contour.flatten().tolist())
    
    float_segmentation = []

    for segseg in segmentation:
        float_segmentation.append([float(x) for x in segseg])
    
    
    return float_segmentation

def load_ho_voc_instances(dirname: str, class_names = None,split = None):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    # list_dir = "/y/evacheng/COCO_split"
    #updated with new split //Feb 3rd 2023
    #list_dir = "/y/evacheng/allMerged6Splits"
    #list_dir = "/home/evacheng/allMerged6Splits"
    #list_dir = "/home/evacheng/10k_subset_splits"
    #list_dir = "/home/evacheng/500EKSplits"


    #list_dir = "/home/evacheng/allMerged6Splits"
    #list_dir = "/y/evacheng/500EKSplits"
    #mask_dir = "/home/evacheng/masks/"
    list_dir = "/w/fouhey/hands2/allMerged6Splits"
    #list_dir = "/y/evacheng/ten_K"
    #list_dir = "/y/evacheng/500EKSplits"
   
    #mask_dir = "/y/ayhassen/allmerged/masks/"
    
    mask_dir = "/home/evacheng/masks/"



    dicts = []

    count_im = 0

    

    print(os.path.join(list_dir, split + ".txt"))
    # with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
    #     fileids = np.loadtxt(f, dtype=np.str)
    with open(os.path.join(list_dir, split + ".txt")) as f:
        fileids = [line.rstrip('\n') for line in f.readlines()]
    
   

    # fileids = ["CC_000000202321.jpg", "EK_0047_P35_105_frame_0000031683.jpg"]

    fileids = fileids[:100]
    
    for fileid in fileids:
        #removed ".jpg" based on dataset format

        #changed temporarily for testing on Blur data //Feb 23 2023
        #change back for further implementation Note!
        anno_file = os.path.join(dirname.replace("Blur", ""), fileid+".txt")
        jpeg_file = os.path.join(dirname, fileid)



        # pdb.set_trace()

        # there_is_second_object = False
     
        if os.path.exists(jpeg_file):
            with PathManager.open(anno_file) as f:
                 annos = f.readlines()

            if len(annos) == 0:
                # print("Ignore file "+ jpeg_file +" with no Annotations")
                continue

            # if count_im >7:
            #     break
            
            # count_im = count_im+1

            r = {
                "file_name": jpeg_file,
                "image_id": jpeg_file.replace(".jpg", ""),
            }
            instances = []

            count = 0
            im_id = jpeg_file.replace(dirname, '')

            for line in annos:

              
                infos = line.split('|')
                # pdb.set_trace()

                hand_box = infos[2]
                object_box = infos[3]

                #changed after adding grasp
                second_obj_box = infos[-2].replace("\n", "")[1:]
                grasp_type = infos[-1].replace("\n", "").replace(" ", "")
                touch_type = infos[-3].strip()
                contact = infos[1]

                touch = 100
                contactState = 100
                grasp = 100

                # pdb.set_trace()

                if "no_contact" in contact:
                    contactState = 0
                elif "other_person_contact" in contact:
                    contactState = 1
                elif "self_contact" in contact:
                    contactState = 2
                elif "object_contact" in contact:
                    contactState = 3
                else:
                    pass
                
                if grasp_type == "None":
                    pass
                elif grasp_type == "NP-Palm":
                    grasp = 0
                elif grasp_type == "NP-Fin":
                    grasp = 1
                elif grasp_type == "Pow-Pris":
                    grasp = 2
                elif grasp_type == "Pre-Pris":
                    grasp = 3 
                elif grasp_type == "Pow-Circ":
                    grasp = 4 
                elif grasp_type == "Pre-Circ":
                    grasp = 5 
                elif grasp_type == "Lat":
                    grasp = 6
                # elif grasp_type == "Exten":
                #     grasp = 7 
                elif grasp_type == "Other":
                    grasp = 7
                else:
                    # pdb.set_trace()
                    print("error! Got: " + grasp_type)
                    grasp = 100



                bbox_hand = [float(x) for x in hand_box.split(',')]
                
                #segmentation to test implementation /Feb 2023
                hand_mask, temp = get_mask("2", count,  im_id, mask_dir, dirname)
                segmentation_hand  = seg_to_poly(bbox_hand[0], bbox_hand[1], bbox_hand[2] - bbox_hand[0], bbox_hand[3] - bbox_hand[1], hand_mask)

                del hand_mask

                # if temp != '': #previously for debug purpose
                #     no_mask  = no_mask+1 

                instances.append(
                    {"category_id": 0, "handId": 0, "objectId": -1, "secondObjectId": -1, "bbox": bbox_hand, "bbox_mode": BoxMode.XYXY_ABS, "segmentation": segmentation_hand,
                     "interaction": -1 if object_box.strip() == "None" else [float(x) for x in object_box.split(',')],
                     "handSide": 0 if "left_hand" in infos[0] else 1,
                     "contactState": contactState, "touch": 100, "grasp": grasp
                     }
                )

                if object_box.strip() != "None":
                    bbox_object = [float(x) for x in object_box.split(',')] 

                    # segmentation to test implementation /Feb 2023
                    object_mask, temp = get_mask("3", count,  im_id, mask_dir, dirname)
                    segmentation_object = seg_to_poly(bbox_object[0], bbox_object[1], bbox_object[2] - bbox_object[0], bbox_object[3] - bbox_object[1], object_mask)
                    
                    del object_mask

                    # if temp != '': #previously for debug purpose
                    #     no_mask  = no_mask+1 


                    if touch_type != "None":
                        
                        if touch_type == "tool_,_touched":
                            touch = 0
                        elif touch_type == "tool_,_held":
                            touch = 1
                        elif touch_type == "tool_,_used":
                            touch = 2
                        elif touch_type == "container_,_touched":
                            touch = 3
                        elif touch_type == "container_,_held":
                            touch = 4
                        elif touch_type == "neither_,_touched":
                            touch = 5
                        elif touch_type == "neither_,_held":
                            touch = 6
                        else:
                            pdb.set_trace()
                            print("error!")
                        
                        if "None" not in second_obj_box:
                            # there_is_second_object = True

                            bbox_second_obj = [float(x) for x in second_obj_box.replace("\n","").split(',')]

                            second_object_mask, temp = get_mask("5", count, im_id, mask_dir, dirname)
                            segmentation_second_object =  seg_to_poly(bbox_second_obj[0], bbox_second_obj[1], bbox_second_obj[2] - bbox_second_obj[0], bbox_second_obj[3] - bbox_second_obj[1], second_object_mask)
                            
                            del second_object_mask

                            instances.append({"category_id": 2, "handId": -1, "objectId": -1, "secondObjectId": 0, "bbox": bbox_second_obj, "bbox_mode": BoxMode.XYXY_ABS,  "segmentation":segmentation_second_object,
                                             "handSide": 2, "interaction": -1,
                                             "contactState": 100, "touch": 100, "grasp": 100
                                            })

                            #tools versus second object contact relationship is represented by contactState 4 //Feb 2023

                            instances.append(
                            {"category_id": 1, "handId": -1, "objectId": 0,  "secondObjectId": -1, "bbox": bbox_object, "bbox_mode": BoxMode.XYXY_ABS,  "segmentation": segmentation_object,
                            "handSide": 2, "interaction": bbox_second_obj,
                            "contactState": 4, "touch": touch, "grasp": 100
                            }
                            )
                        else:

                            instances.append(
                        {"category_id": 1, "handId": -1, "objectId": 0, "secondObjectId": -1, "bbox": bbox_object, "bbox_mode": BoxMode.XYXY_ABS,  "segmentation": segmentation_object,
                        "handSide": 2, "interaction": -1 ,
                        "contactState": 100 , "touch": touch,  "grasp": 100
                        })
                    else:

                        instances.append(
                        {"category_id": 1, "handId": -1, "objectId": 0,"secondObjectId": -1, "bbox": bbox_object, "bbox_mode": BoxMode.XYXY_ABS,  "segmentation": segmentation_object,
                        "handSide": 2, "interaction": -1 ,
                        "contactState": 100 , "touch": touch, "grasp": 100
                        })
                
                count = count + 1
            
            
            random.shuffle(instances)
            
            for i in range(len(instances)):
                if instances[i]["handId"] == 0:
                    instances[i]["handId"] = i
                elif instances[i]["objectId"] == 0:
                    instances[i]["objectId"] = i
                else:
                    instances[i]["secondObjectId"] =i
                
                if instances[i]["interaction"] != -1:
                    for j in range(len(instances)):
                        if instances[j]["bbox"] == instances[i]["interaction"]:
                            instances[i]["interaction"] = j
            
            for i in range(len(instances)):
                try:
                    a = int(instances[i]["interaction"])
                except:
                    pdb.set_trace()



            r["annotations"] = instances
            dicts.append(r)

            # try:
            #     for i in range(len(instances)):
            #         assert instances[i].__len__()>0
            # except:
            #     pdb.set_trace()
            #     print("error in ho")
        
   
    return dicts, fileids


def register_ho_pascal_voc(name, dirname, year, split, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: load_ho_voc_instances(dirname, class_names, split))
  
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )

