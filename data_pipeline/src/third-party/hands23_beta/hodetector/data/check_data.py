import cv2
import pdb
import os

# -*- coding: utf-8 -*-
# adapted from pascal_voc.py

import xml.etree.ElementTree as ET
from typing import List, Tuple, Union

import numpy as np
from parso import split_lines
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

import torch
import glob
import random

import time


def draw_poly_mask(im, float_segmentation, color):
    polygon_list = []
    for poly in float_segmentation:
        points = [(x, y) for (x,y) in zip(poly[0::2], poly[1::2])]
        polygon_list.append(np.array(points).astype(np.int32))
    # new_mask = np.zeros(binary_mask.shape, dtype='uint8')
    im = cv2.fillPoly(im, polygon_list, color)
    return im

def write_obj_html(dir = None):
   file_dir = ("/home/evacheng/public_html/index.html/3/index.html").replace('3', dir)
   f = open(file_dir, "w+")
   f.truncate(0)
  
   image_dir = ("/home/evacheng/public_html/index.html/3/").replace('3', dir)
   

   html = ET.Element('html')
   body = ET.SubElement(html, 'body')

   h2_2 = ET.SubElement(body, 'h2')
   h2_2.text = dir
   # infos = de_f.readlines()



   for file in  glob.glob(image_dir + "/*.jpg"):
   
      image = file.replace(image_dir, '')

      
      
      text = ET.SubElement(body, 'p')
      text.text = image

      try:
         # idx = infos.index(image+ '\n')
         # length = int(infos[idx+1])

       
        
         img = ET.SubElement(body, 'img')
         img.set("src", "./"+file.replace(image_dir, ""))
         img.set("alt", "separate model output")
         img_ = ET.SubElement(body, 'img')
         img_.set("src", "./read_"+file.replace(image_dir, ""))
         img_.set("alt", "separate model output")
      except Exception as e:
         print(image)
       
       
     
   tree = ET.ElementTree(html)
   ET.indent(tree, space='\t', level=0)
   tree.write(open(file_dir, 'wb'))




def seg_to_poly(x,y,w,h,box_segments):

    start_poly_time = time.time()

    box = [x,y,w,h]
  
    contours, hierarchy = cv2.findContours(box_segments.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []

    for contour in contours:
        if contour.size >=6:
            segmentation.append(contour.flatten().tolist())
    
    float_segmentation = []

    for segseg in segmentation:
        float_segmentation.append([float(x) for x in segseg])
    
    
    return float_segmentation, time.time() - start_poly_time

def blendMask(I,mask,color, alpha):
    for c in range(3):
        Ic = I[:,:,c]

        Ic[mask] = ((Ic[mask].astype(np.float32)*alpha) + (float(color[c])*(1-alpha))).astype(np.uint8)
        I[:,:,c] = Ic


def get_mask(class_id, row_num, im_id, mask_dir, dirname):

    start_mask_time = time.time()

    mask_d = mask_dir+class_id+"_"+ str(row_num) +'_'+im_id.replace('.jpg', '')+'.png'

    if os.path.exists(mask_d):
          mask = cv2.imread(mask_d)

          start_d_time = time.time()

          mask = mask[:,:,0] > 128
          mask = mask.astype(bool)

          return mask, '', time.time() - start_mask_time,   start_d_time - start_mask_time

    else:
        im = cv2.imread(dirname+im_id)
        size = im.shape
        start_d_time = time.time()

        # print("********ERROR: Can't find mask: "+ mask_d +" !*************")
        return np.ones(shape = (size[0], size[1], 1)).astype(bool), mask_d, time.time() - start_mask_time, start_mask_time - start_d_time
    


  

   



def load_ho_voc_instances_method(dirname = "/w/fouhey/hands2/allMerged5/", split = None):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
   
    list_dir = "/w/fouhey/hands2/allMerged6Splits"
    mask_dir = "/y/ayhassen/allmerged/masks/"

    save_dir = ("/home/evacheng/public_html/index.html/3/").replace('3', split)

    # print(os.path.join(list_dir, split + ".txt"))
   
    # with open(os.path.join(list_dir, split + ".txt")) as f:
    #     fileids = [line.rstrip('\n') for line in f.readlines()]
    
    count_im = 0
   
    fi = open(list_dir+'/'+split+".txt", "r")
    # fi = open("/y/evacheng/allMerged6Splits/TRAIN.txt", "r")

    fileids = fi.readlines()

    # random.shuffle(fileids)
    
    for fileid in fileids:
      
        fileid = fileid.replace('\n', '')
        anno_file = os.path.join(dirname, fileid+".txt")
        jpeg_file = os.path.join(dirname, fileid)

        
     
        if os.path.exists(jpeg_file):
            with PathManager.open(anno_file) as f:
                 annos = f.readlines()

            r = {
                "file_name": jpeg_file,
                "image_id": jpeg_file.replace(".jpg", ""),
            }
            instances = []

            count = 0


            im = cv2.imread(jpeg_file)

            cv2.imwrite(save_dir+jpeg_file.replace(dirname, ''), im)

            id = jpeg_file.replace(dirname, '')
            
            for line in annos:

              
                infos = line.split('|')
                # pdb.set_trace()

                hand_box = infos[2]
                object_box = infos[3]
                second_obj_box = infos[-1].replace("\n", "")[1:]
                touch_type = infos[-2].strip()
                contact = infos[1]

                    


                bbox_hand = [float(x) for x in hand_box.split(',')]
                # count_hands = count_hands + 1
                
                #segmentation to test implementation /Jan 2023
                segmentation_hand = get_mask("2", count,  id, mask_dir)


                color =  (255,0,0)
                im = cv2.rectangle(im, (int(bbox_hand[0]), int(bbox_hand[1])), (int(bbox_hand[2]), int(bbox_hand[3])), color , 2) 
                if isinstance(segmentation_hand, str) == False:
                    # count_hand_masks = count_hand_masks+1

                    # blendMask(im, segmentation_hand, color, 0.5)
                    seg_hand = seg_to_poly(bbox_hand[0], bbox_hand[1], bbox_hand[2]-bbox_hand[0], bbox_hand[3]-bbox_hand[1], segmentation_hand)
                    # pdb.set_trace()
                    # cv2.fillPoly(im, seg_hand, color)
                    im = draw_poly_mask(im, seg_hand, color)
                

                if object_box.strip() != "None":
                    bbox_object = [float(x) for x in object_box.split(',')] 

                    # segmentation to test implementation /Jan 2023
                    segmentation_object = get_mask("3", count,  id, mask_dir)
                    # count_objects = count_objects + 1

                    color =  (0,0, 255)
                    im = cv2.rectangle(im, (int(bbox_object[0]), int(bbox_object[1])), (int(bbox_object[2]), int(bbox_object[3])), color , 2) 
                    if isinstance(segmentation_object, str) == False:
                        # count_object_masks = count_object_masks +1
                        segmentation_object = seg_to_poly(bbox_object[0], bbox_object[1], bbox_object[2] - bbox_object[0], bbox_object[3] - bbox_object[1], segmentation_object)
                        # cv2.fillPoly(im, segmentation_object, color)
                        # blendMask(im, segmentation_object, color, 0.5)
                        im = draw_poly_mask(im, segmentation_object, color)
                        

                    if touch_type != "None":
                        
                      
                        if "None" not in second_obj_box:
                           

                            bbox_second_obj = [float(x) for x in second_obj_box.replace("\n","").split(',')]
                            segmentation_second_object = get_mask("5", count, id, mask_dir)
                            
                            # count_seconds = count_seconds +1 
                         
                            color =  (255, 200, 0)
                            im = cv2.rectangle(im, (int(bbox_second_obj[0]), int(bbox_second_obj[1])), (int(bbox_second_obj[2]), int(bbox_second_obj[3])), color , 2) 
                            if  isinstance(segmentation_second_object, str) == False:
        
                                segmentation_second_object =  seg_to_poly(bbox_second_obj[0], bbox_second_obj[1], bbox_second_obj[2] - bbox_second_obj[0], bbox_second_obj[3] - bbox_second_obj[1], segmentation_second_object)
                               
                                im = draw_poly_mask(im, segmentation_second_object, color)
         
                    
                count += 1
            
            # cv2.imwrite(save_dir+"read_"+jpeg_file.replace(dirname, ''), im)
            

            # count_im +=1

            # if count_im > 500:
            #     write_obj_html(dir = split)
            #     break
            # # break
                
        else:
            pdb.set_trace()
    

def load_ho_voc_instances(dirname = "/w/fouhey/hands2/allMerged6Blur/" , split = None):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    
    # list_dir = "/home/evacheng/10k_subset_splits/TRAIN.txt"
    # mask_dir = "/home/evacheng/masks/"
    list_dir = "/y/evacheng/ten_K/TRAIN.txt"
    #list_dir = "/w/fouhey/hands2/allMerged6Splits/TRAIN.txt"
    mask_dir = "/y/ayhassen/allmerged/masks/"



    f = open(list_dir, "r")


    dicts = []

    total_time = 0
    process_poly_time = 0
    total_c_time = 0
    total_d_time = 0

    get_mask_time = 0

    start_time = time.time()
   

    

    # print(os.path.join(list_dir, split + ".txt"))
   
    # with open(os.path.join(list_dir, split + ".txt")) as f:
    #     fileids = [line.rstrip('\n') for line in f.readlines()]

    fileids = [line.rstrip('\n') for line in f.readlines()]
    
   

  
    
    for fileid in fileids:
       
        anno_file = os.path.join(dirname.replace("Blur", ""), fileid+".txt")
        jpeg_file = os.path.join(dirname, fileid)


     
        if os.path.exists(jpeg_file):
            with PathManager.open(anno_file) as f:
                 annos = f.readlines()

            if len(annos) == 0:
               
                continue


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
                elif grasp_type == "Later":
                    grasp = 6
                elif grasp_type == "Exten":
                    grasp = 7 
                elif grasp_type == "Other":
                    grasp = 8
                else:
                    # pdb.set_trace()
                    print("error! Got: " + grasp_type)
                    grasp = 100



                bbox_hand = [float(x) for x in hand_box.split(',')]
                
                #segmentation to test implementation /Feb 2023
                hand_mask, temp, mask_time, d_time = get_mask("2", count,  im_id, mask_dir, dirname)
                segmentation_hand, poly_time  = seg_to_poly(bbox_hand[0], bbox_hand[1], bbox_hand[2] - bbox_hand[0], bbox_hand[3] - bbox_hand[1], hand_mask)
                
                process_poly_time = process_poly_time + poly_time
                get_mask_time = get_mask_time +mask_time
                total_d_time = total_d_time + d_time

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
                    object_mask, temp, mask_time , d_time = get_mask("3", count,  im_id, mask_dir, dirname)
                    segmentation_object, poly_time = seg_to_poly(bbox_object[0], bbox_object[1], bbox_object[2] - bbox_object[0], bbox_object[3] - bbox_object[1], object_mask)
                    
                    process_poly_time = process_poly_time + poly_time
                    get_mask_time = get_mask_time +mask_time
                   
                    total_d_time = total_d_time + d_time

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

                            second_object_mask, temp, mask_time, d_time = get_mask("5", count, im_id, mask_dir, dirname)
                            segmentation_second_object, poly_time =  seg_to_poly(bbox_second_obj[0], bbox_second_obj[1], bbox_second_obj[2] - bbox_second_obj[0], bbox_second_obj[3] - bbox_second_obj[1], second_object_mask)
                            
                            process_poly_time = process_poly_time + poly_time
                            get_mask_time = get_mask_time +mask_time
                            
                            total_d_time  = total_d_time+d_time

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

    total_time = time.time() - start_time

    print("Total Number of time used: " + str(total_time))
    print("Percentage used for getting masks: " + str(round((get_mask_time/total_time), 4)*100) + "%")
    print("Percentage used for transforming masks: " + str(round((process_poly_time/total_time), 4)*100) + "%")
    print("Percentage used for reading ind masks: " + str(round((total_d_time/get_mask_time), 4)*100) + "%")

    # g = open("./time_cost.txt", "w+")
    # lines = ["Total Number of time used: " + str(total_time)+"\n", "Percentage used for getting masks: " + str(round((get_mask_time/total_time), 4)*100) + "%\n",
    #          "Percentage used for transforming masks: " + str(round((process_poly_time/total_time), 4)*100) + "%\n" ]
    
    # g.writelines(lines)
    # g.close()

          
   
    return dicts




def check_empty(anno_dir = "/w/fouhey/hands2/allMerged6/"):

    no_anno_list = []

    for file in glob.glob(anno_dir+"*.txt"):
        f = open(file, 'r')

        lines = f.readlines()

        f.close()
        
        if len(lines) == 0 or '|' not in lines[0]:
            no_anno_list.append(file)
    
    g = open("./no_anno.txt", 'w+')
    g.writelines(no_anno_list)
    g.close()

    print("******Num of files with no Annotations: " + str(len(no_anno_list))+" **************")

def check_empty_train(file_dir = "/w/fouhey/hands2/allMerged6Splits/TRAIN.txt", anno_dir = "/w/fouhey/hands2/allMerged6/"):
    f = open(file_dir, 'r')

    files = f.readlines()

    no_anno_list = []

    f.close()

    for file in files:
        g = open(anno_dir+file.replace('\n', '')+".txt", 'r')

        lines = g.readlines()
        if len(lines) == 0 or '|' not in lines[0]:
            no_anno_list.append(file)
        g.close()
    
    f = open("./train_no_anno.txt", "w+")
    f.writelines(no_anno_list)
    f.close()

    print("******Num of files with no Annotations: " + str(len(no_anno_list))+" **************")



def main():
    # # for s in ["TRAIN", "VAL", "TEST"]:
    # #     load_ho_voc_instances(split = s)
    # # check_empty()
    # check_empty_train()
    load_ho_voc_instances()
  
  


if __name__ == '__main__':
    main()