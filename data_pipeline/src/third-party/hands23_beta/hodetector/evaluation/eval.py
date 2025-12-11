# import pdb
# import numpy as np
# import pycocotools.mask as mask_util
# from detectron2.structures import Boxes, BoxMode, pairwise_iou

# from detectron2.data import MetadataCatalog
# from detectron2.evaluation import COCOEvaluator
# from detectron2.evaluation.coco_evaluation import instances_to_coco_json

# from hos.evaluation.hos_postprocessing import hos_postprocessing, combineHO_postprocessing

# class CosEvaluator(COCOEvaluator):
    
#     def __init__(self, dataset_name, output_dir=None, eval_task=None, tasks=None):
#         super().__init__(dataset_name, output_dir=output_dir)
#         self.eval_task = eval_task
#         assert self.eval_task in ['hand_obj', 'handside', 'contact', 'combineHO'], f"Error: target not in ['hand_obj', 'handside', 'contact', 'combineHO']"
#         print(f'**Evaluation target: {self.eval_task}')
#         if tasks is not None:
#             self._tasks = tasks
#         self._metadata = MetadataCatalog.get(dataset_name)
#         print(f'dataset name = {dataset_name}')
#         print(f'meta data = {self._metadata}')

        
#     def process(self, inputs, outputs):
#         """
#         Re-format the inputs and outputs to make handside and contact predictions as another 4 classes. 
    
#         Args:
#             inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
#                 It is a list of dict. Each dict corresponds to an image and
#                 contains keys like "height", "width", "file_name", "image_id".
#             outputs: the outputs of a COCO model. It is a list of dicts with key
#                 "instances" that contains :class:`Instances`.
#         """
        
#         for input, output in zip(inputs, outputs):
#             prediction = {"image_id": input["image_id"]}
            
#             # pdb.set_trace()
#             # print(f'input = {input}')
#             # print(f'output = {output}\n')

#             if "instances" in output:
#                 tmp = output
#                 instances = output['instances'].to(self._cpu_device)
#                 if self.eval_task == 'hand_obj':
#                     # post-processing: link hand and obj
#                     output["instances"] = hos_postprocessing(instances)
#                 elif self.eval_task in ['handside', 'contact']:
#                     # only keep hand preds
#                     output["instances"] = instances[instances.pred_classes==0]
               

#                 prediction["instances"] = instances_to_coco_json_handside_or_contact(output["instances"], input["image_id"], eval_task=self.eval_task)
                
#             if "proposals" in output:
#                 prediction["proposals"] = output["proposals"].to(self._cpu_device)
#             if len(prediction) > 1:
#                 self._predictions.append(prediction)
                
        
        
        
# def instances_to_coco_json_handside_or_contact(instances, img_id, eval_task=None):
#     """
#     Dump an "Instances" object to a COCO-format json that's used for evaluation.
#     Args:
#         instances (Instances):
#         img_id (int): the image id
#     Returns:
#         list[dict]: list of json annotations in COCO format.
#     """
#     # print(f'instances = {len(instances)}')
#     num_instance = len(instances)
#     if num_instance == 0:
#         return []
    
#     assert eval_task in ['hand_obj', 'handside', 'contact', 'combineHO'], "Error: evaluation target should be either 'hand_obj', 'handside', 'contact', 'combineHO'"

#     boxes = instances.pred_boxes.tensor.numpy()
#     boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
#     boxes = boxes.tolist()
#     if eval_task in ['hand_obj', 'combineHO']:
#         scores = instances.scores.tolist()
#         classes = instances.pred_classes.tolist()
#     else:
#         if eval_task == 'handside':
#             preds = instances.pred_handsides.numpy()
#         elif  eval_task == 'contact':
#             preds = instances.pred_contacts.numpy()
#         scores = np.max(preds, axis=1).tolist()
#         classes = np.argmax(preds, axis=1).tolist()
    

#     has_mask = instances.has("pred_masks")
#     if has_mask:
#         # use RLE to encode the masks, because they are too large and takes memory
#         # since this evaluator stores outputs of the entire dataset
#         rles = [
#             mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
#             for mask in instances.pred_masks
#         ]
#         for rle in rles:
#             # "counts" is an array encoded by mask_util as a byte-stream. Python3's
#             # json writer which always produces strings cannot serialize a bytestream
#             # unless you decode it. Thankfully, utf-8 works out (which is also what
#             # the pycocotools/_mask.pyx does).
#             rle["counts"] = rle["counts"].decode("utf-8")

#     has_keypoints = instances.has("pred_keypoints")
#     if has_keypoints:
#         keypoints = instances.pred_keypoints

#     results = []
#     for k in range(num_instance):
#         result = {
#             "image_id": img_id,
#             "category_id": classes[k],
#             "bbox": boxes[k],
#             "score": scores[k],
#         }
#         if has_mask:
#             result["segmentation"] = rles[k]
#         if has_keypoints:
#             # In COCO annotations,
#             # keypoints coordinates are pixel indices.
#             # However our predictions are floating point coordinates.
#             # Therefore we subtract 0.5 to be consistent with the annotation format.
#             # This is the inverse of data loading logic in `datasets/coco.py`.
#             keypoints[k][:, :2] -= 0.5
#             result["keypoints"] = keypoints[k].flatten().tolist()
#         results.append(result)
#     return results

