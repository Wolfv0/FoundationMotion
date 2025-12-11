# Hands23 Beta Version
Hi, thanks for your interest in using Hands23 hand-object (and second-object) detector!

### Environment
Main dependency: pytorch, detectron2

### Download Model Weights
- Option1: model trained on hands23 datasets.
```
wget https://www.dropbox.com/s/t1x6upq91sept98/final_on_blur_model_0399999.pth\?dl\=0 -O final_on_blur_model_0399999.pth
```

- Option2: the model above + finetune on Ego4D.

Note: The model is finetuned on available GT labels on Ego4D (hand box, and hand side) and the others using pseudo-labels from model (above) predictions. **Use this model if you are only interested in the hand box and hand side in Ego4D. We found the performance on other predictions is degraded because of no GT during Ego4D finetuning.**
```
wget https://www.dropbox.com/s/sh5f2bcrwmrwfsm/ego4d_model_0109999.pth?dl=0 -O ego4d_model_0109999.pth
```

### Demo (using GPU)
Change to your `model_weights` and `data_dir` path before running this demo
```
CUDA_VISIBLE_DEVICES=0 python demo.py
```
