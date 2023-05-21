# Models

Only heads are there to build the head see `example_inference.ipny`, the data for the trained models only for ViT-L model exists hopefully in the future I will add the other ones. 

To get the backbones you need to visit open_clip or hugginface clip.

The models weights are initially in autokeras and transfered to pytorch by `https://github.com/LAION-AI/CLIP-based-NSFW-Detector/issues/7#issue-1501095277` the cleaned version can be found in `src/convert_heads_to_pytorch.py`



## Existing models

currently all the existing models are binary classifier nsfw (1) or not (0)

How to know which head for which?
    - naming of the heards are `clip_{$MODEL_NAME}_{$TRAINING_DATA}_binary_nsfw_head.pth`
    Model name is the which model is used
    training data is the which training data is used for the model

Head arhitecture can be found in `src/models.py`.