# Models

To obtain the backbones, please visit the [open_clip](https://github.com/mlfoundations/open_clip) or [hugginface clip](https://huggingface.co/laion) repositories.

The model weights were initially in `AutoKeras` format and were later transferred to `PyTorch` as described in [issue-7](`https://github.com/LAION-AI/CLIP-based-NSFW-Detector/issues/7#issue-1501095277`).

## Existing models

Currently, all the existing models are binary classifiers for NSFW (1) or SFW (0) classification.

To determine which head corresponds to which model, refer to the naming convention of the heads: `clip_{$MODEL_NAME}_{$TRAINING_DATA}_binary_nsfw_head.pth`
* `$MODEL_NAME` represents the CLIP model backbone.
* `$TRAINING_DATA` indicates the training data used to train the backbone.

For more information about the head architecture, you can check [src/models.py](../src/models.py).