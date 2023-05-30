# CLIP-based-NSFW-Detector

The CLIP-based NSFW Detector is a 2-class model primarily trained to detect nudity or pornographic content. It provides an estimation value ranging between 0 and 1, where 1 indicates NSFW content. The detector works well with image embeddings.

Different models are available, ranging from small (`ViT-B-32`) to large models (`ViT-H-14`). Please refer to [models/README.md](models/README.md) for more details.

> **Note**
> The model files (`clip_autokeras_binary_nsfw.zip, clip_autokeras_nsfw_b32.zip, h14_nsfw.pth, violence_detection_vit_b_32.npy, violence_detection_vit_l_14.npy`) need to stay where they are. Becayse they are used in [clip_retrival](https://github.com/rom1504/clip-retrieval/tree/main) see [link](https://github.com/search?q=repo%3Arom1504%2Fclip-retrieval%20CLIP-based-NSFW-Detector&type=code).

# Local Development

To get started with local development, install the dependencies by running the following command:

```bash
pip install -r requirements.txt
```

# Training

We provide an example for training and testing the `ViT-L-14` openai model, as it's the only model for which we provide the training embeddings. You can find the training embeddings in the [Google Drive link](https://drive.google.com/file/d/1yenil0R4GqmTOFQ_GVw__x61ofZ-OBcS/view?usp=sharing). Please see [data/README.md](data/README.md) for more information.

For training and testing, refer to the notebook [Traininig_Test.ipynb](Traininig_Test.ipynb).

# Inference 

You can find inference examples in the notebook [CLIP_based_NSFW_detector.ipynb](CLIP_based_NSFW_detector.ipynb)

# Additional Resources

Here are some other useful NSFW detectors:

* https://github.com/GantMan/nsfw_model
* https://github.com/notAI-tech/NudeNet

For NSFW detection datasets, you can refer to:

* https://github.com/alex000kim/nsfw_data_scraper
* https://archive.org/details/NudeNet_classifier_dataset_v1


# LICENSE

This code and model is released under the MIT license:

Copyright 2022, Christoph Schuhmann

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

