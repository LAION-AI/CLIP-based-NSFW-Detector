# CLIP-based-NSFW-Detector

CLIP based 2 class nsfw detector (mainly trained for nudity content). 

It estimates a value between 0 and 1 (1 = NSFW) and works well with embbedings from images.

From small to big models with different size can be found see [models/README.md](models/README.md)


# Local Development

Install the dependicies:

```bash
pip install -r requirements.txt
```

# Training

***Note:** For training you might need a GPU*

The training example for `ViT-L-14 openai` model is provided because that is the only model with their training embedding are provided. The training embeddings can be found in [google drive](https://drive.google.com/file/d/1yenil0R4GqmTOFQ_GVw__x61ofZ-OBcS/view?usp=sharing). 

The testing dataset only provided for the `ViT-L-14 openai` model as well see [data/README.md](data/README.md)

# Inference 

Inference Examples can be found in DEMO-Colab https://colab.research.google.com/drive/19Acr4grlk5oQws7BHTqNIK-80XGw2u8Z?usp=sharing

Or following files.

# Additional Resources

Additionall usefull nsfw detectors:

* https://github.com/GantMan/nsfw_model
* https://github.com/notAI-tech/NudeNet

The dataset for nsfw detection:

* https://github.com/alex000kim/nsfw_data_scraper
* https://archive.org/details/NudeNet_classifier_dataset_v1


# Disclamier 

I am outsider try to improve the repo to make it more usefull for the others. Some of the information provided my be wrong so keep it in mind.


# LICENSE

This code and model is released under the MIT license:

Copyright 2022, Christoph Schuhmann

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

