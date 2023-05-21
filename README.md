# CLIP-based-NSFW-Detector

CLIP based 2 class nsfw detector (mainly trained for nudity content), From small the big models can be found see `models/README.md`


# Local Development




# Training Model

## Training

The training example only for the vit-l model because the embedding for that exists see `data/README.md`

## Testing

The testing exampe onlt for the vit-l models bacuse the embedding for that exists see `data/README.md`

# Inference 

Inference Examples can be found in DEMO-Colab https://colab.research.google.com/drive/19Acr4grlk5oQws7BHTqNIK-80XGw2u8Z?usp=sharing

Or following files.

# Additional Resources

Additionall usefull nsfw detectors

* https://github.com/GantMan/nsfw_model
* https://github.com/notAI-tech/NudeNet

The dataset for nsfw

* https://github.com/alex000kim/nsfw_data_scraper
* https://archive.org/download/NudeNet_classifier_dataset_v1


# Disclamier 

I am outsider try to improve the repo to make it more usefull for the others. Some of the information provided my be wrong so keep it in mind.

This 2 class NSFW-detector is a lightweight Autokeras model that takes CLIP ViT L/14 embbedings as inputs.
It estimates a value between 0 and 1 (1 = NSFW) and works well with embbedings from images.



# LICENSE

This code and model is released under the MIT license:

Copyright 2022, Christoph Schuhmann

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.






The training CLIP V L/14 embbedings can be downloaded here:
https://drive.google.com/file/d/1yenil0R4GqmTOFQ_GVw__x61ofZ-OBcS/view?usp=sharing (not fully manually annotated so cannot be used as test)


The (manually annotated) test set is there https://github.com/LAION-AI/CLIP-based-NSFW-Detector/blob/main/nsfw_testset.zip

https://github.com/rom1504/embedding-reader/blob/main/examples/inference_example.py inference on laion5B

Example of use of the model:

```python
@lru_cache(maxsize=None)
def load_safety_model(clip_model):
    """load the safety model"""
    import autokeras as ak  # pylint: disable=import-outside-toplevel
    from tensorflow.keras.models import load_model  # pylint: disable=import-outside-toplevel

    cache_folder = get_cache_folder(clip_model)

    if clip_model == "ViT-L/14":
        model_dir = cache_folder + "/clip_autokeras_binary_nsfw"
        dim = 768
    elif clip_model == "ViT-B/32":
        model_dir = cache_folder + "/clip_autokeras_nsfw_b32"
        dim = 512
    else:
        raise ValueError("Unknown clip model")
    if not os.path.exists(model_dir):
        os.makedirs(cache_folder, exist_ok=True)

        from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel

        path_to_zip_file = cache_folder + "/clip_autokeras_binary_nsfw.zip"
        if clip_model == "ViT-L/14":
            url_model = "https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/clip_autokeras_binary_nsfw.zip"
        elif clip_model == "ViT-B/32":
            url_model = (
                "https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/clip_autokeras_nsfw_b32.zip"
            )
        else:
            raise ValueError("Unknown model {}".format(clip_model))  # pylint: disable=consider-using-f-string
        urlretrieve(url_model, path_to_zip_file)
        import zipfile  # pylint: disable=import-outside-toplevel

        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(cache_folder)

    loaded_model = load_model(model_dir, custom_objects=ak.CUSTOM_OBJECTS)
    loaded_model.predict(np.random.rand(10**3, dim).astype("float32"), batch_size=10**3)

    return loaded_model
    
    
nsfw_values = safety_model.predict(embeddings, batch_size=embeddings.shape[0])
```
