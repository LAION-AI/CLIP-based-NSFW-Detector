# CLIP-based-NSFW-Detector

This 2 class NSFW-detector is a lightweight Autokeras model that takes CLIP ViT L/14 embbedings as inputs.
It estimates a value between 0 and 1 (1 = NSFW) and works well with embbedings from images.

DEMO-Colab:
https://colab.research.google.com/drive/19Acr4grlk5oQws7BHTqNIK-80XGw2u8Z?usp=sharing

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
