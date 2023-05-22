# Ref: https://github.com/LAION-AI/CLIP-based-NSFW-Detector/issues/7#issue-1501095277

from collections import OrderedDict
import zipfile
import os

import torch
import autokeras as ak
import torch.nn.functional as F

from tensorflow.keras.models import load_model


from src.models import NSFWModel_B32, NSFWModel_L14

torch.manual_seed(8492)

# Convert Keras mode to Pytorch -> ViT-B32

# For testing
embed = F.normalize(torch.randn([4, 512])).numpy() 

# Unzip
b32_path = './old/clip_autokeras_nsfw_b32.zip'
with zipfile.ZipFile(b32_path, 'r') as zip_ref:
    zip_ref.extractall('./old')

# Keras Model
model_keras = load_model(os.path.splitext(b32_path)[0], 
                         custom_objects=ak.CUSTOM_OBJECTS)
model_out_keras = model_keras.predict(embed)

# Create a state dict for pytorch
state_dict = OrderedDict()
state_dict['norm.mean'] = torch.tensor(model_keras.layers[2].weights[0].numpy())
state_dict['norm.variance'] = torch.tensor(model_keras.layers[2].weights[1].numpy())
state_dict['linear_1.weight'] = torch.tensor(model_keras.layers[3].weights[0].numpy()).T
state_dict['linear_1.bias'] = torch.tensor(model_keras.layers[3].weights[1].numpy())
state_dict['linear_2.weight'] = torch.tensor(model_keras.layers[5].weights[0].numpy()).T
state_dict['linear_2.bias'] = torch.tensor(model_keras.layers[5].weights[1].numpy())
state_dict['linear_3.weight'] = torch.tensor(model_keras.layers[8].weights[0].numpy()).T
state_dict['linear_3.bias'] = torch.tensor(model_keras.layers[8].weights[1].numpy())

# Pytorch Model
model_pytorch = NSFWModel_B32()
model_pytorch.load_state_dict(state_dict)

model_pytorch.eval()
model_out_pytorch = model_pytorch(torch.tensor(embed)).detach().numpy()

# Print the difference
print('ViT-B32 | Difference between keras and torch models is :', model_out_pytorch - model_out_keras)

# Save the model
torch.save(model_pytorch.state_dict(), './models/clip_ViT-B-32_openai_binary_nsfw_head.pth')

# Convert Keras mode to Pytorch -> ViT-L14

# For testing
embed = F.normalize(torch.randn([4, 768])).numpy()

# Unzip
l14_path = './old/clip_autokeras_binary_nsfw.zip'
with zipfile.ZipFile(l14_path, 'r') as zip_ref:
    zip_ref.extractall('./old')

# Keras Model
model_keras = load_model(os.path.splitext(l14_path)[0], 
                         custom_objects=ak.CUSTOM_OBJECTS)
model_out_keras = model_keras.predict(embed)

# Create a state dict for pytorch
state_dict = OrderedDict()
state_dict['norm.mean'] = torch.tensor(model_keras.layers[2].weights[0].numpy())
state_dict['norm.variance'] = torch.tensor(model_keras.layers[2].weights[1].numpy())
state_dict['linear_1.weight'] = torch.tensor(model_keras.layers[3].weights[0].numpy()).T
state_dict['linear_1.bias'] = torch.tensor(model_keras.layers[3].weights[1].numpy())
state_dict['linear_2.weight'] = torch.tensor(model_keras.layers[5].weights[0].numpy()).T
state_dict['linear_2.bias'] = torch.tensor(model_keras.layers[5].weights[1].numpy())
state_dict['linear_3.weight'] = torch.tensor(model_keras.layers[7].weights[0].numpy()).T
state_dict['linear_3.bias'] = torch.tensor(model_keras.layers[7].weights[1].numpy())
state_dict['linear_4.weight'] = torch.tensor(model_keras.layers[9].weights[0].numpy()).T
state_dict['linear_4.bias'] = torch.tensor(model_keras.layers[9].weights[1].numpy())

# Pytorch Model
model_pytorch = NSFWModel_L14()
model_pytorch.load_state_dict(state_dict)
model_pytorch.eval()

model_out_pytorch = model_pytorch(torch.tensor(embed)).detach().numpy()
# Print the difference
print('ViT-L14 | Difference between keras and torch models is :', model_out_pytorch - model_out_keras)

# Save the model
torch.save(model_pytorch.state_dict(), './models/clip_ViT-L-14_openai_binary_nsfw_head.pth')
