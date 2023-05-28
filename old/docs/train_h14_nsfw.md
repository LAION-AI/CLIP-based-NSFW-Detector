This is a basic guide that outlines what is necessary to train a NSFW detector
on top of a CLIP model.

Note that this is a general guide and meant to be used as a guideline.


## Dataset Prep
You will need to obtain thousands of NSFW and SFW images to train a good model.

Once the images are gathered, you can place them in a directory and leverage
[clip-retrieval](https://github.com/rom1504/clip-retrieval#clip-inference)
to easily create a numpy array of embeddings. (For more details read the docs)

Finally, you will need to create target values which can be done easily in the
python interpreter.

### Target Values & Dataset Combination

```python
# find out how many positive samples you have
pos_x = np.load(path/to_positive/samples.npy)
neg_x = np.load(path/to_negative/samples.npy)

num_pos = pos_x.shape[0]
num_neg = neg_x.shape[0]

# create target values
pos_y = np.ones((num_pos, 1))
neg_y = np.zeros((num_neg, 1))

# combine the x samples
# NOTE: we will rely on torch dataloader shuffling to break the ordering here
x = np.vstack((pos_x, neg_x))
y = np.vstack((pos_y, neg_y))

# save the dataset x & y

np.save("train_x.npy", x)
np.save("train_y.npy", y)
```

## Model Training

Thankfully it is possible to use a very simple linear model to train the NSFW
detector.

For the purposes of this guide we will reference [this repo](https://github.com/christophschuhmann/improved-aesthetic-predictor)
and its model architecture.

> NOTE: It is also possible to utilize the training script provided in that repo
> as boilerplate code, provided you have `.npy` files for your dataset's x & y

### Model Architecture

Feel free to tweak the model architecture here, but the important thing to
remember is that your input dimension should match the dimension of your CLIP
embeddings, and your output dimension should be 1.

```python
import torch.nn as nn

class H14_NSFW_Detector(nn.Module):
    def __init__(self, input_size=1024):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)
```

### Training Snippets

Below is a list of snippets that should walk you through the general steps
necessary to train a MLP in PyTorch, however, it is completely fine to replace
the existing MLP in [this-repo](https://github.com/christophschuhmann/improved-aesthetic-predictor)
with the model provided above & begin training.

For those of you who wish to build out custom code, the following snippets should
get the ball rolling...

#### Import the necessary libraries:

```python
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
```

#### Define the dataset and data loaders:

```python
# Define the dataset
x = torch.from_numpy(np.load("train_x.npy"))
y = torch.from_numpy(np.load("train_y.npy"))
train_dataset = TensorDataset(x,y)

# Define the data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
```

#### Initialize the model
```python
model = H14_NSFW_Detector()
```

#### Define the loss function
```python
criterion = nn.MSELoss()
```

#### Define the optimizer
```python
# Define the optimizer
optimizer = Adam(model.parameters())
```

#### Define the training loop
```python
# Define the number of epochs
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
```

#### Define an evaluation loop
```python
# Evaluation loop
model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)
```

Note that this is a basic guide and you may need to add additional functionality
such as model saving and loading, early stopping, etc.
You may want to adjust the learning rate, 
batch size, and number of epochs as well.

