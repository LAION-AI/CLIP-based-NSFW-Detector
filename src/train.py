
import torch
from torch.optim import AdamW
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from .models import NSFWModel_B32, NSFWModel_L14, NSFWModel_H14
from .utils import cross_entropy

def training_step_embeddings(dataloader, model, optimizer, loss):
    model.train()
    
    mean_loss = 0.0
    for x, y in tqdm(dataloader):
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()
        
        # print(x)
        y_p = model(x)
        l = loss(y, y_p)

        l.backward()
        optimizer.step()
        mean_loss += l.item()
    return mean_loss / len(dataloader)

@torch.no_grad()
def validation_step_embeddings(dataloader, model, loss):
    model.eval()
    mean_loss = 0.0
    for x, y in dataloader:
        x, y = x.cuda(), y.cuda()
        y_p = model(nn.functional.normalize(x))
        l = loss(y, y_p)
        mean_loss += l.item()
    
    return mean_loss / len(dataloader)



def train_with_embeddings(X, y, head_type='', test_size=0.2, random_state=None, epochs=5, batch_size=256):

    # split data to train and validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # train dataloader
    dataloader_train = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float)), 
                                  batch_size=batch_size, shuffle=True)

    # test dataloader
    dataloader_val = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float)), 
                                  batch_size=batch_size, shuffle=False)

    # create model
    if head_type == 'ViT-B-32':
        model = NSFWModel_B32()

        model.norm.mean = torch.tensor(X_train.mean(axis=0), dtype=torch.float)
        model.norm.variance = torch.tensor(X_train.var(axis=0), dtype=torch.float)
    elif head_type == 'ViT-L-14':
        model = NSFWModel_L14()

        model.norm.mean = torch.tensor(X_train.mean(axis=0), dtype=torch.float)
        model.norm.variance = torch.tensor(X_train.var(axis=0), dtype=torch.float)
    else:
        model = NSFWModel_H14()

    model.cuda()



    # create optimizers
    optimizer = AdamW(model.parameters(), lr=1e-5)
    loss = cross_entropy

    for epoch in range(epochs):
        loss_train = training_step_embeddings(dataloader_train, model, optimizer, loss)
        loss_val = validation_step_embeddings(dataloader_val, model, loss)
        print(f'Epoch {epoch+1} | Train Loss {loss_train} | Val Loss {loss_val}')
        
        for g in optimizer.param_groups:
            g['lr'] *= 0.1
    return model


# This is not yet
def train_with_images():
    pass