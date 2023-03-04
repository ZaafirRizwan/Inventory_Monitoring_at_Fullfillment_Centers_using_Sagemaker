#Dependencies
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import tempfile
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import logging
import sys
import time
import json
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

import argparse

# For Profiling
from smdebug import modes
from smdebug.pytorch import get_hook



#  For Debugging
import smdebug.pytorch as smd



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def save_model(model, model_dir,image_dataset_train):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save({
            'model_state_dict': model.cpu().state_dict(),
            'class_to_idx' : image_dataset_train.class_to_idx
            }, path)
        


def test(model, test_loader,criterion,hook):
    '''
    This function takes two arguments and returns None
    
    Parameters:
        -model: Trained Image Classification Network
        -test_loader: DataLoader for test dataset
        -hook: hook for saving model tensors during testing for ananlyzing model behavior
        
    Returns:
        None
    '''
    
    # Setting SMDEBUG hook for testing Phase
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    test_loss = 0
    correct = 0
    loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output,target)
            test_loss += loss.item() # sum up batch loss
            pred = output.argmax(dim=1,keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, criterion, optimizer, epoch,hook,args):
    '''
    This function takes five arguments and returns None
    
    Parameters:
        -model: Untrained Image Classification Network
        -train_loader: DataLoader for train dataset
        -criterion: Loss Function
        -optimizer: The optimization algorithm to use
        -epoch: Epoch Number
        -hook: hook for saving model tensors during training for ananlyzing model behavior
        
    Returns:
        None
    '''
    
    model.fc.require_grad = True
    model. train()
    # Setting SMDEBUG hook for model training loop
    hook.set_mode(smd.modes.TRAIN)
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    
def net():
    '''
    This function takes zero parameters and returns an Untrained Network
    
    Parameters:
        None
        
    Returns:
        Untrained Image Classification Model
        
    '''
    pretrained_model = models.resnet18(pretrained=True)
    
    # Freezing Pretrained Weights
    for param in pretrained_model.parameters():
        param.requires_grad = False
    
    # Append Fully_Connected layer
    num_ftrs = pretrained_model.fc.in_features
    pretrained_model.fc = nn.Linear(num_ftrs, 5)

    model_ft = pretrained_model.to(device)
    
    return pretrained_model

def create_data_loaders(data, batch_size):
    '''
        This Function Creates Data_loaders. It takes two arguments data and batch_size and return dataloaders
        
        arguments:
            data: path of data
            batch_size: number of images to pass to the network at a time
            
        return:
            train_dataloader and test_dataloader
    
    '''
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')

    train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(size=224,scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])
            ])

    test_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                        std = [0.229, 0.224, 0.225])
                ])

    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    return train_data_loader, test_data_loader,train_data

def main(args):

    
    logger.info(f'Hyperparameters are LR: {args.lr}, Batch Size: {args.batch_size}')
    logger.info(f'Data Paths: {args.data}')
    
    '''
    Creating DataLoaders
    '''
    train_loader, test_loader,train_dataset = create_data_loaders(args.data, args.batch_size)
    
    '''
    Initializing model by calling the net function
    '''
    model = net()
    
    '''
    Creatring Hook for Debugging and Profiling
    '''
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    
    '''
    Creating loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    
    
    epoch_times = []
    for epoch in range(args.epochs):
        start = time.time()
        train(model, train_loader, loss_criterion, optimizer, epoch, hook,args)
        test(model, test_loader, loss_criterion, hook)
        save_model(model, args.model_dir,train_dataset)
        
        epoch_time = time.time() - start
        epoch_times.append(epoch_time)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=3,metavar="N",help="Number of epochs for training")
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    print(args)
    
    main(args)
