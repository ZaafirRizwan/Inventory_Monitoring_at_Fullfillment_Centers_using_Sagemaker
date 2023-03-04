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
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import logging
import sys
import time
import json
import base64

import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """    
    
    if request_content_type == "application/json":

        deserialized_data = json.loads(request_body)

        plt.imsave("image.png",deserialized_data['arr'])

        data = Image.open("image.png").convert('RGB')
        
        test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])
            ])

        
        train_inputs = test_transform(data)
    
        return train_inputs

    
def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model['model'].to(device)
    input_data = torch.unsqueeze(input_data,0).to(device)
    model['model'].eval()
    with torch.no_grad():      
        return {"prediction": model['model'](input_data),"class": model['class']}
    

def output_fn(prediction_output, response_content_type):    
    if response_content_type == "application/json":
        result = nn.functional.softmax(prediction_output['prediction'],dim=1)

        prob = torch.topk(result, 5)[0][0].tolist()
        indices = torch.topk(result, 5)[1][0].tolist()

        for i in range(len(indices)):
            for key, val in prediction_output['class'].items():
                if indices[i] == val:
                    indices[i] = key
        
        
        temp = {"prob":prob,"indices":indices,"class":prediction_output['class']}
        
        
        data = {'body': temp}

        # Serialize the data using the JSONSerializer
        serialized_data = json.dumps(data)
        
        return serialized_data
    
    
    
def net():
    '''
    This function takes zero parameters and returns a Network
    
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
    
    return model_ft
    

    
def model_fn(model_dir):
    model = net()
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        checkpoint = torch.load(f)
        model.load_state_dict(checkpoint['model_state_dict'])
        class_to_idx = checkpoint['class_to_idx']
        
    return {"model":model,"class":class_to_idx}  