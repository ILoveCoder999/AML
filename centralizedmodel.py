import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import json

class CentralizedModel(nn.Module):
    def __init__(self,num_classes=100,droupout_rate=0.1):
        super(CentralizedModel,self).__init__()
        '''
        Our idea:
        use a "backbone" pre-trained on massive amounts of data -> extract features
        add a simple "head" to adapt to  specific task.
        '''
        '''
        
        '''
        self.backbone=torch.hub.load('facebookresearch/dino:main','dino_vits16')
        self.embed_dimension=384
        self.dropout=nn.Dropout(droupout_rate)
        self.head=nn.Linear(self.embed_dimension,num_classes)
