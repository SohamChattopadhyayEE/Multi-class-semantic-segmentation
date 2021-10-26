import argparse
import logging
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
import warnings 
from torch.utils.data import DataLoader, random_split

from utils.datasets.dataset import CustomDataset
from utils.datasets.dataset_mapillary import mapillaryVistasLoader
from utils.visualize import predict_images
from utils.loss import get_loss_function
from models.model import segmentation_model


warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


n_classes = 65
in_channels = 3
lr = 0.001
batch_size = 1
n_epoch = 10

'''
img_dir_train = 'E:/InternshipNorway/Project_5_SemanticSegmentation/Dataset/Millipiary/Sample/Sample dataset_Millipiary/Original'
mask_dir_train = 'E:/InternshipNorway/Project_5_SemanticSegmentation/Dataset/Millipiary/Sample/Sample dataset_Millipiary/GT'
img_dir_val = 'E:/InternshipNorway/Project_5_SemanticSegmentation/Dataset/Millipiary/Sample/Sample dataset_Millipiary/Original'
mask_dir_val = 'E:/InternshipNorway/Project_5_SemanticSegmentation/Dataset/Millipiary/Sample/Sample dataset_Millipiary/GT'
'''
train_path = 'E:/InternshipNorway/Project_5_SemanticSegmentation/Dataset/Millipiary/Sample/Sample dataset_Millipiary/'
val_path = 'E:/InternshipNorway/Project_5_SemanticSegmentation/Dataset/Millipiary/Sample/Sample dataset_Millipiary/'
snapshot_path = 'E:/GITHUBProjects/medical_image_segmentation'
plot_path = snapshot_path

#train_dataset = CustomDataset(img_dir_train, mask_dir_train, transform=True)
train_dataset = mapillaryVistasLoader(train_path, img_size=(224, 224), is_transform=True)
train_loader = DataLoader(train_dataset, batch_size)

#val_dataset = CustomDataset(img_dir_val, mask_dir_val, transform=True)
val_dataset = mapillaryVistasLoader(val_path, img_size=(224,224), is_transform=True)
val_loader = DataLoader(val_dataset, batch_size)



print('OK -> 1')
models = ['UNet', 'R2UNet', 'AttUNet', 'R2AttUNet']
model_name = models[0]
net = segmentation_model(in_channels, n_classes, model_name)
net = net.to(device)
optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
if n_classes == 1 : 
  criterion = nn.BCEWithLogitsLoss()
else : 
  criterion = get_loss_function()
print('OK -> 2')


load_model=snapshot_path+'/model_'+model_name+'.pt'
loaded_flag=False
if os.path.exists(load_model):
    checkpoint=torch.load(load_model)
    net.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    print("model loaded successfully")
    print('starting training after epoch: ',checkpoint['epoch'])
    loaded_flag=True
    

def plot(val_loss,train_loss):
    plt.title("Loss after epoch: {}".format(len(train_loss)))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(list(range(len(train_loss))),train_loss,color="r",label="Train_loss")
    plt.plot(list(range(len(val_loss))),val_loss,color="b",label="Validation_loss")
    plt.legend()
    plt.savefig(os.path.join(plot_path,"/loss_"+model_name+".png"))
    #plt.figure()
    plt.close()



min_loss=99999
val_loss_gph=[]
train_loss_gph=[]


for epoch in range(n_epoch):
  net.train()
  train_loss = 0.0
  for i, data in enumerate(train_loader) :
    print(i+1)
    optimizer.zero_grad()
    image, label = data
    image, label = image.to(device), label.to(device)
    out = net(image)
    #loss = criterion(out, label.float())
    loss = criterion(out, label)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()*image.size(0)
  #train_loss_gph.append(abs(1/(train_loss/len(train_dataset))))
  train_loss_gph.append(train_loss/len(train_dataset))
  #print('Epoch : {}    Train loss : {}'.format(epoch+1, abs(1/(train_loss/len(train_dataset)))))
  print('Epoch : {}    Train loss : {}'.format(epoch+1, train_loss/len(train_dataset)))
  #predict_images(train_loader, net, 'train')


  net.eval()
  with torch.no_grad():
    val_loss = 0.0
    val_loss_abs = 0.0
    for data_val in val_loader:
      image_val, label_val = data_val
      image_val, label_val = image_val.to(device), label_val.to(device)
      out_val = net(image_val) 
      #loss = criterion(out_val, label_val.float())
      loss = criterion(out_val, label_val)
      val_loss += loss.item()*image_val.size(0)
    val_loss_gph.append(val_loss/(len(val_dataset)))
    #val_loss_abs = abs(1/(val_loss/len(val_dataset)))
    if min_loss>val_loss:
        state={
          "epoch":i if not loaded_flag else i+checkpoint['epoch'],
          "model_state":net.cpu().state_dict(),
          "optimizer_state":optimizer.state_dict(),
          "loss":min_loss,
          "train_graph":train_loss_gph,
          "val_graph":val_loss_gph,
        }
        min_loss=val_loss
        torch.save(state,os.path.join(snapshot_path,"model_"+model_name+'.pt'))
        net.to(device)
    #print('Epoch : {}    Validation loss : {}'.format(epoch+1, abs(1/(val_loss/len(val_dataset)))))
    print('Epoch : {}    Validation loss : {}'.format(epoch+1, val_loss/len(val_dataset)))
    #predict_images(val_loader, net, 'val')
  plot(val_loss_gph,train_loss_gph)
    