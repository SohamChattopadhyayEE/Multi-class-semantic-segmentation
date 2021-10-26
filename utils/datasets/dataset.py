from os.path import splitext
from os import listdir
import os
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from PIL import Image
import cv2
import matplotlib.pyplot as plt
#from google.colab.patches import cv2_imshow

class CustomDataset(Dataset):
  def __init__(self, img_dir, mask_dir, transform = False):

    self.img_dir = img_dir
    self.mask_dir = mask_dir
    self.transform = transform

    self.list_images = os.listdir(img_dir)


  def __len__(self):
    return len(self.list_images)

  def __getitem__(self, idx):
    image_path = os.path.join(self.img_dir, self.list_images[idx])
    mask_path = os.path.join(self.mask_dir, self.list_images[idx].replace(".jpg", ".png"))

    #img = np.array(Image.open(image_path).convert("RGB"))
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('uint8')
    #mask = np.array(Image.open(mask_path).convert("RGB"), dtype = np.float32)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    mask = mask[:, :, 0]
    arr = np.unique(mask)
    for i in range(len(arr)):
      mask[mask == arr[i]] = i;


    if self.transform  :
      img = cv2.resize(img, (400, 400))
      mask = cv2.resize(mask, (400, 400))

    img = torch.from_numpy(img)
    mask = torch.from_numpy(mask)

    img = img.permute(2, 0, 1)
    img = img.float()/255
    mask = mask.long().unsqueeze(0)

    return img, mask