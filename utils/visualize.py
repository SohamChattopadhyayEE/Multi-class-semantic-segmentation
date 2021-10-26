import numpy as np
import matplotlib.pyplot as plt
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show_image(imgs):
  w = 10
  h = 10
  fig = plt.figure(figsize=(8, 8))
  columns = len(imgs)
  rows = 1
  for i in range(0, columns*rows):
      img = imgs[i]
      fig.add_subplot(rows, columns, i+1)
      plt.imshow(img)
  plt.show()


def predict_images(loader, net, mode, img_id = 0):
  data = next(iter(loader))
  images, masks = data

  testImg = images[img_id]
  testMask = masks[img_id]

  testMask = testMask.reshape(-1, 400)

  out = net(testImg.reshape(1, 3, 400, 400).to(device))
  out = out.reshape(-1, 400).cpu().detach().numpy()

  images_stack = [testImg.permute(1,2,0), testMask, out]
  if mode == 'train':
    print('Prediction : Train ================================>')
  elif mode == 'val':
    print('Prediction : Validation ================================>')
  else :
    print('Prediction : Test ================================>')
  show_image(images_stack)