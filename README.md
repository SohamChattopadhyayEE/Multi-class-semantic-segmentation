# Multi-class Semantic Segmentation
This is a python-based project performing multi-class semantic segmentation task with classical UNet and different versions of it. The key idea behind the word **Semantic Segmentation** is recognizing and understanding contents of an image at pixel level. Semantic segmentation has versatile applications in various fields, starting from bio-medical engineering to autonomous vehicle, this has become very effective. More about semantic segmentation can be found [here](https://www.jeremyjordan.me/semantic-segmentation/). Here in this project the segmentation framework is evaluated on a large scale scene segmentation dataset named [Mapillary Vistas Dataset](https://www.mapillary.com/dataset/vistas).  

## Dataset description
The Mapillary Vistas Dataset is a large-scale street-level image dataset containing 25,000 high-resolution images annotated into 66/124 object categories of which 37/70 classes are instance-specific labels (v.1.2 and v2.0, respectively). Annotation is performed in a dense and fine-grained style by using polygons for delineating individual objects. The dataset contains images from all around the world, captured at various conditions regarding weather, season and daytime. Images come from different imaging devices (mobile phones, tablets, action cameras, professional capturing rigs) and differently experienced photographers.

## Architectures
The deep learning architectures which have been used here in this project are-
- `UNet` : https://arxiv.org/abs/1505.04597
- `R2UNet` : https://arxiv.org/abs/1802.06955
- `AttUNet` : https://arxiv.org/abs/1804.03999
- `R2AttUNet` : Embedding Attention module to R2UNet

The codes of these network architectures are given [here](https://github.com/SohamChattopadhyayEE/Multi-class-semantic-segmentation/tree/main/models/network).

## Dependencies
Since the entire project is based on `Python` programming language, it is necessary to have Python installed in the system. It is recommended to use Python with version `>=3.6`.
The Python packages which are in use in this project are `torch, matplotlib, numpy, pandas, OpenCV, and scikit-learn`. All these dependencies can be installed just by the following command line argument
- `pip install requirements.txt` 

## Code Implement
- ### Data paths :
      Current directory -----> data
                                 |
                                 |
                                 |               
                                 ---------------> train 
                                 |                 |
                                 |          ------- -------
                                 |          |             |
                                 |          V             V
                                 |        images        labels
                                 |
                                 |
                                 ---------------> validation    
                                 |                    |
                                 |             ------- -------
                                 |             |             |
                                 |             V             V
                                 |           images        labels
                                 |
                                 |
                                 |
                                 ---------------> test
                                                   |
                                            ------- -------
                                            |             |
                                            V             V
                                          images        labels
       
- Where the folder `images` contains original images in `.jpg` format and the folder `labels` contains corresponding labels/ground truths in `.png` format.  
- ### Training and Validation :
      -help
      
      
      PS E:\semantic_scene_segmentation> python train_val.py -help
      usage: train_val.py [-h] [-tr TR_PATH] [-val VAL_PATH] [-model_path MODEL_PATH] [-plot_path PLOT_PATH] [-image_save_path IMAGE_SAVE_PATH]
                          [-seg_m SEG_MODEL_NO] [-e EPOCHS] [-bs BATCH_SIZE] [-in_channels IN_CHANNELS] [-n_classes NUM_CLASSES]
                          [-lr LEARNING_RATE] [-mm MOMENTUM] [-o OPTIMIZER_FUNC]

      Training segmentation model

      optional arguments:
        -h, --help            show this help message and exit
        -tr TR_PATH, --tr_path TR_PATH
                              Path to the train data
        -val VAL_PATH, --val_path VAL_PATH
                              Path to the validation data
        -model_path MODEL_PATH, --model_path MODEL_PATH
                              Path where the model is going to be saved
        -plot_path PLOT_PATH, --plot_path PLOT_PATH
                              Path where the loss plots will be saved
        -image_save_path IMAGE_SAVE_PATH, --image_save_path IMAGE_SAVE_PATH
                              Paths where sample test images will be saved
        -seg_m SEG_MODEL_NO, --seg_model_no SEG_MODEL_NO
                              The segmentation model to be used
        -e EPOCHS, --epochs EPOCHS
                              Number of epochs
        -bs BATCH_SIZE, --batch_size BATCH_SIZE
                              Batch size
        -in_channels IN_CHANNELS, --in_channels IN_CHANNELS
                              Input channels
        -n_classes NUM_CLASSES, --num_classes NUM_CLASSES
                              Number of classes the segment map has
        -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                              The learning rate of the optimizer
        -mm MOMENTUM, --momentum MOMENTUM
                              The momentum of the optimizer
        -o OPTIMIZER_FUNC, --optimizer_func OPTIMIZER_FUNC
                              The optimizer
   - Run the following for training and validation 
  
      `python3 train_val.py -tr data/train/ -val data/validation/ -model_path data -seg_m 0 -e 200 -bs 50 -n_classes 65 -o RMSprop`
