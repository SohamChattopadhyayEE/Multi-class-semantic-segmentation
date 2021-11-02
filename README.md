# Multi-class Semantic Segmentation
This is a python-based project performing multi-class semantic segmentation task with classical UNet and different versions of it. The key idea behind the word **semantic segmentation** is recognizing and understanding contents of an image at pixel level. Semantic segmentation has versatile applications in various fields, starting from bio-medical engineering to autonomous vehicle, this has become very effective. More about semantic segmentation can be found [here](https://www.jeremyjordan.me/semantic-segmentation/). Here in this project the segmentation framework is evaluated on a large scale scene segmentation dataset named [Mapillary Vistas Dataset](https://www.mapillary.com/dataset/vistas).  

## Dataset description
The Mapillary Vistas Dataset is a large-scale street-level image dataset containing 25,000 high-resolution images annotated into 66/124 object categories of which 37/70 classes are instance-specific labels (v.1.2 and v2.0, respectively). Annotation is performed in a dense and fine-grained style by using polygons for delineating individual objects. The dataset contains images from all around the world, captured at various conditions regarding weather, season and daytime. Images come from different imaging devices (mobile phones, tablets, action cameras, professional capturing rigs) and differently experienced photographers.

## Architectures
The deep learning architectures which have been used here in this project are-
- `UNet` : https://arxiv.org/abs/1505.04597
- `R2UNet` : https://arxiv.org/abs/1802.06955
- `AttUNet` : https://arxiv.org/abs/1804.03999
- `R2AttUNet` : Embedding Attention module to R2UNet
The codes of these network architectures are given [here](https://github.com/SohamChattopadhyayEE/Multi-class-semantic-segmentation/tree/main/models/network).
