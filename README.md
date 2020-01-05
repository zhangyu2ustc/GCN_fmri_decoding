

# benchmark of cnn for fmri decoding
In this project, we attempts to analyze brain imaging data by training the classical deep neural network architectures from scratch. We specificially focused on brain decoding using functional MRI (fMRI). Brain decoding aims to infer cognitive functions from the recorded brain response. Here, we develop a pipeline which takes a short series of fMRI scans as input and predicts the corresponding cognitive states. We mainly tested two DNN architectures, i.e. _**3d-CNN**_ and _**ResNet50**_ 


## Introduction
Comparing to the natural images for which the DNNs are usually designed for, the neuroimaging data are much larger in size, usually in the 4D format: (X, Y, Z, time). This large feature maps bring several challenges when training DNNs. First of all, we need a large dataset to train. But fMRI signals are usually sensitive to the scanning environments. How to generate a universal representations using datasets from multiple centres is still unknown. Here, we decide to use data from one center but consisting of 1200 healthy subjects, scanned for the same task designs. Second, the large feature maps are memory consuming, which is a more serious problem when training and validating the model on GPU processors. To solve this, we propose to training the DNNs directly using CPU processors. 
#### 3d-CNN
> Our 3d-CNN achitecture consist of 10 convolutional layers, followed by global pooling and two dense layers. We used stride in convolutional layers instead of using Maxpooling after the convolutional layers in order to save memory. BatchNorm and Dropout was also used in the pipeline. The entire model has **617,238** parameters to train.
  

#### ResNet
> A more complex architecture, e.g. ResNet50, was used, which consist of 50+ convolution layers. There are two types of residual blocks, i.e. identity block, which has the shortcut directly from input to the output, and convolution block, where the input passes convolutional layers first and then sends a shortcut to the output. The residual blocks have been shown to reslove the vanishing gradient problem, which makes the training process much harder especially when DNN goes deeper. Here, we also want test whether the residual blocks will accelerate the training on large-dimensional brain imaging. The entire model has **2,510,742** parameters to train.



\
\

## Code
 * ```notebooks``` includes all the functions and modules you need for this tutorial

 * ```notebooks/Tutorials_GCN_practice2_graph-Laplacian_GCN.ipynb``` the main notebook

 * ```notebooks/model.py``` contains the model definition, including fully-connected, 1stGCN [1] and ChebyNet [2]

 * ```notebooks/utils.py``` contains helpful functions


