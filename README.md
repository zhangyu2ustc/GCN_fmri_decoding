

# Functional Decoding using graph convolutional networks on brain graphs
**Brain graphs** provide a relatively simple way of modeling the human brain connectome by associating nodes with brain regions and defining edges via anatomical or functional connections. Based on this architecture, a non-linear embedding tool, called graph Laplacian, can be used to project the high dimensional brain activities onto subspaces of the graph Laplacian eigenbasis. This method has gained more and more attention in neuroscience studies, for instance identifying functional areas and networks, generating connectivity gradients and harmonics, and even predicting atrophy patterns of dementia. Recently, **graph convolutional networks (GCN)** was proposed, which combines the graph Laplacian theory with deep learning architectures by extending convolution operations onto the graph domain. This approach has shown some promising findings in neuroscience applications, for instance parcellating brain areas and detecting alterations in AD and Autism. In our recent study, we applied GCN to annotate the spatiotemporal dynamics of brain dynamics of human cognitive functions using a short series of fMRI volumes. 


## Introduction
Brain graphs provide a relatively simple way of modeling the human brain connectome, by associating nodes with brain regions, and defining edges via anatomical or functional connections. 
#### Graph Laplacian
> Based on this architecture, a non-linear embedding tool, called graph Laplacian, can be used to project the high dimensional brain activities onto subspaces of the graph Laplacian eigenbasis.
This method has gained more and more attention in neuroscience studies, for instance identifying functional areas and networks, generating connectivity gradients and harmonics, and even predicting atrophy patterns of dementia. 
#### Graph Convolutional Networks
> Recently, graph convolutional networks (GCN) was proposed, which combines the graph Laplacian theory with deep learning architectures by extending convolution operations onto the graph domain. 
> This approach has shown some promising findings in neuroscience applications, for instance parcellating brain areas and detecting alterations in AD and Autism. 

> In our recent study, we applied GCN[1,2] to annotate the spatiotemporal dynamics of brain dynamics of human cognitive functions using a short series of fMRI volumes. 
I will use this as a case study to illustrate how to apply GCN to brain imaging.




\
\

## Code
 * ```notebooks``` includes all the functions and modules you need for this tutorial

 * ```notebooks/Tutorials_GCN_practice2_graph-Laplacian_GCN.ipynb``` the main notebook

 * ```notebooks/model.py``` contains the model definition, including fully-connected, 1stGCN [1] and ChebyNet [2]

 * ```notebooks/utils.py``` contains helpful functions


## References
<a id="1">[1]</a> Zhang, Yu, and Pierre Bellec. "Functional Decoding using Convolutional Networks on Brain Graphs." 2019 Conference on Cognitive Computational Neuroscience, Berlin, Germany [PDF](https://ccneuro.org/2019/proceedings/0001137.pdf)

<a id="2">[2]</a> Zhang, Yu, and Pierre Bellec. "Functional Annotation of Human Cognitive States using Graph Convolution Networks." 2019 Conference on Neural Information Processing Systems (NeurIPS) Neuro-AI workshop - Real Neurons & Hidden Units, Vancouver, Canada [PDF](https://openreview.net/pdf?id=HJenmmF8Ir)

