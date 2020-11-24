# Tree_Species_Recognition

This project intends to train a ML model that can predict the species of a tree given an aerial image of the tree. The approach taken is to train a deep CNN using automatically generated data. 

## The Data
The dataset used is based upon a tree inventory from the London Borough of Camden. It has detailed species and location information for each record. The dataset is cleaned to remove entries with missing locations, vacant plots or unknown species, then only species with greater than 50 locations are kept. For each of these, the co-ordinates are used to download a Google satellite image, which is labelled with the species name. Subsequently the data is split in a 7:2:1 ratio of train, validation and test subsets. The training data is augmented to increase the number of training examples.

## The Models
The project explores two avenues for finding a candidate CNN that can classify tree species. In the first instance, the VGG-16 model is used, having reviewed the relevant literature in order to have chosen a well-known, high performing model to adapt to the classification problem. The second approach involves the construction of numerous convolutional models from scratch, to see if equivalent or better performance can be achieved by a model that is less complex.
### VGG-16 Model
The VGG-16 network is preloaded with weights from ImageNet before being fine-tuned on the tree images, and the loss is optimized using the Adam algorithm.
The parameters to be varied are dropout and class weights.
### Constructed CNNs approach
This process involves constructing a range of CNNs which are fit and evaluated against the same data to find the layer configuration and parameter choices that lead to the best set of results. The basic template for these models is illustrated below.
INPUT -> [[CONV -> RELU]*2 -> MAXPOOL]*N -> [FC -> RELU] -> FC 
The number, N, of convolutional blocks ranges between 1 and 6, with each block consisting of two convolutional layers (CONV) with a ReLU activation function (RELU) followed by a pooling layer (MAXPOOL). The size of the kernel and choice of kernel initialiser within the CONV layers are to be varied between models. Dropout is added after each convolutional block and after the penultimate fully connected (FC) layer. Each model is then compiled a number of times using a different optimiser each time.
Initially, a small number of simple models are considered. Based on the results of these, a model build and test framework is developed in order to bulk assess a great quantity of models.
