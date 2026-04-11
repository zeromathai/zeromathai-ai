# 05. Convolutional Neural Networks (CNNs)

## 📌 Lecture Overview

This lecture introduces **Convolutional Neural Networks (CNNs)** through the full story of image classification.

It starts from the question:

👉 How can a computer tell images apart?

Then it explains:

- why CNNs were introduced  
- how convolution and pooling extract visual patterns  
- how CNN architectures evolved from LeNet to ResNet  
- how training is improved through augmentation and normalization  
- how feature maps and CAM help interpret what the model is looking at  

Together, these ideas show how CNNs became the dominant architecture for image understanding.

---

## 📖 Full Article

- English  
  https://zeromathai.com/en/dl-convolutional-neural-networks-cnn-en/

- Korean  
  https://zeromathai.com/dl-convolutional-neural-networks-cnn/

👉 Other languages are available on the website

---

## 📚 Table of Contents

- What is image classification?  
- Why do we need CNNs?  
- Convolution layer  
- Pooling, ReLU, and fully connected layers  
- Evolution of CNN architectures  
- Training better CNNs  
- Feature maps and CAM visualization  

---

## 1. What Is Image Classification?

Image classification is the task of assigning a single input image to a class.

Examples:
- cat  
- dog  
- car  
- airplane  

Humans do this naturally, but a computer does not see “objects.”  
It sees only arrays of pixel values.

For example, a color image of size 248 × 400 × 3 is just a structured collection of numbers.

The challenge is to learn which patterns in those numbers correspond to meaningful visual categories.

### Early approaches
Before deep learning, image classification often relied on:

- hand-crafted features such as SIFT or HOG  
- classifiers such as SVM or k-NN  

These methods were useful, but they could only use the features humans explicitly designed.

CNNs changed this by learning features directly from raw pixels.

---

## 2. Why Do We Need CNNs?

A standard MLP treats an image as one long vector.

That causes two major problems:

- the number of parameters becomes extremely large  
- the spatial structure of the image is ignored  

CNNs solve this with two key design principles:

### Local connectivity
Each neuron looks only at a small local region of the input.

### Parameter sharing
The same filter is reused across many spatial locations.

These ideas make CNNs:

- more efficient  
- better suited to images  
- more capable of learning visual structure  

CNNs are therefore designed specifically for data with spatial organization.

---

## 3. Convolution Layer

The convolution layer is the core feature extractor in a CNN.

A small filter (kernel) slides across the image and computes local weighted sums.

This produces a **feature map**.

### What filters learn
Different filters become sensitive to different patterns, such as:

- horizontal edges  
- vertical edges  
- corners  
- textures  
- local shapes  

👉 Intuition:

A filter is like a small pattern detector scanning the image.

### Important hyperparameters
- **Stride** — how far the filter moves each step  
- **Padding** — whether border information is preserved  
- **Number of filters** — determines output depth  

Together, these control output size, computational cost, and representational power.

---

## 4. Pooling, ReLU, and Fully Connected Layers

A CNN is not just convolution. Several layer types work together.

### ReLU
ReLU introduces nonlinearity by keeping positive values and zeroing negative ones.

ReLU(x) = max(0, x)

Why it matters:
- helps the network model complex patterns  
- improves gradient flow  
- supports stable deep training  

### Pooling
Pooling reduces the spatial size of feature maps.

Most common:
- max pooling  
- average pooling  

What pooling does:
- keeps strong responses  
- reduces computation  
- improves robustness to small spatial shifts  

### Fully connected (FC) layers
Near the end of a CNN, extracted features are combined by fully connected layers to produce the final classification decision.

This acts as the **classification head**.

---

## 5. Evolution of CNN Architectures

CNN architecture development can be seen as a progression toward:

- deeper networks  
- richer representations  
- more stable training  

### Major milestones

- **LeNet**  
  early proof-of-concept for handwritten digit recognition  

- **AlexNet**  
  introduced ReLU, dropout, GPU training, and launched the modern deep CNN era  

- **VGG**  
  showed that stacking many small 3×3 convolutions can improve depth and representation power  

- **GoogLeNet (Inception)**  
  improved efficiency by combining multiple filter sizes in parallel  

- **ResNet**  
  introduced residual (skip) connections, enabling very deep networks to train effectively  

👉 Key trend:

CNNs evolved by making networks deeper and more expressive while preserving trainability.

---

## 6. Training Better CNNs

Good CNN performance depends not only on architecture, but also on training techniques.

### Data Augmentation
Applies label-preserving transformations such as:

- horizontal flip  
- random crop  
- translation  
- color jitter  
- noise  

Why it matters:
- increases effective training diversity  
- reduces overfitting  
- improves robustness  

### Data Preprocessing
Input data are often organized through:

- zero-centering  
- normalization  
- per-channel standardization  

This makes optimization more stable.

### Batch Normalization
BatchNorm normalizes internal activations during training and then applies learned scale and shift parameters.

Why it helps:
- stabilizes learning  
- allows larger learning rates  
- speeds up convergence  
- improves deep-network trainability  

Together, these techniques improve both generalization and optimization stability.

---

## 7. What Is the Model Looking At?

CNNs are often interpreted using visualization tools.

### Feature maps
A feature map shows which image regions strongly activate a given filter.

Some maps emphasize:
- edges  
- contours  
- textures  
- local patterns  

### CAM (Class Activation Map)
CAM highlights which image regions contributed most to the model’s final class prediction.

Example:
- if the red region is on a cat’s face, the model is using the face for classification  
- if the background is strongly highlighted, the model may be relying on spurious context  

CAM is important because it helps us move from:

- “Is the prediction correct?”  

to:

- “Is the model looking at the right evidence?”  

---

## 🎯 Key Insight

CNNs became central to computer vision because they solve a fundamental problem:

how to learn meaningful visual features efficiently from raw pixels.

They do this through:

- local connectivity  
- shared filters  
- hierarchical feature learning  
- trainable deep architectures  

This makes CNNs one of the most important foundations of modern deep learning.

---

## 📚 Related Advanced Topics

- [Image Classification](https://zeromathai.com/en/image-classification-en/)
- [Background and Design Principles Behind CNNs](https://zeromathai.com/en/introduction-to-cnns-en/)
- [Core Components and Structure of CNNs](https://zeromathai.com/en/convolutional-layer-lec-en/)
- [Spatial Behavior of Convolution Operations and Hyperparameters](https://zeromathai.com/en/pooling-activation-layers-en/)
- [CNN Layer Composition](https://zeromathai.com/en/cnn-layer-composition-en/)
- [Evolution of Deep CNNs: AlexNet to ResNet](https://zeromathai.com/en/deep-cnn-evolution-alexnet-resnet-en/)
- [Data Processing and Normalization for CNN Training](https://zeromathai.com/en/cnn-data-processing-normalization-en/)

---

## ⭐ Note

This material is organized and reconstructed based on lectures by **Professor Sungroh Yoon (Seoul National University)**.

---

## 🔗 Navigation

← Previous: [MLP](04-mlp-representation-and-backprop.md)  
→ Next: [RNNs](06-rnn.md)
