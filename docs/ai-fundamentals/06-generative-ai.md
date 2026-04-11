# 06. Generative AI

A structured introduction to deep learning and generative AI systems, covering deep neural architectures, training breakthroughs, generative models, latent spaces, large language models, and multimodal AI.

---

## 📖 Full Article

- English  
https://zeromathai.com/en/generative-ai-systems-en/

- Korean  
https://zeromathai.com/generative-ai-systems/

👉 Other languages are available on the website

---

## 🧠 Overview

Generative AI extends deep learning beyond prediction.

Instead of only assigning labels or making classifications, generative models learn the structure of data itself and produce new outputs such as:

- images  
- text  
- audio  
- code  
- multimodal content  

This section organizes deep learning and generative AI into one coherent model landscape.

---

## 📚 What This Section Covers

This lecture connects the following topics:

- deep learning concepts and background  
- breakthroughs that made deep learning practical  
- core architectures such as CNN, DBN, and DHN  
- applications in vision, speech, and video  
- composite deep learning architectures  
- generative models and latent spaces  
- large language models  
- multimodal AI and semantic alignment  

---

## 🏗 Deep Learning: Concepts and Background

Deep learning refers to deep neural systems that transform input step by step across many layers.

### General idea

- lower layers detect simple patterns  
- middle layers detect structures and parts  
- higher layers detect semantic concepts  

This hierarchical view was inspired by neuroscience, especially the layered structure of the visual cortex.

Deep learning also replaced manual feature engineering with end-to-end learning, allowing networks to learn representations automatically.

---

## 🚀 Key Innovations and Training Breakthroughs

Training deep networks used to be difficult because of problems such as vanishing gradients.

Several innovations made practical deep learning possible:

- **Layer-wise pretraining**  
  learned lower layers first, then stacked higher ones  

- **Convolution and weight sharing**  
  reduced parameters while preserving expressiveness  

- **Pooling**  
  improved robustness and reduced dimensionality  

- **ReLU activation**  
  helped maintain useful gradients during training  

These innovations made much deeper networks trainable in practice.

---

## 🧱 Core Deep Learning Architectures

Deep learning is not one model but a family of architectures.

### CNN (Convolutional Neural Networks)
- specialized for images and spatial data  
- strong at feature extraction and recognition  

### DBN (Deep Belief Networks)
- probabilistic deep models  
- important in early unsupervised deep learning  

### DHN (Deep Hypernetworks)
- one network generates the parameters of another  
- highly flexible, but computationally complex  

Different architectures are useful for different tasks and data types.

---

## 🎯 Applications of Deep Learning

Deep learning reshaped many major AI application areas.

### Vision
- ImageNet  
- AlexNet  
- GoogLeNet  

### Speech
- deep speech recognition  
- LSTM-based sequence modeling  

### Video
- 3D CNNs  
- motion and temporal pattern modeling  

These applications established deep learning as the dominant AI paradigm in practice.

---

## 🧩 Composite and Extended Architectures

As deep learning matured, systems began combining multiple architectures.

### Main examples

- **Deep Reinforcement Learning (DQN)**  
  combines Q-learning with deep neural networks  

- **RNN / LSTM systems**  
  process temporal and sequential structure  

- **End-to-End learning**  
  learns the full mapping directly from input to output  

- **GANs**  
  use adversarial learning between generator and discriminator  

- **ResNets**  
  use residual connections to stabilize very deep training  

- **Memory Networks**  
  incorporate external memory for long-range reasoning  

These systems expanded deep learning beyond simple pattern recognition.

---

## 🎨 Generative Models and Latent Spaces

Generative AI learns the data-generating structure itself.

A central idea is the **latent space**:

- a compact hidden representation behind observed data  
- supports generation, interpolation, and reconstruction  

### Main generative models

- **VAE (Variational Autoencoders)**  
  explicitly model latent distributions and reconstruct samples  

- **GANs**  
  generate realistic samples through adversarial learning  

- **Transformer-based generative models**  
  model sequences such as text, code, and time series  

- **Diffusion Models**  
  learn to reverse noise and generate high-quality samples  

These models form the core of modern generative AI.

---

## 🗣 Large Language Models

Large Language Models (LLMs) are generative models trained on massive text corpora.

### Core mechanism
- predict the next token in context  
- use Transformer architecture and self-attention  

### Training paradigm
- self-supervised or unsupervised pretraining  
- learn language structure and semantics from raw text  

### Main limitations
- hallucination  
- knowledge cutoff  
- limited domain specialization  

### Mitigation strategies
- fine-tuning  
- retrieval-augmented generation (RAG)  

LLMs are one of the most influential forms of modern generative AI.

---

## 🌐 Multimodal AI and Semantic Alignment

Multimodal AI aligns multiple types of input such as:

- text  
- images  
- audio  
- video  

Each modality passes through its own encoder, and the system learns a shared semantic space.

### Example tasks
- image captioning  
- text-to-image generation  
- visual storytelling  
- multimodal question answering  

More recently, multimodal LLMs use language models as central reasoning engines across multiple modalities.

---

## 🔗 Related Topics

This section connects naturally to:

- Neural Networks  
- Deep Learning  
- CNNs  
- GANs  
- VAEs  
- Diffusion Models  
- Large Language Models  
- Multimodal AI  
- RAG  

---

## 💡 Why This Matters

Generative AI changes the role of AI from recognizing the world to creating within it.

It enables systems to:

- generate new content  
- model latent structure  
- reason across modalities  
- combine perception and generation  

This is why generative AI has become one of the most important areas in modern AI.

---

## ⭐ Note

This material is organized and reconstructed based on lectures by **Professor Byoung-Tak Zhang (Seoul National University)**.

---

## 🚀 Next Steps

Future documents in this repository can expand this section into:

- GANs and adversarial learning  
- diffusion models  
- large language models  
- multimodal generation  
- semantic alignment and cross-modal reasoning
