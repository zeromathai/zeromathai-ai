# 06. Recurrent Neural Networks (RNNs)

## 📌 Lecture Overview

This lecture introduces **Recurrent Neural Networks (RNNs)** through the full story of sequence modeling.

It starts from the question:

👉 How can a model understand data that unfolds over time?

Then it explains:

- why sequential data requires a different approach  
- how hidden states store information across time  
- how RNNs learn through Backpropagation Through Time  
- how language modeling enables sequence generation  
- how architectures evolved from vanilla RNN to LSTM and Seq2Seq  
- how Attention and Transformer replaced recurrence  
- how modern AI models build on these ideas  

Together, these ideas show how RNNs became the foundation of modern sequence-based AI.

---

## 📖 Full Article

- English  
  https://zeromathai.com/en/rnn-overview/

- Korean  
  https://zeromathai.com/rnn-overview/

👉 Other languages are available on the website

---

## 📚 Table of Contents

- What is sequential data?  
- Why do we need RNNs?  
- Hidden state and recurrence  
- Learning through time (BPTT)  
- Language modeling and generation  
- Evolution of sequential models  
- Attention and Transformer  
- Modern AI beyond RNN  

---

## 1. What Is Sequential Data?

Sequential data is data where **order matters**.

Examples:
- text  
- speech  
- time series  
- video  

In these cases, meaning depends on how elements are arranged over time.

For example, the sentence:

- “I love this movie”  
- “I do not love this movie”  

contains similar words, but the meaning changes because of order.

The challenge is to learn patterns that depend not only on values, but also on their temporal relationships.

---

## 2. Why Do We Need RNNs?

A standard neural network processes each input independently.

This creates a limitation:

- it cannot remember previous inputs  

RNNs solve this by introducing **memory**.

### Core idea

Instead of treating inputs separately:

👉 the model maintains a running summary of past information

This summary is called the **hidden state**.

---

## 3. Hidden State and Recurrence

The hidden state is the core component of an RNN.

At each time step:

- the model reads the current input  
- combines it with the previous hidden state  
- produces a new hidden state  

👉 Intuition:

The hidden state acts as a memory that carries context forward.

### What it captures

- previous words in a sentence  
- earlier patterns in time series  
- contextual information in speech  

However, the hidden state is compressed.

So it is useful, but not perfect, especially for long sequences.

---

## 4. Learning Through Time (BPTT)

RNNs are trained using:

👉 **Backpropagation Through Time (BPTT)**

### How it works

- the sequence is unrolled across time  
- loss is computed at each step  
- gradients are propagated backward through all time steps  

### Key challenge

Gradients can become:

- too small → vanishing gradient  
- too large → exploding gradient  

This makes it difficult for RNNs to learn long-term dependencies.

---

## 5. Language Modeling and Generation

One of the most important applications of RNNs is **language modeling**.

### Goal

Predict the next token given previous tokens

Example:

Input:  
The pen is  

Output:  
→ mightier  

### Autoregressive generation

RNNs generate sequences by:

- predicting the next token  
- feeding it back as input  
- repeating the process  

This allows the model to generate full sequences such as sentences.

---

## 6. Evolution of Sequential Models

RNNs are part of a broader family of sequence models.

### Classical approaches

- ARIMA  
- Hidden Markov Models (HMM)  
- Kalman Filter  

### Neural approaches

- 1-D CNN  
- Temporal Convolutional Networks (TCN)  
- RNN family  

### RNN extensions

- **Bidirectional RNN**  
- **LSTM**  
- **GRU**  
- **Seq2Seq**  

These models improve the ability to capture long-term dependencies and complex sequence relationships.

---

## 7. Attention and Transformer

RNN-based models compress information into a single hidden state.

This creates an **information bottleneck**.

### Attention

Attention allows the model to:

- directly focus on relevant parts of the input  

### Transformer

Transformer removes recurrence entirely and uses **self-attention**.

Advantages:

- parallel computation  
- better long-range dependency modeling  
- improved scalability  

This shift marked a major turning point in sequence modeling.

---

## 8. Modern AI Beyond RNN

Transformer-based models now dominate AI.

Examples:

- BERT  
- GPT  
- Vision Transformer (ViT)  
- multimodal models  

These models extend sequence modeling to:

- text  
- images  
- audio  
- video  

The original ideas from RNNs continue to influence how modern models understand and generate data.

---

## 🎯 Key Insight

RNNs became important because they introduced a fundamental idea:

how to model data as a sequence with memory.

They do this through:

- hidden states  
- recurrence  
- temporal learning  

Even though modern AI is dominated by Transformers, the core idea of sequence modeling originates from RNNs.

---

## 📚 Related Advanced Topics

- [Sequential Data and RNN Basics](https://zeromathai.com/en/rnn-and-sequential-data-overview/)
- [RNN Hidden State and Dynamics](https://zeromathai.com/en/rnn-dynamical-systems-hidden-state/)
- [Vanilla RNN and BPTT](https://zeromathai.com/en/vanilla-rnn-architecture-and-training/)
- [RNN Language Modeling](https://zeromathai.com/en/rnn-language-modeling/)
- [Sequential Model Evolution](https://zeromathai.com/en/sequential-models-overview/)
- [BiRNN, LSTM, GRU, and Seq2Seq](https://zeromathai.com/en/birnn-lstm-seq2seq/)
- [Attention and Transformer Architecture](https://zeromathai.com/en/attention-transformer-architecture/)
- [Post-Transformer Modern AI](https://zeromathai.com/en/post-transformer-modern-ai/)

---

## ⭐ Note

This material is organized and reconstructed based on lectures by **Professor Sungroh Yoon (Seoul National University)**.

---

## 🔗 Navigation

← Previous: [CNNs](05-convolutional-neural-networks-cnn.md)  
→ Next: [RNNs](07-transformer.md)
