# 06. Recurrent Neural Networks (RNNs)

## 📌 Lecture Overview

This lecture introduces **Recurrent Neural Networks (RNNs)** through the full story of **sequence modeling**.

It starts from one simple question:

👉 How can a neural network understand data where order matters?

RNNs are designed for data such as:

- text  
- speech  
- time series  
- music  
- video  

In these data types, meaning comes from how inputs are connected over time.

This lecture explains:

- why sequential data needs a different modeling approach  
- how RNNs use hidden states as memory  
- how RNNs learn through time (BPTT)  
- how language modeling enables sequence generation  
- why vanilla RNNs fail on long dependencies  
- how LSTM, GRU, Seq2Seq extend RNN  
- how Attention and Transformer replace recurrence  
- how this leads to modern AI models  

Together, these ideas form the bridge from classical neural networks to modern foundation models.

---

## 📚 Table of Contents

- Why sequence modeling matters  
- RNN hidden state and memory  
- Vanilla RNN and learning through time  
- RNN language modeling  
- Sequential model evolution  
- BiRNN, LSTM, GRU, Seq2Seq  
- Attention and Transformer  
- Transformer-based modern AI  

---

## 1. Why Sequence Modeling Matters

Most neural networks assume inputs are independent.

But many real-world data types are not:

- sentence meaning depends on word order  
- speech depends on time flow  
- stock prices depend on previous values  
- video depends on frame sequence  

👉 This is **sequential data**

Key idea:

👉 The current input depends on previous inputs

That is why sequence modeling is needed.

---

## 2. RNN Hidden State and Memory

RNN introduces memory through the **hidden state**.

At each step:

- read input  
- combine with previous state  
- update state  

👉 Hidden state = compressed past information

### Intuition

- sentence → accumulated meaning  
- time series → pattern so far  
- speech → acoustic context  

Limitation:

- compressed → lossy  
- long-term memory is weak  

---

## 3. Vanilla RNN and Learning Through Time

Vanilla RNN repeats the same update rule over time.

This is called **recurrence**.

Training uses:

👉 Backpropagation Through Time (BPTT)

Process:

- unfold sequence  
- compute loss across time  
- propagate gradients backward  

### Problem

Gradients can:

- shrink → vanishing gradient  
- explode → unstable learning  

👉 Result:

RNN struggles with long-term dependencies.

---

## 4. RNN Language Modeling

RNN is widely used for **language modeling**.

Goal:

👉 predict next token from previous tokens

Example:

The pen is → mightier

### Autoregressive generation

- predict next token  
- feed it back  
- repeat  

Used in:

- text generation  
- early chat systems  
- sequence generation  

---

## 5. Sequential Model Evolution

Sequence modeling includes multiple approaches:

- ARIMA (time series)  
- HMM, Kalman Filter (probabilistic)  
- 1-D CNN, TCN (convolutional)  
- RNN family  
- Attention models  
- Transformer  

All solve the same problem:

👉 how to use past information

Different answers:

- RNN → hidden state  
- Attention → direct access  
- Transformer → full connectivity  

---

## 6. BiRNN, LSTM, GRU, Seq2Seq

### Bidirectional RNN

- forward + backward processing  
- captures full context  

### LSTM / GRU

- use gates  
- control memory flow  
- solve long dependency problem  

### Seq2Seq

- sequence → sequence mapping  
- encoder–decoder structure  

Applications:

- translation  
- speech recognition  
- summarization  

---

## 7. Attention and Transformer

Problem in RNN:

👉 compressing everything into one vector

Solution:

👉 Attention

- directly access relevant inputs  

### Self-attention

- every token interacts with every other token  

### Transformer

- removes recurrence  
- uses self-attention  

Advantages:

- parallel computation  
- long-range dependency handling  
- scalable architecture  

---

## 8. Transformer-Based Modern AI

Transformer enables:

- BERT (understanding)  
- GPT (generation)  
- Vision Transformer  
- multimodal models  
- text-to-image generation  
- conversational AI  

Key idea:

👉 sequence modeling becomes universal

Used across:

- text  
- image  
- audio  
- video  

---

## 🎯 Key Insight

RNN introduced the core idea:

👉 modeling data as a sequence with memory

This connects:

- hidden state  
- sequence modeling  
- language modeling  
- Seq2Seq  
- Attention  
- Transformer  
- modern AI  

Even though Transformers dominate today:

👉 RNN is the conceptual starting point

---

## 📚 Related Advanced Topics

- Sequential Data and RNN Basics  
- RNN Hidden State and Dynamics  
- Vanilla RNN and BPTT  
- RNN Language Modeling  
- Sequential Model Evolution  
- BiRNN, LSTM, GRU, Seq2Seq  
- Attention and Transformer  
- Post-Transformer Modern AI  

---

## 🔧 6.1~6.8 Refactoring Guide

### 6.1 RNN and Sequential Data
- 역할: 왜 RNN이 필요한가  
- 핵심: Sequential Data, Variable Length, Mapping, 응용  
- 제거: hidden state 수식, BPTT, transformer 설명  

### 6.2 RNN Hidden State and Dynamics
- 역할: 상태 기반 구조 이해  
- 핵심: Hidden State, Recurrence, Dynamical System, Lossy Summary  
- 제거: language modeling, transformer  

### 6.3 Vanilla RNN and BPTT
- 역할: 실제 계산과 학습  
- 핵심: forward, loss, BPTT, gradient 문제  
- 제거: sequence 개론  

### 6.4 RNN Language Modeling
- 역할: 생성 문제 이해  
- 핵심: chain rule, next-token, teacher forcing  
- 제거: RNN 일반 설명  

### 6.5 Sequential Models Overview
- 역할: 전체 모델 계보  
- 핵심: ARIMA → Transformer 흐름  
- 제거: 세부 수식  

### 6.6 BiRNN, LSTM, GRU, Seq2Seq
- 역할: RNN 확장 구조  
- 핵심: memory, bidirectionality, encoder-decoder  
- 제거: attention 상세  

### 6.7 Attention and Transformer
- 역할: 병목 해결  
- 핵심: attention, self-attention, parallelization  
- 제거: GPT/BERT 상세  

### 6.8 Post-Transformer Modern AI
- 역할: 최신 AI 연결  
- 핵심: foundation model, GPT, BERT, multimodal  
- 제거: attention 반복 설명  

---

## 🔗 Navigation

← Previous: CNNs  
→ Next: Transformers
