# 06. Recurrent Neural Networks (RNNs)

## 📌 Lecture Overview

This lecture introduces **Recurrent Neural Networks (RNNs)** through the full story of **sequence modeling**.

It starts from one simple question:

👉 How can a neural network understand data where order matters?

RNNs were designed for data such as:

- text  
- speech  
- time series  
- music  
- video frames  

In these data types, meaning does not come from one input alone.  
It comes from how inputs are connected over time.

This lecture explains:

- why sequential data needs a different modeling approach  
- how RNNs use hidden states as memory  
- how RNNs learn through Backpropagation Through Time  
- how language modeling predicts the next token  
- why vanilla RNNs struggle with long-term dependencies  
- how LSTM, GRU, Seq2Seq, Attention, and Transformer extend the idea  
- how this flow leads to modern AI models such as BERT, GPT, ViT, and multimodal models  

Together, these ideas show how RNNs became one of the key bridges from classical neural networks to modern foundation models.

---

## 📖 Full Article

- English  
  https://zeromathai.com/en/rnn-overview/

- Korean  
  https://zeromathai.com/rnn-overview/

👉 Other languages are available on the website

---

## 📚 Table of Contents

- Why sequence modeling matters  
- RNN hidden state and memory  
- Vanilla RNN and learning through time  
- RNN language modeling  
- Sequential model evolution  
- BiRNN, LSTM, GRU, and Seq2Seq  
- Attention and Transformer  
- Transformer-based modern AI  

---

## 1. Why Sequence Modeling Matters

Most basic neural networks assume that inputs can be processed independently.

But many real-world data types are not independent.

Examples:

- in a sentence, word order changes meaning  
- in speech, sound patterns unfold over time  
- in stock prices, previous values affect interpretation  
- in video, motion appears across multiple frames  

This kind of data is called **sequential data**.

👉 Key idea:

The meaning of the current input depends on what came before it.

That is why sequence modeling is needed.

---

## 2. RNN Hidden State and Memory

RNNs solve sequence modeling by introducing a simple but powerful idea:

👉 keep a state that summarizes the past

This state is called the **hidden state**.

At each time step, an RNN:

- reads the current input  
- combines it with the previous hidden state  
- creates a new hidden state  

So the model does not treat each input as isolated.

It carries context forward through time.

### Intuition

The hidden state is like a running memory.

For a sentence, it stores the meaning built so far.  
For a time series, it stores the recent pattern.  
For speech, it stores acoustic context.

However, this memory is compressed.

So it is useful, but not perfect.

---

## 3. Vanilla RNN and Learning Through Time

The simplest RNN is often called:

- Vanilla RNN  
- Simple RNN  
- Elman RNN  

It repeatedly applies the same update rule across time.

This repetition is called **recurrence**.

To train an RNN, the model is unfolded across time and trained with:

👉 **Backpropagation Through Time (BPTT)**

BPTT allows the model to learn how earlier inputs affected later outputs.

But this also creates a major challenge.

When gradients travel through many time steps, they can become:

- too small → vanishing gradient  
- too large → exploding gradient  

This makes vanilla RNNs difficult to train on long sequences.

---

## 4. RNN Language Modeling

One of the most important applications of RNNs is **language modeling**.

The goal is simple:

👉 predict the next token from previous tokens

Example:

Input:

```text
The pen is
