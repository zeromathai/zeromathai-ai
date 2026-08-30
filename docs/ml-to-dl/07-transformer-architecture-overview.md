# Transformer — From Self-Attention to Modern LLM Architectures

## 📌 Lecture Overview

The Transformer replaced sequential recurrent processing with a structure that can compare tokens across an entire sequence in parallel. Its central mechanism, **self-attention**, lets every token build a context-aware representation by selecting information from other tokens.

Understanding the Transformer, however, requires more than memorizing the attention equation. The full architecture connects several ideas: sequence-to-sequence modeling, token embeddings, Query–Key–Value attention, multi-head attention, positional information, residual blocks, autoregressive decoding, efficient attention, KV caching, and the architectural changes used in modern large language models.

This chapter develops those ideas as one continuous progression: from the limitations of RNN-based sequence models to the structure of modern Transformer blocks using **Pre-LN, RMSNorm, RoPE, GQA, SwiGLU, Mixture of Experts, and Multi-Token Prediction**.

---

## 📖 Full Article - English

[https://zeromathai.com/en/transformer-architecture-overview-en/](https://zeromathai.com/en/transformer-architecture-overview-en/)

* Korean
  [https://zeromathai.com/transformer-architecture-overview/](https://zeromathai.com/transformer-architecture-overview/)

👉 Other languages are available on the website

---

## 📚 Table of Contents

* [1. Why Transformers Replaced Sequential Processing](#1-why-transformers-replaced-sequential-processing)
* [2. From Sequence-to-Sequence Models to the Transformer](#2-from-sequence-to-sequence-models-to-the-transformer)
* [3. Language Modeling and the Need for Attention](#3-language-modeling-and-the-need-for-attention)
* [4. Query, Key, and Value](#4-query-key-and-value)
* [5. Self-Attention as Matrix Computation](#5-self-attention-as-matrix-computation)
* [6. Multi-Head Attention and Representation Subspaces](#6-multi-head-attention-and-representation-subspaces)
* [7. Positional Information: Sinusoidal Encoding, APE, RPE, and RoPE](#7-positional-information-sinusoidal-encoding-ape-rpe-and-rope)
* [8. Residual Connections, LayerNorm, and Transformer Blocks](#8-residual-connections-layernorm-and-transformer-blocks)
* [9. Transformer Decoder and Autoregressive Generation](#9-transformer-decoder-and-autoregressive-generation)
* [10. LM Head, Temperature, and Decoding Strategies](#10-lm-head-temperature-and-decoding-strategies)
* [11. The Long-Context Attention Bottleneck](#11-the-long-context-attention-bottleneck)
* [12. Local, Sparse, and FlashAttention](#12-local-sparse-and-flashattention)
* [13. KV Cache and Autoregressive Inference](#13-kv-cache-and-autoregressive-inference)
* [14. MQA, GQA, and MLA](#14-mqa-gqa-and-mla)
* [15. Modern Transformer Blocks in LLMs](#15-modern-transformer-blocks-in-llms)
* [16. RMSNorm, SwiGLU, Mixture of Experts, and Multi-Token Prediction](#16-rmsnorm-swiglu-mixture-of-experts-and-multi-token-prediction)

---

## 1. Why Transformers Replaced Sequential Processing

Before Transformers, **Recurrent Neural Networks (RNNs)** were a standard way to process sequences. An RNN reads one position at a time and carries a hidden state from one step to the next.

That design naturally represents order, but it creates two important limitations.

First, computation is sequential. Later states depend on earlier states, so all positions cannot be processed independently at the same time. This limits parallelization.

Second, information from distant positions must pass through many recurrent transitions. As sequences become longer, maintaining important long-range information becomes difficult.

Traditional **sequence-to-sequence (Seq2Seq)** models inherited another bottleneck: the encoder could compress an entire input sequence into a limited representation that the decoder then used to generate the output. As the input grew longer, representing all relevant information through such a bottleneck became increasingly difficult.

The Transformer changes this computational pattern. Instead of propagating information token by token, it places the sequence into a common representation space and directly computes relationships among tokens.

The key mechanism is **self-attention**.

A token can therefore reference information from nearby or distant positions without requiring that information to travel through a chain of recurrent states. Because the sequence is represented as matrices, many of these interactions can also be computed in parallel during training.

This combination of direct token interaction and parallel computation is the architectural foundation on which modern large language models are built.

---

## 2. From Sequence-to-Sequence Models to the Transformer

The original Transformer is an **encoder–decoder architecture** designed for sequence-to-sequence tasks.

Its basic data flow is:

```text
Input sequence
    ↓
Tokenization
    ↓
Token embeddings + positional information
    ↓
Transformer Encoder
    ↓
Contextual token representations
    ↓
Transformer Decoder
    ↓
Output sequence
```

The encoder transforms the input into contextual representations. The decoder generates the output while attending both to previously generated output tokens and, in an encoder–decoder model, to the encoder representations.

### Tokens, vocabulary, and embeddings

Text must first be converted into model-compatible units called **tokens**. A token may correspond to a word or a subword unit.

The complete set of possible tokens defines the model's **vocabulary**, whose size is the **vocabulary size**.

Each token is mapped to a dense numerical vector through a token or word embedding. These vectors form a **dense vector representation** in which semantic relationships can be represented geometrically.

Earlier embedding approaches such as **Word2Vec, GloVe, FastText, and ELMo** illustrate the broader idea of vector-space semantics and semantic similarity. In a Transformer, embeddings become the initial representations that successive Transformer blocks refine using context.

The dimensionality of these vectors is the **embedding dimension**.

### Transformer encoder

The encoder is a stack of repeated encoder layers. Each layer contains two major transformations:

* **Self-Attention**, which exchanges information among token positions.
* **Position-wise Feed-Forward Network (FFN)**, which applies the same learned nonlinear transformation independently to every position.

The FFN therefore uses shared parameters across positions: each token receives the same feed-forward transformation, even though its contextual representation may differ.

### Transformer decoder

The original decoder contains:

* masked multi-head self-attention,
* encoder–decoder cross-attention,
* a feed-forward network.

Masked self-attention prevents a position from reading future output tokens. Cross-attention lets the decoder selectively retrieve information from encoder representations.

This architecture naturally leads to two different forms of attention:

* **Self-Attention:** relationships within one sequence.
* **Cross-Attention:** relationships between two different sequences.

---

## 3. Language Modeling and the Need for Attention

A **language model** estimates which token is likely to come next given the preceding context.

Given:

```text
I love
```

the model does not simply choose one token immediately. It assigns scores or probabilities to candidate tokens in its vocabulary.

Generation proceeds autoregressively: after selecting one token, that token becomes part of the context used for the next prediction.

Conceptually,

$$
x^{(t)}=\hat{y}^{(t-1)}
$$

where the previous model output becomes an input at the next generation step.

For example:

```text
I love
I love you
I love you ...
```

The quality of the next prediction therefore depends heavily on how well the model can identify relevant information in its current context.

### Why attention was needed

Earlier encoder–decoder systems could compress the input into a fixed representation. That becomes restrictive for long sequences because every relevant detail must survive this compression.

**Attention** instead computes a different weighted combination of input representations for each output position.

If the encoder produces hidden states $h_1,\ldots,h_T$, the context representation at decoder step $t$ can be written as:

$$
c^{(t)}=\sum_{i=1}^{T}\alpha_i^{(t)}h_i
$$

where $\alpha_i^{(t)}$ indicates how strongly the decoder attends to encoder position $i$.

For example, if

$$
h_1=10,\quad h_2=20,\quad h_3=30
$$

and the attention weights are

$$
[0.6,0.3,0.1],
$$

then

$$
c=0.6\times10+0.3\times20+0.1\times30=15.
$$

The first encoder position contributes most strongly.

The important conceptual shift is that the context representation is no longer necessarily fixed for the entire output sequence. It can change according to what the decoder currently needs.

---

## 4. Query, Key, and Value

The general attention mechanism can be understood through **Query–Key–Value (QKV)**.

* **Query:** what information the current position is looking for.
* **Key:** the features used to determine whether a candidate position matches that query.
* **Value:** the information that will actually be retrieved if that position receives attention.

An information-retrieval analogy is useful:

```text
Query  → search request
Key    → index describing each candidate
Value  → content associated with the candidate
```

A stronger match between a query and key leads to a larger contribution from the corresponding value.

At a high level, attention follows three steps:

```text
Query–Key comparison
        ↓
Attention scores
        ↓
Softmax normalization
        ↓
Attention weights
        ↓
Weighted combination of Values
```

If the raw scores are approximately

$$
[2.0,1.0,0.1],
$$

Softmax produces weights of approximately

$$
[0.65,0.24,0.11].
$$

If the corresponding scalar values are

$$
[10,20,30],
$$

the weighted result is

$$
0.65\times10+0.24\times20+0.11\times30=14.6.
$$

Attention is therefore not an ordinary average. It is a learned, selective combination of information.

---

## 5. Self-Attention as Matrix Computation

In **self-attention**, the queries, keys, and values come from the same input sequence.

Let $X$ denote the matrix of input representations. Learned projection matrices create the three attention representations:

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V.
$$

These projections separate three roles that every token must play:

1. what the token wants to retrieve,
2. how the token should be matched,
3. what information the token can provide.

A model may, for example, project a 512-dimensional representation into a 64-dimensional attention representation. This is not merely dimensional compression; it creates specialized spaces for the attention computation.

### Attention scores

For a single query $q$ and key $k$, similarity is measured by the dot product:

$$
\text{score}=q\cdot k.
$$

If

$$
q_1\cdot k_1=112
$$

and

$$
q_1\cdot k_2=96,
$$

the first key receives the stronger raw match.

These numbers are scores, not probabilities.

### Scaling

As the key dimension increases, dot products can become large. Feeding very large values into Softmax can produce excessively concentrated distributions and unstable gradients.

Scaled dot-product attention therefore divides the score by the square root of the key dimension:

$$
\frac{q\cdot k}{\sqrt{d_k}}.
$$

Softmax then converts the scaled scores into normalized attention weights.

For example, scores

$$
[14,12]
$$

produce approximate weights

$$
[0.88,0.12].
$$

### Weighted values

The output for a query is the weighted combination of value vectors:

$$
z=\sum_i\alpha_i v_i.
$$

If the values are $[10,20]$ and the weights are $[0.88,0.12]$, then

$$
z=0.88\times10+0.12\times20=11.2.
$$

### Matrix formulation

All positions are processed together in practical Transformer implementations:

$$
\text{Attention}(Q,K,V)
=
\text{softmax}\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V.
$$

This equation captures the core Transformer operation:

1. $QK^T$ computes pairwise query–key relationships.
2. Division by $\sqrt{d_k}$ scales the scores.
3. Softmax converts them into attention weights.
4. Multiplication by $V$ combines the information carried by values.

The matrix formulation makes parallel processing possible during training and allows each token representation to incorporate information directly from other positions.

---

## 6. Multi-Head Attention and Representation Subspaces

A single attention map provides only one view of token relationships. Language, however, contains many kinds of dependencies: nearby lexical relationships, syntactic relationships, semantic associations, and long-distance references may all matter simultaneously.

**Multi-Head Attention (MHA)** addresses this by computing several attention operations in parallel using different learned projection matrices.

Each **attention head** has its own Q, K, and V projections and can therefore operate in a different **representation subspace**.

Conceptually:

```text
Input representations
      ├── Head 1 → attention pattern 1
      ├── Head 2 → attention pattern 2
      ├── ...
      └── Head h → attention pattern h
                    ↓
                Concatenate
                    ↓
             Output projection
```

One head may focus strongly on local relationships while another captures long-distance references. The architecture does not explicitly assign those roles; they are learned.

The original Transformer example used eight attention heads, illustrating how the model can distribute representation capacity across several subspaces.

After computing the heads independently, their outputs are concatenated and projected:

$$
\text{MultiHead}(Q,K,V)
=
\text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O.
$$

If two heads each produce three-dimensional vectors, concatenation produces a six-dimensional representation. The learned output matrix $W^O$ then mixes the information from the heads into the model's required output dimension.

Thus:

* concatenation collects information,
* the output projection recombines it.

Multi-head attention should therefore be understood not merely as "running attention several times," but as learning several parallel views of sequence relationships.

---

## 7. Positional Information: Sinusoidal Encoding, APE, RPE, and RoPE

Self-attention compares token representations without inherently knowing their sequence order.

Consider:

```text
dog bites man
man bites dog
```

The tokens are the same, but their order changes the meaning. Transformer representations therefore require positional information.

### Sinusoidal positional encoding

The original Transformer used fixed sinusoidal positional signals added to token embeddings:

$$
PE(pos,2i)
=
\sin\left(
\frac{pos}{10000^{2i/d_{\text{model}}}}
\right),
$$

$$
PE(pos,2i+1)
=
\cos\left(
\frac{pos}{10000^{2i/d_{\text{model}}}}
\right).
$$

Here:

* $pos$ is the token position,
* $i$ identifies a positional dimension,
* $d_{\text{model}}$ is the model dimension.

For $d_{\text{model}}=4$ and $pos=1$:

$$
PE(1)
=
[\sin(1),\cos(1),\sin(0.01),\cos(0.01)]
$$

and approximately

$$
PE(1)
\approx
[0.8415,0.5403,0.0100,0.99995].
$$

Different dimensions use different frequencies, giving each position a structured multi-scale pattern.

Learned positional embeddings provide another option: instead of generating the positional vector through a fixed function, the model learns it as a parameter.

As Transformer systems evolved, positional methods increasingly focused on how position should interact with attention itself.

### Absolute Positional Embedding

**Absolute Positional Embedding (APE)** associates a representation with each absolute position and combines it with the token representation:

$$
E=X+P.
$$

If

$$
X=[0.2,0.5]
$$

and

$$
P=[0.1,-0.2],
$$

then

$$
E=[0.2+0.1,0.5+(-0.2)]=[0.3,0.3].
$$

APE is simple and effective for positions covered during training.

Its limitation is long-context extrapolation. If the model must process positions beyond those encountered during training, fixed learned absolute positions may not transfer naturally.

### Relative Positional Embedding

**Relative Positional Embedding (RPE)** emphasizes the displacement between two positions rather than their independent absolute indices.

A simplified attention score can be written as:

$$
A_{ij}
=
\frac{Q_iK_j^T+R_{i-j}}{\sqrt{d}},
$$

where $R_{i-j}$ contributes information about the relative displacement between positions $i$ and $j$.

If

$$
Q_iK_j^T=12,\quad R_{i-j}=4,\quad \sqrt{d}=4,
$$

then

$$
A_{ij}=\frac{12+4}{4}=4.
$$

Without the positional contribution, the score would be:

$$
\frac{12}{4}=3.
$$

RPE therefore makes positional relationships explicit inside the attention calculation.

This can be useful for variable-length inputs, but it can increase implementation complexity because relative position terms or biases must be integrated directly into attention. Some forms can also be less convenient to combine with KV caching.

### Rotary Positional Embedding

**Rotary Positional Embedding (RoPE)** takes a different approach. Instead of adding a position vector, it rotates Query and Key representations according to their positions.

A two-dimensional rotation matrix is:

$$
R_{\theta}
=
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}.
$$

For example, rotating $[1,0]$ by $90^\circ$ gives:

$$
[1,0]\rightarrow[0,1].
$$

The important property of RoPE appears when the rotated Query and Key are compared:

$$
(R_{\theta}^{i}Q)^T(R_{\theta}^{j}K)
=
Q^TR_{\theta}^{j-i}K.
$$

Although position-dependent rotations are applied using absolute positions $i$ and $j$, the dot product depends on their relative difference $j-i$.

RoPE therefore combines convenient position injection with a naturally relative effect in attention.

### APE vs RPE vs RoPE

| Method | Where position enters                          | Main strength                                    | Main limitation                                     |
| ------ | ---------------------------------------------- | ------------------------------------------------ | --------------------------------------------------- |
| APE    | Added to token representation                  | Simple and effective                             | Can struggle outside trained positions              |
| RPE    | Added to or incorporated into attention scores | Explicit relative-distance modeling              | More attention and caching complexity               |
| RoPE   | Rotates Query and Key                          | Relative position emerges naturally in attention | Still part of a broader long-context design problem |

RoPE has become especially important in modern long-context Transformer designs because it integrates cleanly with the Q/K attention computation while avoiding a large separate positional table.

---

## 8. Residual Connections, LayerNorm, and Transformer Blocks

Attention alone does not make a complete Transformer layer.

A classic encoder block combines:

```text
Input
  ↓
Multi-Head Self-Attention
  ↓
Residual Connection + LayerNorm
  ↓
Feed-Forward Network
  ↓
Residual Connection + LayerNorm
```

The residual connection preserves the input while a sublayer learns a transformation.

In the original Post-LN form:

$$
\text{Output}
=
\text{LayerNorm}\left(
x+\text{Sublayer}(x)
\right).
$$

The addition is important because the network does not need to reconstruct the entire representation from scratch at every layer. Existing information can propagate through the residual path while each sublayer contributes a learned modification.

For a simplified scalar illustration, if

$$
x=10
$$

and

$$
\text{Sublayer}(x)=3,
$$

the residual addition produces $13$ before normalization. In an actual Transformer, normalization operates over the components of the token vector rather than over one scalar value.

**Layer Normalization** performs normalization within each token representation rather than mixing statistics across different sequence positions.

### Post-LN and Pre-LN

The original Transformer used **Post-Layer Normalization (Post-LN)**: the sublayer is evaluated first, the residual is added, and normalization follows.

Modern large Transformers commonly use **Pre-Layer Normalization (Pre-LN)**, where normalization is applied before the attention or FFN sublayer.

Conceptually:

```text
Post-LN:
x → Sublayer → Add x → Norm

Pre-LN:
x → Norm → Sublayer → Add x
```

This placement matters as networks become deeper. Pre-LN is widely used because it can provide more stable gradient flow through deep residual Transformer stacks.

---

## 9. Transformer Decoder and Autoregressive Generation

The Transformer decoder is responsible for producing output sequences.

In an encoder–decoder model, each decoder layer contains:

1. masked multi-head self-attention,
2. cross-attention over encoder representations,
3. a feed-forward network.

The first component is **causal**.

If the decoder is predicting the token at position $t$, it must not read tokens from positions greater than $t$. A **causal mask** prevents those future positions from contributing to attention.

For an input-conditioned encoder–decoder model, the output factorization is:

$$
P(y_1,y_2,\dots,y_T\mid x)
=
\prod_{t=1}^{T}
P(y_t\mid y_1,\dots,y_{t-1},x).
$$

Every output token is therefore predicted from the source input $x$ and previously available output tokens.

### Teacher forcing

During training, the complete target sequence is available. Instead of repeatedly feeding the model its own possibly incorrect predictions, **teacher forcing** supplies the correct preceding target tokens.

The target sequence is shifted one position to the right.

For example:

```text
Target:
I love you

Decoder input:
<START> I love

Prediction targets:
I       love       you
```

This is often described as **output embeddings shifted right**.

The causal mask is still required even though the full target sequence exists during training. It guarantees that each prediction can use only information that would actually be available at generation time.

At inference time, the correct future tokens are unavailable. The decoder therefore uses its own generated token as part of the next input context.

### Encoder–decoder vs decoder-only Transformers

Encoder–decoder Transformers are natural for tasks where one sequence is transformed into another, such as translation.

**Decoder-only Transformers** instead specialize the architecture around causal autoregressive language modeling:

```text
Existing context
      ↓
Causal self-attention
      ↓
Next-token prediction
      ↓
Selected token appended to context
      ↓
Repeat
```

This decoder-only organization is central to many modern large language models.

---

## 10. LM Head, Temperature, and Decoding Strategies

The final hidden representation is not yet a vocabulary probability distribution.

A **Language Modeling Head (LM Head)** projects each final hidden state into a vector whose dimensionality equals the vocabulary size. The resulting scores are called **logits**.

```text
Final hidden vector
       ↓
LM Head
       ↓
Vocabulary-sized logits
       ↓
Softmax
       ↓
Token probability distribution
```

During training, the objective encourages high probability for the correct next token, commonly using cross-entropy loss.

Some Transformer models use **weight tying**, sharing the input embedding matrix with the output vocabulary projection. This connects the input token representation and output token scoring spaces while reducing parameter usage.

### Temperature scaling

Before sampling, the shape of the probability distribution can be adjusted by dividing the logits by a temperature $\tau$:

$$
p_i(\tau)
=
\frac{\exp(z_i/\tau)}
{\sum_{j=1}^{n}\exp(z_j/\tau)}.
$$

Lower temperature makes the distribution sharper. Higher temperature makes it flatter.

For logits

$$
[2,1,0],
$$

at $\tau=1$ the probabilities are approximately:

$$
[0.67,0.24,0.09].
$$

At $\tau=0.5$:

$$
[0.87,0.12,0.02].
$$

At $\tau=2$:

$$
[0.51,0.31,0.19].
$$

Temperature changes the distribution from which decoding operates; it does not itself choose the output token.

### Greedy decoding

**Greedy decoding** chooses the highest-probability token at every step.

If the distribution is:

$$
[0.70,0.20,0.10],
$$

the first token is selected.

Greedy decoding is simple and fast, but a locally optimal choice at one position does not guarantee the best complete sequence.

### Beam search

**Beam search** keeps several candidate sequences alive simultaneously.

With beam size $k$, the decoder repeatedly expands candidates and retains the best $k$ partial sequences.

When

$$
k=1,
$$

beam search reduces to greedy decoding.

### Top-k sampling

**Top-k sampling** retains only the $k$ highest-probability tokens, renormalizes their probabilities, and samples within that candidate set.

When $k=1$, it again becomes equivalent to greedy decoding.

### Top-p sampling

**Top-p sampling**, or **nucleus sampling**, chooses the smallest candidate set whose cumulative probability reaches or exceeds a threshold $p$.

Suppose:

```text
honeycomb     0.45
gingerbread   0.20
donut         0.12
cupcake       0.04
```

With Top-k and $k=3$, the first three candidates are retained.

With Top-p and $p=0.6$, the first two candidates are enough because:

$$
0.45+0.20=0.65.
$$

The nucleus therefore contains honeycomb and gingerbread.

These strategies illustrate an important distinction: a Transformer computes a distribution, while **decoding determines how that distribution becomes an actual sequence**.

---

## 11. The Long-Context Attention Bottleneck

Self-attention's greatest strength also creates one of its major scaling problems.

With **Full Attention**, each of $n$ positions may attend to every other position, producing an attention matrix whose size grows quadratically:

$$
\text{Attention Cost}=O(n^2).
$$

For 1,000 tokens, the number of pairwise attention scores is approximately:

$$
1000\times1000=1{,}000{,}000.
$$

For 10,000 tokens:

$$
10000\times10000=100{,}000{,}000.
$$

Increasing sequence length by a factor of ten therefore increases the number of pairwise scores by roughly a factor of one hundred.

This is the **attention bottleneck**.

Longer context can expose the model to more information, but it also increases attention computation and memory requirements. Efficient long-context modeling therefore requires architectural or implementation strategies that reduce these costs.

---

## 12. Local, Sparse, and FlashAttention

Not all efficient-attention methods solve the same problem.

### Local Attention

**Local Attention** restricts each token to a nearby window.

For example, a token might attend only to the previous 128 tokens rather than to the entire sequence.

This substantially reduces the number of attention relationships and works well when nearby context contains most of the necessary information.

Its limitation is straightforward: important long-distance information may lie outside the local window.

### Sparse Attention

**Sparse Attention** is a more general family of restricted attention patterns.

Rather than computing every token-to-token edge, the model computes only selected connections. Patterns may include:

* local windows,
* block patterns,
* strided patterns,
* selected global positions.

Local attention is therefore one particular sparse pattern.

The trade-off is between efficiency and connectivity. If sparse patterns block a useful distant relationship, model quality can suffer. Some architectures consequently mix full-attention blocks and sparse-attention blocks.

### FlashAttention

**FlashAttention** optimizes a different dimension.

It does not primarily redefine which tokens may attend to which other tokens. Instead, it reorganizes how the same attention computation interacts with GPU memory.

GPUs contain:

* relatively small, fast on-chip SRAM,
* much larger but slower HBM.

A conventional implementation may repeatedly write large intermediate attention matrices to HBM and read them back. This data movement can become a major bottleneck, making attention **memory-bound**.

FlashAttention processes attention in blocks designed to keep useful intermediate values in faster memory when possible, reducing unnecessary transfers between SRAM and HBM.

Its central idea is therefore **IO-awareness**.

The distinction is important:

```text
Local / Sparse Attention
→ reduce the number of attention connections

FlashAttention
→ perform attention with more efficient memory movement
```

Efficient attention is not a single technique. It is the broader goal of preserving useful context while reducing computation, memory use, and memory-transfer overhead.

---

## 13. KV Cache and Autoregressive Inference

Training and autoregressive inference have different computational patterns.

During generation, a decoder produces one token at a time. After each token is appended, the next step attends over the preceding context again.

Without caching, the key and value representations of previous tokens would repeatedly be recomputed.

**KV caching** avoids this redundancy.

Suppose the current context is:

```text
Prompt → Dear
```

When the next token is generated, the key and value representations for the prompt and `Dear` already exist.

Instead of recomputing them, the model stores them:

```text
Previous tokens
      ↓
Cached K and V
      ↓
New token arrives
      ↓
Compute new Q, K, V
      ↓
Append new K and V to cache
      ↓
New Q attends to cached K/V
```

The query is associated with the new position being processed, while key and value states accumulate across previous positions.

### What KV caching does not eliminate

KV caching does **not** remove attention itself.

It eliminates repeated computation of previous K/V states, but the new query still has to attend to cached keys and combine cached values.

As context length increases, the cache itself also grows.

This creates a new inference bottleneck: **KV-cache memory**.

Reducing that memory requirement leads directly to MQA, GQA, and MLA.

---

## 14. MQA, GQA, and MLA

Standard **Multi-Head Attention (MHA)** gives every head its own Q, K, and V projections. This supports representation diversity but requires separate K/V states for every attention head.

Modern inference-oriented attention variants reduce this storage requirement.

### Multi-Query Attention

**Multi-Query Attention (MQA)** preserves multiple query heads but shares a single K/V pair across them.

For eight attention heads:

```text
MHA:
8 query heads
8 K/V pairs

MQA:
8 query heads
1 shared K/V pair
```

This can greatly reduce KV-cache memory.

The trade-off is reduced K/V diversity among heads, which can result in some loss of representation quality.

### Grouped-Query Attention

**Grouped-Query Attention (GQA)** provides an intermediate design.

Query heads are divided into groups, and each group shares a K/V pair.

For example, with eight query heads split into two groups:

```text
MHA → 8 K/V pairs
GQA → 2 K/V pairs
MQA → 1 K/V pair
```

GQA therefore trades somewhat more cache memory than MQA for greater K/V diversity.

It can be viewed as a practical middle ground between standard MHA and fully shared MQA.

### Multi-Head Latent Attention

**Multi-Head Latent Attention (MLA)** reduces cache size differently.

Rather than simply sharing full K/V states, it compresses key/value information into a smaller latent representation. The compressed latent state is cached and later projected into forms needed for attention.

The progression is therefore:

```text
MHA
→ independent K/V per head

MQA
→ one shared K/V

GQA
→ shared K/V within groups

MLA
→ compressed latent K/V representation
```

### Comparison

| Attention structure | K/V organization                             | Main benefit                             | Main trade-off                         |
| ------------------- | -------------------------------------------- | ---------------------------------------- | -------------------------------------- |
| MHA                 | Independent K/V per head                     | High representation diversity            | Large KV cache                         |
| MQA                 | One K/V shared by all query heads            | Very small cache                         | Reduced K/V diversity                  |
| GQA                 | K/V shared within head groups                | Balance of memory and representation     | More cache than MQA                    |
| MLA                 | K/V information compressed into latent state | Strong cache reduction for long contexts | Requires compressed latent projections |

These mechanisms share one objective: reducing the memory cost of autoregressive inference as context length grows.

---

## 15. Modern Transformer Blocks in LLMs

Modern LLMs still retain the basic Transformer logic:

* attention exchanges information across positions,
* feed-forward layers transform each position,
* residual connections preserve information across depth.

What has changed is how these pieces are assembled for large-scale training and long-context inference.

A representative modern block can be understood as:

```text
Input
  ↓
RMSNorm / Pre-Layer Normalization
  ↓
Self-Attention
  ├── Grouped-Query Attention
  └── Rotary Positional Embedding
  ↓
Residual Connection
  ↓
RMSNorm / Pre-Layer Normalization
  ↓
Feed-Forward Transformation
  ├── SwiGLU
  └── or Mixture of Experts
  ↓
Residual Connection
```

Several trends converge in this design.

**Pre-LN** improves stability in deep residual stacks.

**RMSNorm** simplifies normalization.

**RoPE** incorporates positional relationships into Q/K representations.

**GQA** reduces KV-cache memory relative to standard MHA.

**SwiGLU** provides gated feed-forward transformations.

**Mixture of Experts** increases parameter capacity without activating all parameters for every token.

The modern Transformer is therefore not a fundamentally different neural architecture. It is a refined assembly of the original attention-and-residual framework, optimized for deeper networks, longer contexts, and more efficient inference.

---

## 16. RMSNorm, SwiGLU, Mixture of Experts, and Multi-Token Prediction

### RMSNorm

**Root Mean Square Normalization (RMSNorm)** simplifies LayerNorm by removing the re-centering step.

For a hidden vector $\mathbf{h}$,

$$
\text{RMS}(\mathbf{h})
=
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}h_i^2
}.
$$

The normalized representation is:

$$
\mathbf{h}_{\text{norm}}
=
\frac{\mathbf{h}}
{\text{RMS}(\mathbf{h})+\epsilon}
\odot\mathbf{g},
$$

where $\mathbf{g}$ is a learned scaling parameter.

If

$$
\mathbf{h}=[3,4],
$$

then

$$
\text{RMS}(\mathbf{h})
=
\sqrt{\frac{9+16}{2}}
=
\sqrt{12.5}
\approx3.54.
$$

Ignoring $\epsilon$ and using

$$
\mathbf{g}=[1,1],
$$

the normalized vector is approximately

$$
[0.85,1.13].
$$

RMSNorm provides a computationally simpler way to stabilize activation scale and is widely used in large Transformer models.

### SwiGLU

Modern feed-forward blocks frequently use a gated activation such as **SwiGLU**.

A simplified form is:

$$
\text{SwiGLU}(x)
=
(W_1x)\odot\text{Swish}(W_2x).
$$

One pathway produces a value representation while the other acts as a gate.

If

$$
W_1x=4
$$

and

$$
\text{Swish}(W_2x)=0.5,
$$

then

$$
\text{SwiGLU}(x)=4\times0.5=2.
$$

The gating mechanism allows the FFN to modulate which information is emphasized or suppressed.

### Mixture of Experts

A **Mixture of Experts (MoE)** replaces or augments a dense feed-forward computation with multiple expert networks.

A **router**, or gating network, selects which experts should process each input token.

For example, suppose the routing probabilities for four experts are:

$$
[0.45,0.19,0.05,0.31].
$$

With Top-2 routing, experts 1 and 4 are selected.

The important idea is that total parameter capacity and per-token computation can be partially separated. A model may contain many expert parameters while activating only a subset for each token.

MoE is therefore a form of sparse computation.

Its advantages come with important trade-offs:

* load imbalance when too many tokens select the same experts,
* routing instability,
* communication overhead in distributed systems.

Increasing parameter capacity alone does not remove these systems-level constraints.

### Multi-Token Prediction

Standard autoregressive language modeling usually trains a position $t$ to predict only the immediate next token at $t+1$.

**Multi-Token Prediction (MTP)** extends that supervision so that one representation can be trained to predict several future positions, such as:

$$
t+1,\quad t+2,\quad t+3.
$$

Conceptually:

```text
Standard next-token training:
representation at t
      ↓
predict t+1

Multi-Token Prediction:
representation at t
      ├── predict t+1
      ├── predict t+2
      └── predict t+3
```

This provides more supervisory signals from the same input representation and can improve sample efficiency. In some architectures, it can also connect to faster inference strategies.

---

## 🎯 Key Insight

The Transformer is best understood as a sequence of connected design decisions rather than as a single attention formula.

Self-attention removes the need to propagate sequence information recurrently by computing token relationships directly. QKV projections turn information retrieval into differentiable matrix operations, while multi-head attention lets those relationships be represented in multiple learned subspaces. Positional mechanisms restore the order information that self-attention does not inherently contain, and residual connections with normalization make deep stacks trainable.

Autoregressive decoders then convert contextual representations into vocabulary logits and repeatedly choose new tokens. As context length grows, the architecture faces new computational constraints: quadratic full attention, GPU memory movement, repeated K/V computation, and expanding KV caches.

Local and sparse attention reduce connectivity, FlashAttention reduces memory-transfer overhead, KV caching avoids recomputing past K/V states, and MQA, GQA, and MLA reduce the storage burden of those states.

Modern LLM blocks continue to use the same Transformer foundation while combining it with Pre-LN or RMSNorm, RoPE, GQA, SwiGLU, and sometimes Mixture of Experts and Multi-Token Prediction. The Transformer is therefore less a fixed historical architecture than a reusable design framework for building scalable sequence models.

---

## 📚 Related Advanced Topics

* [Transformer Architecture and Core Components — Sequence-to-Sequence Models and Encoder–Decoder Architecture](https://zeromathai.com/en/transformer-architecture-core-components-en/)
* [Attention and Language Modeling Basics — Next-Token Prediction and Query–Key–Value Fundamentals](https://zeromathai.com/en/attention-language-modeling-basics-en/)
* [Self-Attention Mechanism — From Query–Key–Value to Matrix Computation](https://zeromathai.com/en/self-attention-qkv-matrix-en/)
* [Multi-Head Attention, Positional Encoding, and Add & Norm — Completing the Transformer Block](https://zeromathai.com/en/multi-head-attention-positional-encoding-add-norm-en/)
* [Transformer Decoder, LM Head, and Decoding Strategies — From Output Representations to Token Selection](https://zeromathai.com/en/transformer-decoder-lm-head-decoding-en/)
* [Efficient Attention Mechanisms for Transformer LLMs — Full Attention, Sparse Attention, and FlashAttention](https://zeromathai.com/en/efficient-attention-flashattention-sparse-en/)
* [KV Cache and Shared Key-Value Attention — MQA, GQA, and MLA](https://zeromathai.com/en/kv-cache-shared-key-value-attention-en/)
* [Advanced Positional Embeddings — APE, RPE, and RoPE in Transformer Models](https://zeromathai.com/en/advanced-positional-embeddings-en/)
* [Modern Transformer Blocks in Large Language Models — Core Changes in 2024-era Transformer Architecture](https://zeromathai.com/en/modern-transformer-blocks-llm-en/)

---

## ⭐ Note

This GitHub version is adapted from the full ZeroMathAI lecture for repository-friendly reading.

---

## 🔗 Navigation

This is the only lecture in the current set.
