---
title: Transformer精读
markmap:
  colorFreezeLevel: 12
---

## ResNet

- [Local Thesis](../Transformer/1706.03762v7_Attention%20Is%20All%20You%20Need.pdf)
- Basic info 
  - Author:
    - Ashish Vaswani
    - Noam Shazeer
    - Niki Parmar
    - Jakob Uszkoreit
    - Llion Jones
    - Aidan N. Gomez
    - Łukasz Kaiser
    - Illia Polosukhin
  - Publish:
    - Time: 2017
    - Journal: 31st Conference on Neural Information Processing Systems (NIPS 2017)

## Abstract
- dominent model: RNN with encoder-decoer with attention mehcanism
- **Transfomer** : 'Solely on ATTENTION Mechanisms'
- *Achievement*;
  - 28.4 BLEU on the WMT 2014 English-to-German translation task(over 2 BLEU)
  - 41.8 BLEU on the WMT 2014 English-to-French translation task
- Transformer **generalizes** well to other tasks

## 1 Introduction
- RNN: factor computation along the *symbol positions* of the input and output sequences
  - a sequence of hidden states $h_t$ : 
  combine previous hidden state $h_{t-1}$ 
  and the input for position $t$
  - precludes parallelization :  memory constraints limit batching across examples
    - gradient vanish/explode
- Attention mechanism: regardless to distance in the input/output sequences
- **Transfomer** :draw global dependencies between input/outputs
  - more parallelization

## 2 Background
- CNN: difficult to learn dependencies between distant positions
  - reason: number of operations to relate signals from two arbitrary input or output positions grows in the distance between positions
  - **Transfomer Resort** : a *constant* number of operations
    - Multi-Head Attention
- Self-attention: relating different positions of a single sequence to compute a representation of the sequence
- End-to-end memory networks: a **recurrent** attention mechanism
- **Transfomer** :'Solely ATTENTION' without using sequence-aligned RNNs or convolution.
  
## 3 Model Architecture
### 3 Model Architecture
- encoder-decoder: '*auto-regressive*'
  - encoder
    - maps input (sequence of symbol representations) $(x_1, ..., x_n)$ 
    to continuous representations $\mathbf{z} = (z_1, ..., z_n)$ 
  - decoder
    - generates an output sequence $(y_1, ..., y_m)$ of symbols one element at a time.
    - consuming *previously generated symbols* as **additional input** when generating the next.
- **Transfomer** : stacked **self-attention** and point-wise, **fully-connected** layers for both encoder & decoder
- fig:[Transformer Architecture](Transformer%20Architecture.png)
  
### 3.1 Encoder and Decoder Stacks
- Encoder: 
  - stack of $\mathcal{N} = 6$ identical layers
  - Each layer: 2 sub-layers
      1. multi-head self-attention
         - output: LayerNorm($x$ + Sublayer($x$)) '**Residual**'
      2. **position-wise fully-connected** feed-forward network
          - output: LayerNorm($x$ + Sublayer($x$)) 
      - Output dimension: $d_{\text{model}=512}$
         - facilitate these residual connections
         - all sublayer & embedding layer
- Decoder: 
  - stack of $\mathcal{N} = 6$ identical layers
  - Each layer: 3 sub-layers
      1. **Masker** multi-head self-attention
         - output: LayerNorm($x$ + Sublayer($x$))
         - why masked: ensures predictions for position $i$ can depend only on the known outputs at positions **before** i.
         - how masked: output shifted rights & masked 
      2. multi-head self-attention
         - output: LayerNorm($x$ + Sublayer($x$))
      3. **position-wise fully-connected feed-forward network**
         - output: LayerNorm($x$ + Sublayer($x$)) 
      - Output dimension: $d_{\text{model}}=512$

### 3.2 Attention

#### 3.2.1 Scaled Dot-Product Attention

#### 3.2.2 Multi-Head Attention

#### 3.2.3 Applications of Attention in our Model

### 3.3 Position-wise Feed-Forward Networks

### 3.4 Embeddings and Softmax

### 3.5 Positional Encoding

## 4 Why Self-Attention

## 5 Training

### 5.1 Training Data and Batching

### 5.2 Hardware and Schedule

### 5.3 Optimizer

### 5.4 Regularization

## 6 Results

### 6.1 Machine Translation

### 6.2 Model Variations

### 6.3 English Constituency Parsing

## 7 Conclusion
