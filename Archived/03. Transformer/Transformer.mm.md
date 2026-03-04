---
title: Transformer精读
markmap:
  colorFreezeLevel: 12
---

## Transformer

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
- **Encoder**: 
  - stack of $\mathcal{N} = 6$ identical layers
  - Each layer: 2 sub-layers
      1. multi-head self-attention
         - output: LayerNorm($x$ + Sublayer($x$)) '**Residual**'
      2. **position-wise fully-connected** feed-forward network
          - output: LayerNorm($x$ + Sublayer($x$)) 
      - Output dimension: $d_{\text{model}=512}$
         - facilitate these residual connections
         - all sublayer & embedding layer
- **Decoder**: 
  - stack of $\mathcal{N} = 6$ identical layers
  - Each layer: 3 sub-layers
      1. **Masker** multi-head self-attention
         - output: LayerNorm($x$ + Sublayer($x$))
         - why masked: ensures predictions for position $i$ can depend only on the known outputs at positions **before** $i$
         - how masked: output shifted rights & masked 
      2. multi-head self-attention
         - output: LayerNorm($x$ + Sublayer($x$))
      3. **position-wise fully-connected feed-forward network**
         - output: LayerNorm($x$ + Sublayer($x$)) 
      - Output dimension: $d_{\text{model}}=512$

### 3.2 Attention
#### 3.2 Attention
- mechanism：mapping  a query and a set of key-value pairs to an output(all out put are weighted summation)
#### 3.2.1 Scaled Dot-Product Attention
- input: 
  - **Query**: (batch_size,num_query $d_Q$,len_query $n_Q$)
  - **Key**: (batch_size,num_key $d_K=d_Q$,len_key $n_K$)
  - **Value**: (batch_size,num_value $d_V$,len_value $n_V$ )
- **Attention Score**: $\text{Atterntion} = \text{SoftMax}(\frac{Q \cdot K^{T}}{\sqrt{d_Q}})$
- why **Dot-Product Attention**: **faster and space-efficient**
  - optimized matrix multiplication
- why **Scaled**: dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients

#### 3.2.2 Multi-Head Attention
- **Formula**: $$\begin{aligned}
  \text{MultiHead}(Q,V,K) &=\text{Concat}(\text{head}_{1},...,\text{head}_{h})W^{o} \\
  \text{where} \quad  \text{head}_{i} & = \text{Attention}(QW^{Q}_{i},kW^{k}_{i},VW^{V}_{i})
\end{aligned}$$
  - **Paramter**: $$\begin{aligned} W^{Q}_{i} &\in \mathbb{R}^{d_{\text{model}} \times d_Q} , W^{K}_{i} \in \mathbb{R}^{d_{\text{model}} \times d_K} \\ W^{V}_{i} &\in \mathbb{R}^{d_{\text{model}} \times d_V} , W^{O} \in \mathbb{R}^{hd_V \times d_{\text{model}}}\end{aligned}$$
  - Real Case：$$\begin{aligned} d_{\text{model}}&=512, \quad h = 8 \\ d_Q=d_K&=d_V =d_{\text{model}}/h = 64 \end{aligned}$$
- merits: allows Transformer to **jointly** attend to information from **different representation
subspaces** at different positions, i.e. "**Parallelization & Multi-Representation**"
- Computation cost: multihead + reduced dimension of each head $\approx$ single-head + full-dimension

#### 3.2.3 Applications of Attention in our Model
- **encoder self-attention layers**： QKV from previous encoder layer. 
**Each** position in the encoder can attend to **all positions** in the previous layer of encoder.
- **decoder self-attention layers(Masked)**： QKV from previous decoder layer. 
Each position in the decoder can attend to all positions in the decoder **up to and including that position**(preserve *auto-regressive*).
- **encoder-decoder attention layers**：Query from previous decoder layer,Key-Value pair from output of encoder. 
**Every position in decoder** to attend over **all positions** in output of encoder.

### 3.3 Position-wise Feed-Forward Networks
- Architecture: each of the layers contains a fully connected feed-forward network(FFN) 
- Application: each position(token) **separately and identically**
- Formula: $\text{FFN}(x) = ReLU(xW_1+b_1)W_2+b_2$
  - $W_1(512,2048)$: $d_{\text{model}} = 512 \rightarrow d_{\text{FFN}}=2048$
  - $W_2(2048,512)$: $d_{\text{FFN}} = 2048 \rightarrow d_{\text{model}}=512$
- 为什么需要“升维”
  1. **增强非线性表达能力**
     - 单个线性层（512→512）只能做**仿射变换**，无法引入非线性。
     - 虽然注意力机制本身是非线性的（softmax），但**每个位置的表示仍需局部非线性变换**来增强表达力。
     - **ReLU + 高维中间层** 提供了强大的**分段线性函数逼近能力**。 
  2. **避免信息瓶颈**（Information Bottleneck）
     - 如果使用 512 → 512 → 512：
       - 第一层输出仍是 512 维，可能**无法充分展开特征空间**
       - 相当于在相同维度下做非线性变换，**表达能力受限**
     - 而 512 → 2048 → 512：类似于“先展开细节，再提炼精华”。
       - **先将信息“展开”到更高维空间**（2048D）
       - 在高维空间中用 ReLU 进行**稀疏激活和特征组合**
       - 再投影回原空间
  3. **模拟“专家混合”**（Mixture of Experts） 思想
     - 高维中间层（2048D）可以看作 **大量潜在特征检测器**（如 2048 个“专家”）
     - ReLU 激活使得**只有部分专家被激活**（稀疏性）
     - 这增强了模型对**不同输入模式的适应能力**
  4. **实验验证：更大的中间层效果更好**
     -  **Table 3** 中，作者做了消融实验：**2048 是最佳平衡点**：更大4096反而略降，可能是过拟合或优化困难,说明 **适度扩大中间层能提升性能** 
- Transformer 中 FFN 的升维逻辑与 CNN 中使用 1×1 卷积升维的逻辑在本质上高度相似，
都体现了深度学习中一种通用的设计范式：
“**先扩展表示空间以增强非线性能力，再压缩回原空间以保持接口一致**”。

### 3.4 Embeddings and Softmax
- learned embedding: convert input tokens and output tokens to vectors of dimension $d_{\text{model}}$
- linear transformation + softmax: predicted next-token probabilities
- **same weight matrix**: two embedding layers and the pre-softmax linear transformation
  - 为什么 Embedding 和 Pre-softmax Linear 可以共享权重？
    - 在标准语言模型/翻译模型中：
      - **Embedding 层**：将词索引 $w \in \{1, ..., V\}$ 映射为向量 $\mathbf{e}_w \in \mathbb{R}^{d_{\text{model}}}$
      - **Pre-softmax Linear 层**：将解码器最终输出 $\mathbf{z} \in \mathbb{R}^{d_{\text{model}}}$ 映射为 logits $\mathbf{l} \in \mathbb{R}^V$，用于 softmax 预测
    - 具体做法
      - 设词表大小为 $V$ (vocabulary size)，模型维度为 $d = d_{\text{model}}$
      - Embedding 矩阵：$\mathbf{W}_e \in \mathbb{R}^{W \times d}$,
        - 输入token $w$
        - 输出 $\mathbf{x} = \mathbf{W}_e[w, :]$
        - 输出shape $(d,)$ 
      - Pre-softmax 线性层：通常为 $\mathbf{W}_o \in \mathbb{R}^{d \times V}$
        - 输入：解码器输出 $\mathbf{z} \in \mathbb{R}^d$ 
        - 输出 $\text{logits} \quad \mathbf{l} = \mathbf{W}_e^\top \mathbf{z}$
        - 输出shape $(V,)$
      - **共享策略**：令 $\mathbf{W}_o = \mathbf{W}_e^\top$
  - 为什么可以共享？动机是什么？
    - Conclusion:**这是“输入表示”与“输出分类”使用同一语义空间的自然体现**
    - **语义一致性**
      - 如果词 $w$ 的 embedding 是 $\mathbf{e}_w$，
      - 那么当解码器输出 $\mathbf{z} \approx \mathbf{e}_w$ 时，应高概率预测 $w$
      - 共享权重天然实现：$\text{logit}(w) = \mathbf{e}_w^\top \mathbf{z}$
    - **减少参数量**
    - **提升泛化**
      - - Press & Wolf (2017) 在 *Using the Output Embedding to Improve Language Models* 中证明：权重共享可提升语言模型性能，尤其在低资源场景
- Scale: multiply those weights by $\sqrt{d_{\text{model}}}$ in the **embedding layers**
  - 为什么 Embedding 要乘以 $\sqrt{d_{\text{model}}}$？
    - 问题根源：**Embedding 向量的尺度太小**
      - 标准初始化（如 Xavier）下，embedding 向量的**元素均值为 0，方差为 $1/d$**
      - 所以向量的 **L2 范数期望** ≈ $\sqrt{d \cdot (1/d)} = 1$
      - 但 **位置编码**（Positional Encoding） 的每个维度是 **有界常数**（正弦/余弦值 ∈ [-1,1]）
        - 所以位置编码的范数 ≈ $\sqrt{d}$
      - 如果直接相加：  
        $$
        \text{input} = \text{embedding} + \text{pos\_enc}
        $$
        - embedding 范数 ≈ 1
        - pos_enc 范数 ≈ $\sqrt{d}$
        - **位置信息会 dominate 内容信息**！
    - 解决方案：放大 embedding
      - 将 embedding 乘以 $\sqrt{d}$：
        - 新 embedding 范数 ≈ $\sqrt{d}$
        - 与位置编码尺度匹配
        - 两者贡献均衡
  - 若不放大：模型可能“**只看位置，不看词义**”

### 3.5 Positional Encoding
- **sinusoid**
  - $$\begin{aligned}P_{i,2j} & = \sin \left(\frac{i}{10000^{\frac{2j}{d_{\text{model}}}}} \right) \\ P_{i,2j+1} & =\cos \left(\frac{i}{10000^{\frac{2j}{d_{\text{model}}}}} \right) \end{aligned}$$
  - inject relative or absolute position of the tokens in the sequence
  - 基于Fourier transformation设计的绝对位置编码
    - $$\mathbf{y}_i = f(\mathbf{x}_i, (\mathbf{x}_1, \mathbf{x}_1), \ldots, (\mathbf{x}_n, \mathbf{x}_n)) \sim  A_{0}+ \sum_{i=0}^{\infty}\left(A_{i} \sin(2\pi \frac{i}{P} \cdot x_{i})+ B_{i} \cos(2\pi \frac{i}{P} \cdot x_{i})\right)$$
      - 其中$P$是$f(\mathbf{x}_i, (\mathbf{x}_1, \mathbf{x}_1), \ldots, (\mathbf{x}_n, \mathbf{x}_n))$的周期
    - 这么设计的原因
      - 在频域上（也就是每一列），P的维度互相之间正交（三角函数积分的性质）
      - $\frac{i}{10000^{2j/d}}$这种非线性设计，需要比等分的更小的单位频率，即能在有限的维度拆分出更多的固有频率特征（即位置相关的信息）
    - 这两个特点在时域（也就是每一行）带来的性质
      - 让每一行$f(\mathbf{x})$的周期更大，可以同时容纳更多的token，且token在频率化位置编码后依然保持正交性；
      - 让不同token之间的相对位置关系$f(x) \rightarrow f(x+\Delta)$变成了可以由高维旋转矩阵表示的旋转位置关系
        $$\begin{bmatrix}
        \sin(\omega_{i} \cdot (x+\Delta)) \\
        \cos(\omega_{i} \cdot (x+\Delta))
        \end{bmatrix}
        =\begin{bmatrix}
        \cos(\omega_{i} \cdot \Delta) & \sin(\omega_{i} \cdot \Delta) \\
        -\sin(\omega_{i} \cdot \Delta) & \cos(\omega_{i} \cdot \Delta)
        \end{bmatrix} \cdot
        \begin{bmatrix}
        \sin(\omega_{i} \cdot x)\\
        \cos(\omega_{i} \cdot x)
        \end{bmatrix}
        $$

## 4 Why Self-Attention
- 3 factors
  - **complexity** per layer
  - **Parallization**
  - **Path length** between long-range dependencies
- Result
  - 
  | Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
  |------------|----------------------|------------------------|----------------------|
  | Self-Attention | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
  | Recurrent | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
  | Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k(n))$ |
  | Self-Attention (restricted) | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |
  - 在Self-Attention (restricted) 中，手动选择长度为$r$的window来限制输入以便于**提升计算性能**。
  - $r=n$时， 退化到基本的Self-Attention layer
- **Side benefit**, self-attention could yield more **interpretable** models. 
  - Individual attention heads learn to **perform different tasks**
  - **Many heads** exhibit behavior related to *syntactic and semantic structure of the sentences*.

## 5 Training

### 5.1 Training Data and Batching
- WMT 2014 English-German dataset
  - 4.5M sentence-pair
  - byte-pair encoding : 37K tokens
- WMT 2014 English-French dataset
  - 36M sentences -> 32k word-piece vocabulary 
  - 25k source tokens & 25k target tokens per batch
### 5.2 Hardware and Schedule
- 8 NVIDIA P100 GPUs
- base model
  - 100,000 steps or 12 hours
- big model
  - 300,000 steps(3.5 days)
### 5.3 Optimizer
- Adam optimizer
  - Params: $\beta_1 = 0.9,\beta_2 = 0.98, \epsilon = 10^{-9}$
  - formula : $\text{lrate} = d^{-0.5}_{\text{model}} \cdot min\left(\text{step\_num}^{-0.5}, \text{step\_num} \cdot \text{warmup\_num}^{-1.5} \right)$
  - warmup_num  = 4000

### 5.4 Regularization
- **Residual Dropout**：
  - output of each sub-layer before input and normalized
  - sums of the embeddings and positional encodings in **both encoder and decoder stacks**
  - $P_{\text{drop}}=0.1$
- **Label Smoothing**： $\epsilon_{ls}=0.1$,improves accuracy and BLEU score.

## 6 Results

### 6.1 Machine Translation
- WMT 2014 English-to-German
  - Transformer (big):
    - BLEU:28.4
  - Transformer (base):
    - BLEU:27.3
- WMT 2014 English-to-French
   - Transformer (big):
    - BLEU:41
  - Transformer (base):
    - BLEU:27.3 
- beam search : beam size 4 
- length penalty $α = 0.6$
### 6.2 Model Variations
- **Heads**:single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads.
- **Attention key size**:  determining compatibility is not easy and that a more sophisticated compatibility
function than dot product may be beneficial
- **bigger models are better**
- **Dropout**:  very helpful in avoiding over-fitting.
- **Learned positional embeddings**: nearly identical to sinusoidal positional encoding

### 6.3 English Constituency Parsing
- Good result in genral
## 7 Conclusion
- [source code](https://github.com/tensorflow/tensor2tensor)
- **Transformer**
  - based entirely on attention: **multi-headed self-attention**
  - great generalization ablity