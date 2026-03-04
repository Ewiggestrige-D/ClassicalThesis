---
title: ResNet精读
markmap:
  colorFreezeLevel: 12
---

## ResNet

- [Local Thesis](../Resnet/1512.03385v1_Deep%20Residual%20Learning%20for%20Image%20Recognition.pdf)
- Basic info 
  - Author:
    - Kaiming He*（Main）
    - Xiangyu Zhang 
    - Shaoqing Ren
    - Jian Sun
  - Publish:
    - Time: 2015
    - Journal: Conference on Computer Vision and Pattern Recognition（CVPR）

## Abstract
- **Main issue**: Deeper neural networks are more difficult to train
  - gradients vanish/explode when deeper
- **Achievement** : *Residual networks* are easier to optimize and  gain accuracy from increased depth
- Result ：
  - ImageNet：3.57% error
  - CIFAR-10: 100/1000 Layers net
  - COCO detection: 28% relative improvement

## 1. Introduction
- Background:
  - network depth is of crucial importance
- Problem:
  1. gradients vanish/explode
      - **Solution**: normalized initialization 
    & intermediate normalization layers
  2. *degradation* problem: 
  accuracy gets **saturated and degrades**
      - **Reason**: " *not caused by overfitting* "
      - **Evidence**: Fig.1 -The deeper network
has higher training error, and thus test error
      - **Solution**: added layers are **identity mapping**,
and the other layers are copied from the learned shallower model.
        - **deduction**： a deeper model should produce **no higher training error** than its shallower counterpart.
        - **Architecture**：Fig.2
- "a deep **residual learning** framework"
  - **Main Idea**: explicitly let layers fit **a residual mapping**
  - **Formulation**:  desired underlying mapping $\mathcal{H}(x) :=\mathcal{F}(x)+x$ ;
  nonlinear layers $\mathcal{F}(x)$
  - **Merits**:
    - hypothetically **easier** to optimize the residual mapping than  
    the original unreferenced mapping
    - at least, easier to push the residual to zero than 
    to fit an identity mapping by a stack of nonlinear layers.
    - **Identity shortcut connections** add neither extra parameter nor computational complexity.
  - Comparisons:
    - deep residual nets are *easy to optimize*, 
    but the counterpart “plain” nets exhibit higher training error when the depth increases
    - deep residual nets can *easily enjoy accuracy gains* from increased depth, 
    producing results better than previous.

## 2. Related Work

### Residual Representations
- Historic methods：
  1. VLAD（Vector of Locally Aggregated Descriptors）
      - 核心思想
        1. **构建视觉词典**（codebook）：
            - 对大量局部特征（如 SIFT）进行 K-Means 聚类，得到 $K$ 个聚类 中心 $\{c_1, c_2, ..., c_K\}$
        2. **对每个局部特征 $x_i$**：
            - 找到其最近的聚类中心 $c_{k(i)}$
            - 计算**残差**（residual）：$x_i - c_{k(i)}$
        3. **对每个聚类中心 $k$，累加所有分配给它的残差**：$$ v_k = \sum_ {i: k(i)=k} (x_i - c_k) $$
        4. **拼接所有 $v_k$ 得到 VLAD 向量**：$$\text{VLAD} =   [v_1^\top, v_2^\top, ..., v_K^\top]^\top \in \mathbb{R}^{K  \cdot d}$$
        其中 $d$ 是局部特征维度（如 SIFT 的 128 维）
      - 关键特点
        1. **残差编码** : 编码的是“局部特征与聚类中心的偏差”，比 BoW 的 0/1 计数包含更多信息 
        2. **硬分配** ： 每个特征只分配给一个最近的聚类中心（类似 BoW）
        3. **无概率模型** ： 纯几何方法，不涉及生成模型 
  2. Fisher Vector（FV）
      - 核心思想
        1. **用概率生成模型建模局部特征分布**：
            - 通常使用 **高斯混合模型**（GMM）
            - GMM 有 $K$ 个高斯分量，参数为 $\Theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K$
              - $\pi_k$：第 $k$ 个分量的权重
              - $\mu_k$：均值（相当于 VLAD 的聚类中心）
              - $\Sigma_k$：协方差矩阵
        2. **对每个局部特征 $x_i$，计算其对每个高斯分量的后验概率**（软分配）：$$\gamma_k(x_i) = P(k | x_i) = \frac{\pi_k \mathcal{N}(x_i; \mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_i; \mu_j, \Sigma_j)}$$
        3. **计算 Fisher Score**（对 GMM 参数的梯度）：
            - 对均值 $\mu_k$ 的梯度（最常用）：
                $$
                G_{\mu_k} = \frac{1}{N \sqrt{\pi_k}} \sum_{i=1}^N \gamma_k(x_i)\cdot \Sigma_k^{-1/2} (x_i - \mu_k)
                $$
            - （也可对 $\pi_k$ 或 $\Sigma_k$ 求导，但通常只用 $\mu_k$ 项）
        4. **拼接所有 $G_{\mu_k}$ 得到 Fisher Vector**：$$\text{FV} = [G_{\mu_1}^\top, G_{\mu_2}^\top, ..., G_{\mu_K}^\top]  ^\top$$
      - 与 VLAD 的关系
          **FV 是 VLAD 的概率泛化**：  
          - 当 GMM 协方差 $\Sigma_k = \sigma^2 I$ 且 $\pi_k = 1/K$，  
          - 且后验 $\gamma_k(x_i)$ 趋近于 one-hot（即硬分配），  
          - 则 FV ≈ VLAD（相差一个常数缩放）
  3. low-level vision and computer graphics
      -  solvers converge much faster than standard solvers 
      that are *unaware of the residual nature of the solutions*.

### Shortcut Connections.
- Practices and theories before:
  - MLPs: add a linear layer connected from the network input to the  output
  - intermediate layers **directly connected** to auxiliary classifiers
  for addressing vanishing/exploding gradients.
  - shortcut connections for centering layer responses,
  gradients, and propagated errors
  - an “inception” layer is composed of **a shortcut branch** 
  and a few deeper branches.


## 3. Deep Residual Learning

### 3.1. Residual Learning
- **Motivation**: a deeper model should have training error **no greater** than its shallower counterpart.
- **Reformulation**:  rather than expect stacked layers to approximate $\mathcal{H}(x)$, we
explicitly let these layers approximate a residual function $\mathcal{F}(x): =\mathcal{H}(x) -x $
- **feature**: the learned residual functions in general have **small responses**(Fig.7)

### 3.2. Identity Mapping by Shortcuts
- building blocks: $$y =\mathcal{F}(x, \{ W_{i} \}) +x $$
  - Fig.2 : $\mathcal{F} = W_{2}\sigma (W_{1}x)$
  - constrain: dimensions of $x$ and $\mathcal{F}(x)$ must be equal
    - perform a linear projection $W_{s}$ to match:$$y =\mathcal{F}(x, \{ W_{i} \}) +W_{s}x$$
  - about $\mathcal{F}(x)$
    - $\mathcal{F}(x)$ can have two or three layers or more layers 
    - single layer $\mathcal{F}(x)$ has no observed advantages

### 3.3. Network Architectures

#### Plain Network
- 34-layer plain net (Fig. 3-middle)
  - Architecture:
    - 7 $\times$ 7 conv layer, 64 channels, stride 2 
    - 3 $\times$ 3 max-pooling, 64 channels, stride 2 (not-counted)
    - 3 $\times$ 3 conv layers, 64 channels, stride 1 ,6-layers
    - 3 $\times$ 3 conv layers, 128 channels, stride 1 ,8-layers 
    (to double channels, first layer of 8 has stride 2 )
    - 3 $\times$ 3 conv layers, 256 channels, stride 1 ,12-layers 
    (to double channels, first layer of 8 has stride 2 )
    - 3 $\times$ 3 conv layers, 512 channels, stride 1 ,6-layers 
    (to double channels, first layer of 8 has stride 2 )
    - average-pooling layer
    - 1000-d fully-connected layer
    - sofmax layer
  - Parameters:
    - 3.6B FLOPs (**18%** of VGG-19 - 19.6 billion FLOPs).
  - Rules:
    - **same output feature map size, same number of filters**
    - **half feature map size, double channels**

#### Residual Networks
- 34-layer residual net (Fig. 3-right)
  - Architecture:
    - 7 $\times$ 7 conv layer, 64 channels, stride 2 
    - 3 $\times$ 3 max-pooling, 64 channels, stride 2 (not-counted)
    - 3 $\times$ 3 conv layers, 64 channels, stride 1 ,6-layers
    (*identity shortcut connections* every 2-layers)
    - 3 $\times$ 3 conv layers, 128 channels, stride 1 ,8-layers 
    (to double channels, first layer of 8 has stride 2 
    and *1 $\times$ 1 convolutions shortcut connections* 
    and *identity shortcut connections* every 2-layers)
    - 3 $\times$ 3 conv layers, 256 channels, stride 1 ,12-layers 
    (to double channels, first layer of 8 has stride 2 
    and *1 $\times$ 1 convolutions shortcut connections*
    and *identity shortcut connections* every 2-layers)
    - 3 $\times$ 3 conv layers, 512 channels, stride 1 ,6-layers 
    (to double channels, first layer of 8 has stride 2 
    and *1 $\times$ 1 convolutions shortcut connections*
    and *identity shortcut connections* every 2-layers)
    - average-pooling layer
    - 1000-d fully-connected layer
    - sofmax layer
  - Parameters:
    - 3.6B FLOPs
  - Rules:
    - **same output feature map size, identity mapping**
    - **half feature map size, 1 $\times$ 1 convolution layer**

### 3.4. Implementation
- resize image with its shorter side randomly sampled in [256; 480]
- randomly 224 $\times$ 224 crop  or its horizontal flip
- standard color augmentation
- Normalization: **batch normalization (BN) after each convolution 
and before activation**
- SGD
- Batch Size: 256
- Learning Rate:  0.1 and *divided by 10 when the error plateaus*
- Epochs: 600K 
- Weight Decay: 0.0001
- Momentum: 0.9

## 4. Experiments

### 4.1. ImageNet Classification
- ImageNet 2012 dataset
  - 1000 classes
  - 1.28M training images
  - 50k validation images
  - 100k test images
- Plain Networks
  - 18-layer and 34-layer (Fig.4a)
    - training/validation errors
      - 18-layer Top1 error rate: 27.94%
      - 34-layer Top1 error rate: 28.54%（**degradation problem**）
    - BN  ensures neither forward nor backward signals vanish.
      - Conjecture :
      deep plain nets have exponentially low convergence rates
- Residual Networks
  - 18-layer and 34-layer (Fig.4b)
    - training/validation errors
      - 18-layer Top1 error rate: 27.88%
      - 34-layer Top1 error rate: 25.03%（**Better**）
  - **Observation**
    - ResNets address degradation problem 
    and obtain accuracy gains from increased depth.
    - ResNets performace better than plain nets
    - ResNet eases the optimization 
    by providing faster convergence at the early stage.
- Identity vs. Projection Shortcuts
  - modes
    1. Mode **A**: zero-padding shortcuts for increasing dimensions
    2. Mode **B**: projection shortcuts are used for increasing dimensions, 
    and other shortcuts are identity
    3. Mode **C**: all shortcuts are projections.
  - Results:
    - Mode **B**  slightly better than Mode **A**
      - Attribution: zero-padded dimensions have no residual learning. 
    - Mode **C**  marginally better than Mode **B**, 
      - Attribution:  extra parameters introduced by projection shortcuts.
- Deeper **Bottleneck** Architectures
  - Modification:
    - use a stack of 3 layers instead of 2 (Fig. 5). 
    - 2-layers: 3 $\times$ 3 conv layer + 3 $\times$ 3 conv layer (64 channels)
    - 3-layers: 1 $\times$ 1-64 channels, 3 $\times$ 3-64 channels, and 1 $\times$ 1-256 channels  conv layers
    - 1 $\times$ 1 conv layers for reducing and then increasing (restoring) dimensions
  - Identity shortcuts in thjs case hold **lower time complexity and model size**
- **Deeper ResNet**: 50/101/152-layer ResNets are more accurate than
the 34-layer ones by considerable margins (Table 3 and 4).

### 4.2. CIFAR-10 and Analysis
- CIFAR-10 dataset
  - 10 classes
  - 50k training images
  - 10k test images
  - 32 $\times$ 32 images
- Architecture:
  - 3 $\times$ 3 conv layer, 16 channels, 32 $\times$ 32 pixels
  - 3 $\times$ 3 conv layer, 16 channels, 32 $\times$ 32 pixels-2$n$ layer
  - 3 $\times$ 3 conv layer, 32 channels, 16 $\times$ 16 pixels-2$n$ layer
  - 3 $\times$ 3 conv layer, 64 channels, 8 $\times$ 8 pixels-2$n$ layer
  - average-pooling layer
  - 10-d fully-connected layer
  - sofmax layer 
  - Mode **A** shortcut
- implementation
  - Batch Size: 128
  - Learning Rate:  0.1 and *divided by 10 when 32K & 48K epoch*
  - Epochs: 64K 
  - Weight Decay: 0.0001
  - Momentum: 0.9
- Results in $n=\{3,5,7,9\}$ 
  - Plain net：suffer from increased depth and exhibit higher training error
  (**same as on ImageNet**)
  - Resnet：overcome the optimization difficulty and demonstrate
accuracy gains (**same as on ImageNet**)

- Analysis of Layer Responses
  - Results
    - residual functions closer to zero than the non-residual functions
    - an individual layer of ResNets tends to modify the
signal less
  - Graph: 
    - **Top 图**： 层按原始顺序排列（从输入到输出）
      - Observation
        - **plain 模型**（虚线）：std 在前几层很高（>2），然后剧烈波动，甚至在某些层出现尖峰（如 plain-56 在第 40 层附近）
        - **ResNet 模型**（实线）：std 从一开始就低（<2），且**平稳下降**，几乎没有剧烈波动
      - 含义：
        - plain 模型中，有些层（尤其是中间层）对信号进行了**剧烈修改**（高 std）
        - ResNet 中，各层的响应更平缓，**没有某一层突然“爆炸”式地改变信号**
        - **ResNet 的残差函数 $F(x)$ 更温和，不会像普通网络那样让某一层主导整个变换过程**
    - **Bottom 图**：所有层的响应标准差按**降序排列**（最大在前）
      - Observation
        - 所有模型的 std 都从高到低递减
        - **plain 模型**（虚线）：前面几个层的 std 很高（>2），但很快下降
        - **ResNet 模型**（实线）：前面几个层的 std 较低（~1.5），且**下降速度更快、更平滑**
      - 含义：
        - 在 plain net中，**少数几个层承担了大部分信号变化**（即“关键层”）
        - 在 ResNet 中，**所有层的响应都较小且均匀分布**，没有明显的“主控层”
        - **ResNet 的学习是“分布式”的**，每层只做微调，而不是由少数层完成主要任务
      - 实际意义：
        - 深度 ResNet **不依赖**某几层“强力修改”信号
        - 而是通过**大量浅层微调**逐步逼近目标
        - 类似于“渐进式学习”：每一步只做一点点调整，避免破坏已有结构

- over 1000 layers
  - no optimization difficulty
  - 1202 layers($n=600$)
    - params:19.4M FLOPs
    - error rate: 7.93% 
### 4.3. Object Detection on PASCAL and MS COCO
- good generalization performance on other tasks
