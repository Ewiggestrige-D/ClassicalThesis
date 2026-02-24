---
title: AlexNet精读
markmap:
  colorFreezeLevel: 12
---

## AlexNet

- [Local Thesis](../AlexNet/NIPS-2012-ImageNet%20Classification%20with%20Deep%20Convolutional%20Networks.pdf)
- Basic info 
  - Author:
    - Alex Krizhevsky*（Main）
    - Ilya Sutskever
    - Geoffrey E. Hinton 
  - Publish:
    - Time: 2012
    - Journal: NeurIPS


## Abstract
- Dataset: ImageNet LSVRC-2010 
  - 1.2M Images
  - 1000 classes
- Results in error-rate :*Better than previous SOTA*
  - 37.5% in Top-1
  - 17.0% in Top-5
- Parameters:
  - **60M** parameters
  - **650k** neurons
  - **5** convolutional layers
  - **3** fully-connected layers
  - **1000** softmax-outputs
- optimization
  - **Dropout** for prevent *Overfitting*

## 1 Introduction
- **CNN** in HISTORY
  - controlled by depth & breadth
  - correct assuumption about nature of image
    - stationarity of statistics 
    - locality of pixel dependencies
  - much fewer connections and parameters 
    - compared to *standard feedforward neural networks，FNN*(标准前馈神经网络) 
    with similarly-sized layers,
      - 其结构由*输入层、隐含层和输出层* 单向连接组成，
        信号通过**有向无环图单向传播**，
        各层神经元**仅与相邻层建立连接**
      - FNNs:
        - **Perceptrons**
        - **BP-FNN**, Back-Propagation Feedforward Neural Networks
        - **RBF-FNN**, Radial-Basis-Function Feedforward Neural Networks
  - Demerits :
    - hard to apply in large scale
    - unable to find large datasets contain enough labeled examples to train
- Contributions
  - SOTA
  - new and unusual features 
  - techniques for preventing overfitting
  - **" Depth is IMPORTANT "**
  - **" END to END "**
## 2 Datasets
- ImageNet in 2010
  - examples : $\geq$ 15M labeled high-resolution images
  - categories : $\geq$ 22K
- ILSVRC-2010
  - Trainging Sets : $\approx$ 1.2M images
  - Validation Sets : $\approx$ 50k images
  - Test Sets : $\approx$ 150K images
- Image input 
  - a constant input dimensionality for AlexNet
    - rescaled the shorter side was of length 256
    - cropped out the central 256 $\times$ 256 patch 
      from the resulting image
  - **No Pre-Processing** : ***RAW*** RGB values
    - **" END to END "** 
  
## 3 Architecture
### 3.1 ReLU Nonlinearity
- Activation Function
  - Saturating
    - Defintion : 激活函数在某个区域的导数趋近于0,被称为 “饱和”
    - examples : sigmiods/Tanh
  - Non-Saturating
    - Defintion : 激活函数在**至少一个方向上导数**不趋近于 0。
    - examples : ReLU
  - 为什么饱和函数导致训练*慢*？
    - 梯度消失的连锁反应
  - 常见误解
    - “ReLU 完全没有梯度消失” 
      - 事实：ReLU 在负区梯度为 0（“神经元死亡”），但正区无饱和，整体比 Sigmoid/Tanh 好得多。
    - “所有非饱和函数都好”
      - 事实：需平衡表达能力与优化性。例如 $f(x) =x$（线性）不饱和，但无法引入非线性。
  - ReLU 的变体
    - Leaky ReLU: $f(x) =\max (0.01x,x)$
    - Parametric ReLU (PReLU):负区斜率可学习
    - ELU:负区平滑且有非零梯度

### 3.2 Training on Multiple GPUs (*Engineering*)
- Limitations
  - GTX 580 GPU - 3GB Memory
- cross-GPU parallelization
  - read from and write to 
    another’s memory 
  - trick : GPUs communicate only in **certain** layers.

### 3.3 Local Response Normalization
- $b_{x,y}^i = \frac{a_{x,y}^i}{\left( k + \alpha \sum_{j=\max(0, i-n/2)}^{\min(N-1, i+n/2)} (a_{x,y}^j)^2 \right)^\beta}$
  - symbols
    - $a_{x,y}^i$: 原始激活值（ReLU 之后）
    - $b_{x,y}^i$: 归一化后的激活值
    - $N$: 总通道数
    - $n$: **归一化窗口大小**（跨通道邻域），默认值5
    - $k$: 避免除零的小常数，默认值2
    - $\alpha$: 缩放因子，默认值10⁻⁴
    - $\beta$: 指数，默认值0.75
  - Key: **LRN 只在通道维度操作，不改变空间维度**（即对每个 $(x,y)$ 独立处理）
  - Effect ：类似“局部 softmax”，但更平滑。
    - **高激活通道被抑制**（分母变大）
    - **低激活通道相对凸显**
    - **增强通道间的差异化表达**
  - Source ：在真实视觉皮层中，**相邻神经元会相互抑制**（侧抑制 Lateral Inhibition）
    - 增强局部对比度
    - 抑制无用响应
    - 促进**局部竞争**（local competition）
  - 为什么 LRN 很快被淘汰？
    - 效果不稳定
      - 在其他数据集（如 CIFAR）上**几乎无效甚至有害**
      - 对超参数（$n, \alpha, \beta$）极度敏感
    - 计算开销大
      - 需要跨通道平方求和 ，内存访问不友好
      - 无法像 BatchNorm 那样高效实现
    - 被 Batch Normalization (2015) 全面超越
      - BN 在**批维度（Batch Dimensionality）** 归一化，稳定训练 + 加速收敛
      - BN 提供更强的正则化效果，且**与激活函数解耦**
      - **ResNet、Inception v2+ 全部弃用 LRN，改用 BN**

### 3.4 Overlapping Pooling
- grid of pooling units spaced $s$ pixels apart
- each summarizing a neighborhood of size $z\times z$ 
centered at the location of the pooling unit
- set $s < z$, obtain **Overlapping Pooling**
- *Instance* : $s=2,z=3$
- *Result*：reduces the top-1 and top-5 error rate
compared to  $s=2,z=2$

### 3.5 Overall Architecture
- 8 Layers
  - **5** convolutional layers
  - **3** fully-connected layers
  - **1000** softmax-outputs
- maximize **multinomial logistic regression**
  - equvilant: maximize *log-probability*
- Architecture in GPU
  - conv-layer 2/4/5 : single-connected only to previous kernel map in same GPU
  - conv-layer 3 : fully-connected to kernel map in both GPUs
  - Local Response Normalization : follow conv-layer 1/2
  - Max-Pooling: follow conv-layer 1/2/5
- Layers Format
  - Layer 1(Conv)
    - Input size: $3 \times 227 \times 227$ 
            (channels $\times$ heights $\times$ widths )
    - 96 Kernels (in 2 GPUs) :
    $3 \times 11 \times 11$, stride 4, padding 0
    - calculation ：$\left\lfloor \frac{227 + 2 \times 0 - 11}{4} \right\rfloor + 1 = \left\lfloor \frac{216}{4} \right\rfloor + 1 = \left\lfloor 54 \right\rfloor + 1 = 55$
    - Output size ： $48 \times 55 \times 55$  in single GPU
    - Activation:"ReLU"
    - Local Response Normalization: "True"
    - Max-Pooling : "True"
      - kernel：$3 \times 3$, stride 2, padding 0
        - $\left\lfloor \frac{55 + 2 \times 0 - 3}{2} \right\rfloor + 1 = \left\lfloor 26 \right\rfloor + 1 = 27$
      - Max-Pooling Output size ：$96 \times 27 \times 27$
  - Layer 2(Conv)
    - Input size: $48 \times 27 \times 27$
    - 256 Kernels (in 2 GPUs) : $48 \times 5 \times 5$, stride 1, padding 2 
    (数据来源[torchvision.models.alexnet](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/alexnet.html))
    - Output size ： $128 \times 27 \times 27$ in single GPU
    - Activation:"ReLU"
    - Local Response Normalization: "True"
    - Max-Pooling : "True"
      - kernel：$3 \times 3$, stride 2, padding 0
      - Max-Pooling Output size ：$256 \times 13 \times 13$
  - Layer 3(Conv & **Cross-Connected**)
    - Input size: $256 \times 13 \times 13$
    - 384 Kernels (in 2 GPUs) : $256 \times 3 \times 3$, stride 1, padding 1
    - Output size ： $192 \times 13 \times 13$ in single GPU
    - Activation:"ReLU"
  - Layer 4(Conv)
    - Input size: $192 \times 13 \times 13$
    - 384 Kernels (in 2 GPUs) : $192 \times 3 \times 3$, stride 1, padding 1
    - Output size ： $192 \times 13 \times 13$ in single GPU
    - Activation:"ReLU"
  - Layer 5(Conv)
    - Input size: $384 \times 13 \times 13$
    - 256 Kernels (in 2 GPUs) : $192 \times 3 \times 3$, stride 1, padding 1
    - Output size ： $128 \times 13 \times 13$ in single GPU
    - Activation:"ReLU"    
    - Max-Pooling : "True"
      - kernel：$3 \times 3$, stride 2, padding 0
      - Max-Pooling Output size ：$128 \times 6 \times 6$ 
  - Layer 6(Fully-Connected)
    - Input size:$256 \times 6 \times 6$ 
    - Flatten: $256 \times 6 \times 6 = 9216$
    - Weight : $W_{6} \in \mathbb{R}^{4096 \times 9216}$
    - Output size : 4096
    - Activation:"ReLU" 
  - Layer 7(Fully-Connected)
    - Input size:$4096 \times 1$ 
    - Flatten: 4096
    - Weight : $W_{7} \in \mathbb{R}^{4096 \times 4096}$
    - Output size :  4096
    - Activation:"ReLU"
  - Layer 8(Fully-Connected)
    - Input size:$4096 \times 1$ 
    - Flatten: 4096
    - Weight : $W_{8} \in \mathbb{R}^{1000 \times 4096}$
    - Output size : 1000 

## 4 Reducing Overfitting
### 4.1 Data Augmentation
- artificially enlarge datasets 
  using label-preserving transformations
    - generating image translations 
      and horizontal reflections
        - extract 5 224 $\times$ 224 patches (4-corner patches and the center patch)
        - their horizontal reflections
        - average scores of softmax-layer on 10 patches.
    - alter the intensities of RGB channels
      - PCA（**Principal Component Analysis**，主成分分析） every images
      - add multiples of components
        - augmented-multiplier $\sim \text{Gaussian} \quad N(\mu = 0, \sigma^2 = 0.1^2)$
      - each RGB pixel $ I_{xy} = [I^R_{xy}, I^G_{xy}, I^B_{xy}]^T $
        - add $[\mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3] \begin{bmatrix} \alpha_1\lambda_1，  \alpha_2\lambda_2 ，\alpha_3\lambda_3 \end{bmatrix}^{T}$
          - $\mathbf{p}_i$ :第 $i$ 个特征向量（eigenvector）
          - $\lambda_i$ : 第 $i$ 个特征值（eigenvalue）
          - $\alpha_i$ ： 从高斯分布 $\mathcal{N}(0, 0.1^{2})$ 中采样的随机变量
          - $[\mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3]$ : RGB 通道协方差矩阵的正交基（主成分方向）

### 4.2 Dropout
- $P(\text{neurons} = 0 | \forall \text{neurons} \in \text{hidden-layers})=0.5$
  - Forward Pass：No Contribution
  - Back-Propogation: No Participation
- Apply：Layer 6 & 7
## 5 Details of learning
- train
  - methods:**stochastic gradient descent,SGD**
  - Formulae
    - Velocity Update：$ v_{i+1} := 0.9 \cdot v_i - 0.0005 \cdot \eta \cdot w_i - \eta \cdot \left\langle \frac{\partial L}{\partial w} \bigg|_{w_i} \right\rangle_{D_i} $
    - Weight Update： $w_{i+1} := w_i + v_{i+1} $
  - Initiation
    - initial weights in **all layers** : $w_{1} \sim \mathcal{N}(0, 0.1^{2}) $
    - biases
      - 1, in Conv Layer 2 & 4 & 5 , all FC layers
      - 0, in Conv Layer 1 & 3
  - batch size:128
  - Taraining Set:**1.2M** images 
  - Epoch:90
  - momentum:0.9 （高动量 → 加速收敛）
    - 目标：**平滑梯度波动，加速收敛**
    - 新替代：cos学习率
    - 核心作用
      - **平滑梯度噪声**  
         - Batch gradient 有随机性（因数据采样）
         - 动量通过“记忆过去梯度”减少抖动
      - **加速收敛**  
         - 当梯度方向一致时，动量会“加速”更新
         - 类似滚石下山：越滚越快
      - **穿越鞍点和浅谷**  
         - 小梯度区域可能被“卡住”
         - 动量提供惯性，帮助跳过局部极小值
      - **提高鲁棒性**  
         - 对学习率不那么敏感
         - 更适合大规模训练
  - Weight Decay：0.0005（小的权重衰减 → 稳定训练并提升泛化）
    - 目标：惩罚大权重，防止过拟合
    - 替代：**权重衰减等价于 L2 正则化**
    - 核心作用：
      - **避免权重过大导致激活饱和**（如 ReLU 输出恒定）
      - **改善优化路径**：使损失函数更平滑，梯度更稳定
      - **增强数值稳定性**：防止梯度爆炸
  - Learning Rate($\eta$): dynamic, adjusted manually
    - initiation: 0.01
    - heuristic:reduced by **10** when the validation error rate stopped improving.
## 6 Results 
### ILSVRC-2010
- Sparse coding
  - Top-1:47.1%
  - Top-5:28.2%
- SIFT + FVs    
  - Top-1:45.7%
  - Top-5:25.7%
- AlexNet(CNN)
  - Top-1:37.5%
  - Top-5:17.0%
  
### 6.1 Qualitative Evaluations
- differences
  - GPU 1:color-agnostic
  - GPU 2:color-specific
- Good semantically similarity
## 7 Discussion
- a large, **deep convolutional neural network** is good
 on datasets using **purely supervised learning**