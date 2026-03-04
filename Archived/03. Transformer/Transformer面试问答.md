# Transformer面试20题
1. Transformer为何使用多头注意力机制？（为什么不使用一个头）
2. Transformer为什么Q和K使用不同的权重矩阵生成，为何不能使用同一个值进行自身的点乘？ （注意和第一个问题的区别）
3. Transformer计算attention的时候为何选择点乘而不是加法？两者计算复杂度和效果上有什么区别？
4. 为什么在进行softmax之前需要对attention进行scaled（为什么除以dk的平方根），并使用公式推导进行讲解
5. 在计算attention score的时候如何对padding做mask操作？
6. 为什么在进行多头注意力的时候需要对每个head进行降维？（可以参考上面一个问题）
7. 大概讲一下Transformer的Encoder模块？
8. 为何在获取输入词向量之后需要对矩阵乘以embedding size的开方？意义是什么？
9. 简单介绍一下Transformer的位置编码？有什么意义和优缺点？
10. 你还了解哪些关于位置编码的技术，各自的优缺点是什么？
11. 简单讲一下Transformer中的残差结构以及意义。
12. 为什么transformer块使用LayerNorm而不是BatchNorm？LayerNorm 在Transformer的位置是哪里？
13. 简答讲一下BatchNorm技术，以及它的优缺点。
14. 简单描述一下Transformer中的前馈神经网络？使用了什么激活函数？相关优缺点？
15. Encoder端和Decoder端是如何进行交互的？（在这里可以问一下关于seq2seq的attention知识）
16. Decoder阶段的多头自注意力和encoder的多头自注意力有什么区别？（为什么需要decoder自注意力需要进行 sequence mask）
17. Transformer的并行化提现在哪个地方？Decoder端可以做并行化吗？
19. Transformer训练的时候学习率是如何设定的？Dropout是如何设定的，位置在哪里？Dropout 在测试的需要有什么需要注意的吗？
20. 解码端的残差结构有没有把后续未被看见的mask信息添加进来，造成信息的泄露。

---
## 回答：
1. 多头注意力的机制类似CNN中的多通道，能够学习到不同的QKV关系的信息，最终合成一个更大的语义空间。相比于一个头，能明显注意到更多的语义关系和信息，理解更多更抽象的内容。
2. Query和Key共享权重会造成encoder或decoder在生成注意力分数的时候始终给自身信息打分最高，极大地影响softmax输出，同时影响整个模型理解这个位置和其他位置的关系。
3. additive attention需要对于query和key矩阵进行前馈全连接的隐藏层计算，需要额外在调用内存计算两个weight tensor，从计算结果上两者的作用基本相同，但是点积注意力的计算开销要更小。
4. 在进行softmax时，让数据更分散，而不是更集中在0，1这样的饱和区
假设：
$$
\mathbf Q_{\forall,:} \sim \mathcal{N}(0,1) \\
\mathbf K^\top_{\forall,:} \sim \mathcal{N}(0,1)
$$
那么对于未被缩放的$\widetilde{A} =\mathbf Q \cdot \mathbf K^\top $,其中任意元素$\widetilde{A}_{ij} =\mathbf Q_{i,:} \cdot \mathbf K^\top_{:,j} $,即
$$\widetilde{A}_{ij} =\mathbf Q_{i,:} \cdot \mathbf K^\top_{:,j} \\
= \sum_{m=1}^{d}\mathbf Q_{i,m} \cdot \mathbf K^\top_{m,j} 
$$
分析其方差
$$
\begin{aligned}
& Var(\mathbf Q_{i,m} \cdot \mathbf K^\top_{m,j} )\\
& = E(\mathbf Q_{i,m}^{2} \cdot \mathbf (K^\top_{m,j})^{2} )-E(\mathbf Q_{i,m} \cdot \mathbf K^\top_{m,j} )^2 \\
& = E(\mathbf Q_{i,m}^{2}) E(\mathbf (K^\top_{m,j})^{2} )-E(\mathbf Q_{i,m})^{2} E(\mathbf K^\top_{m,j})^{2} \\
& = \left(Var(\mathbf Q_{i,m})+E(\mathbf Q_{i,m})^{2}\right)\left(Var(K^\top_{m,j})+E(K^\top_{m,j})^{2}\right)-E(\mathbf Q_{i,m})^{2} E(\mathbf K^\top_{m,j})^{2} \\
& = (1+0)(1+0)-0 \times 0  \\
& = 1
\end{aligned}
$$
因此求和之后
$$
Var(\widetilde{A}_{ij} ) = d \longrightarrow Std(\widetilde{A}_{ij} ) = \sqrt{d}
$$
因此缩放之后的依旧服从正态分布
$$
{A}_{ij}  \sim \mathcal{N}(0,1)
$$
5. 在decoder中，由于decoder的输入是将query和上一时刻的输出concat之后作为输入，因此在预测序列i+1时， 将weighted Query中行列都大于i的矩阵元换为0，或者将进入softmax之前的add norm之后的masked输出换成一个极大的负值-1e10，保证softmax之后的输出不会影响到没有masked的部分
6. 降低维度保证n_head * reduced_dim =  dim_model,使得整个计算不需要额外的全连接层来匹配多头维数和model的维度
7. Transformer的encoder是由一个带有位置编码positional encoding的嵌入层和6个完全相同的block组成，每个encoder block有两个sublayer
   1. self-attention + add + layer norm 其中注意力层有8个头
   2. Feed-Forward Net(FFN)+ add + layer norm
   由于在self-attention layer中每一个query都要和所有位置的KV做内积，而FFN是一个带有隐藏层的全连接层，因此两个sublayer中都忽视了sequence的相对位置信息，因此必须在嵌入层加入位置编码信息。
   在首先传入的序列w被带有位置编码positional encoding的嵌入层转换到预料的向量空间。
   然后将向量复制层三份分别作为QKV输入到self-attention层与对应的weight tensor相乘得到weighted Query/Key/Value.此时在通过add&norm，利用残差链接加上原来的Query/Key/Value，并分别进行layer norm. layer norm即对于输入batch中的每一个单独的sequence，其中的所有feature进行normalization。将norm之后的结果输入到第二层FFN，将多头的注意力通过FFN的结构concat起来，同时压缩每一层的维度使得输入输出的总维度保持不变，因此需要保证n_head * reduced_dim =  dim_model。
8. 为了保证输入的数量级和位置编码的数量级在一个等级上。对于原版Transformer，使用了sinoid位置编码，值域范围在(0,1)，而嵌入之后的位置编码的值域均值为$\frac{1}{\sqrt{size_embeding}}$，在embeddingsize 等于2048时， 语义向量会被忽略不计导致后续学习中严重丢失信息，因此需要将输入层乘以$\sqrt{size_embeding}$使得两部分的值域大小保持相对一致
9. Transformer的原始注意力编码使用的是sinuoid位置编码具体形式如下
$$
\begin{aligned}
   p_{i,2j} & =\sin \left(\frac{i}{10000^{2j/d_{\text{model}}}} \right), \\
   p_{i,+1} & =\cos \left(\frac{i}{10000^{2j/d_{\text{model}}}} \right)
\end{aligned}
$$
这种编码形式有几种优点
   1. 对于数量相同的d_model能划分更多的hilbert空间作为语义空间，对比线性划分，相同d_model情况下能构造更大的语义空间，理解更复杂的语义
   2. 将向量之间的绝对位置信息转变为能用空间旋转矩阵mapping后得到的向量空间信息，提高了各个token之间的位置信息关联性
10. 不了解
11. 残差结构来自于Resnet，已经变成了现在深度学习模型的基本架构之一，其特点是通过identical mapping使得模型的优化不会比上一层更差。residual结构最开始设计是为了解决深度神经网络传递之后的梯度爆炸或者消失的问题吗，在此背景下residual保证了梯度不小于1，即梯度更新不会消失，而从更高的视角来看，identical mapping保证了信息传递的稳定性，尤其是在RNN中，经过长序列或长记忆的有损压缩，很有可能丢失某些重要的语义信息和细节，而identical mapping即保证了长记忆传输的可靠性，使得模型优化过程中对于序列开始的比重始终不会降的很低。
12. layer norm在encoder和decoder的每个sublayer中都有使用，加入残差结构之后都需要使用layenorm. 对比batchnorm，ln是将每个batch中的每个序列输出对所有的feature做normalization,即对于输入形态为（batch_size,size_QKV,size_hidden）的tensor对batch中每个输入的（size_QKV,size_hidden）做norm。这样可以保证对于不同长度的输入，带有不同的padding或者mask也能做norm并且不干扰结果输出的尺寸一致性。
13. batchnorm就是将输入的batch和seq在feature维度上做normalization，保证每个feature维度上的数满足N(0,1)分布，这样的处理方式使得输出结果再下一层中的softmax或者激活中能够有效的输出，不会被样本中具体数值的大小影响分布的趋势，将整体的分布都控制在合适的范围内。如果不进行norm处理，则在softmax或者其他激活时，输出结果可能会集中在0和1附近，对于输出完整的语义信息不利
14. Transformer中的FFN是使用了ReLU的带有一个隐藏层的MLP，且中间进行了一次升维和降维，将512 d_model升到了2048.这样做的意义是将单薄的语义空间升到高维，提升复杂语义和抽象语义的空间，使得能提取出更多的语义信息。同时降维是得residual结构前后size一致，能完成直接相加，同时降低计算开销。ReLU函数在这个过程中提供了足够的非线性，保留权重为正的输出，将信息之间的相关性大大提升，同时将负相关的输出过滤，降低语义信息检索的复杂度。
15. encoder经过训练之后输出最终的K-V pair，decoder将masked self-attention之后输出的Query对所有的encoder输出KV做注意力内积，得到一个输出，并传入下一层的sublayer中。
16. decoder的第一个sublayer是masked self-attention，即对于时间步n，n之后的序列输出全部被masked为非常大的负数，这样在softmax之后的输出为0.使用这种输出是因为，decoder是时序序列输出，他的主要目的就是预测下一个token的最大输出概率。那么在训练过程中就不能看见之后的输出
17. 并行化体现在在self-attention pooling过程中，我们能一次性直接拿到全部的序列的weighted QKV，而根据self-attention的原理，我们需要让每一个token对所有位置做内积计算attention score，此时的计算可以并行执行因为再次过程中不涉及保持token之间相对关系的仿射计算
19. 学习率是在最终的反向传播阶段确定的，使用了adam优化器；dropout是在add norm阶段加入的dropout概率0.1
20. 没有添加被masked信息