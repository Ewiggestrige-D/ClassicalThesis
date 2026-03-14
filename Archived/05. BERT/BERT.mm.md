---
title: BERT精读
markmap:
  colorFreezeLevel: 12
---

## Transformer

- [Local Thesis](/Archived/05.BERT/1810.04805v2_BERT%20Pre-training%20of%20Deep%20Bidirectional%20Transformers.pdf)
- Basic info 
  - Author:
    - Jacob Devlin
    - Ming-Wei Chang
    - Niki Parmar
    - Kenton Lee
    - Kristina Toutanova
  - Publish:
    - Time: 2018
    - Journal: arXiv 

## Abstract
- **BERT**: Bidirectional Encoder Representations from Transformers
- **Purpose**: pre-train deep bidirectional representations from **unlabeled text** by jointly conditioning 
on both *left and right* context in **all layers**.
- **Achievement**:*one additional output layer* for other tasks
- **Result**: 
  - GLUE: 80.5% (7.7% point absolute improvement)
  - MultiNLI: 86.7% (4.6% absolute improvement)
  - SQuAD v1.1 question answering Test F1:  93.2 (1.5 point absolute improvement)
  - SQuAD v2.0 Test F1: 83.1 (5.1 point absolute improvement)

## 1 Introduction
- Language Model Tasks
  - **Sentence**-level:
    - natural languaage inference
    - paraphrasing
      - predict the relationships between sentences by analyzing them **holistically**
  - **Token**-level:
    - named entity recognition
    - question answering

- Pre-training Strategy
  - **Feature-Based**
    - Model: ELMo
    - Reference: [Deep contextualized word representations](https://arxiv.org/abs/1802.05365)
    - **Architecture**: Task-specific architecture. include pre-trained representations as **adiitional features**
  - **Fine-Tuning**
    - Model: GPT
    - Reference: [Improving Language Understanding by Generative Pre-Training](https://www.semanticscholar.org/paper/Improving-Language-Understanding-by-Generative-Radford-Narasimhan/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035)
    - **Architecture**: minimal task-specific parameters. using **fine-tuning** all trained-params on downstreaam tasks
- Restriant on current pre-trained models
  - OpenAI GPT: using left-to-right architecture
  - where every token can only attend to previous tokens in the self-attention layer
  - demirits: sub-optimal for sentence-level tasks, espicially for quastion answering
- BERT's advantage
  - using a "masked language model" pre-training objective
  - **Masked** mechanism: *randomly* maskes some tokens from input and to predict *masked* word based on contexts 
  - enbale models use left and right context
- **Contribution**
  - demostration of importance of bidirectional pre-training for language representation
  - pre-trained representation reduce the need for task-specific acchitecture
  - SOTA   for mamy NLP tasks

## 2 Related Work

### 2.1 Unsupervised Feature-based Approaches
- Pre-trained word embeddings are an integral of modern NLP system
  - **Training meteods**; left-to-right models; discriminate correct from wrong (labeled)
  - generalization: 
    - sentence embedding
    - paragraph embedding
    - sentence generation
      - rank the next sentence(*softmax output*)
      - left-to-right generation
      - denoising auto-encoder
- ELMo
  - extract context-sensitive feature (**Feature-based** approach)
  - integrating with task-specific achitecture
  - **Output**: contextual repesentations of each token is *concatanation of left-to-right & right-to-left reps*

### 2.2 Unsupervised Fine-tuning Approaches
- Pre-trained embedding such as **GPT**
- **Training methods**: unlabeled text and fin-tuned for a **supervised** downstream task
- **Advantage**:dew parameters need to be learned from scratch

### 2.3 Transfer Learning from Supervised Data
- transfer from supervised tasks with large datasets
- methods: fine-tuning
- **BERT 和之后的大量工作证明了使用没有label的大量的dataset上的工作要好于稍小的有label的dataset**
## 3 BERT
### 3 BERT
- 2-steps in framework
  - **pre-training**
    - train on **unlabeled** data over different pre-training tasks.
  - **fine-tuning**
    - initialized with **pre-trained params**
    - using labeled data from the downstream tasks to **fine-tune**
    - different downstream tasks $\leftrightarrow$ *differernt* fine-tuned models but *same* initial params
- [BERT Architecture](/Archived/05.%20BERT/img/BERT%20Architecture.png)
- **Model Architecture**
  - **multi-layer bidirectional** Transformer encoder
  - Num of params
    - number of layers(Transformer blocks) $L$
    - hidden size $H$
    - self-attention head(**multi-head** self-attention) $A$
  - params的主要来源是嵌入层和Transformer block
  - $\text{BERT}_{\text{base}}$
    - $L$ = 12, $H$=768, $A$=12
    - Num of params 
      - 嵌入层： 30K(size of vocab) $\times$ $H$ 
      - Transformer block：12 $\times$ (3 $\times$ $H^2$ (QKV投影矩阵+多头concat) +  $H^2$(多头concat) + $H \times 4H \times 2$(双层MLP) )
      - 30K $\times$ 768 + 12 $\times$ (12 $\times$ 768 $\times$  768) = 107,974,656 $\approx$ 110M
  - $\text{BERT}_{\text{large}}$
    - $L$ = 24, $H$=1024, $A$=16
    - Num of params = 30K $\times$ 1024 + 24 $\times$ (12 $\times$ 1024 $\times$  1024) = 332,709,888 $\approx$ 332M
- **Input/Output Representations**
  - **Purpose**: represent a *single* sentence 
  and a *pair* of sentences (e.g., <Question, Answer>) in one token sequence.
  - **Sentence**: an arbitrary span of **contiguous text**, rather than an actual linguistic sentence
  - **Sequence**: a single sentence or two sentences packed together
  - **Methods**: 
    - WordPiece embeddings ($E$) - 30k tokens vocabulary
    - special tokens
      - [CLS]($C \in \mathbb{R}^{HY}$): first token of every sequence
      - [SEP]: seperate differentiate he sentences
    - a **learned embedding** to distinguish whether token is in sentence A or B
  - [BERT input representation](/Archived/05.%20BERT/img/BERT%20input%20representation.png)
  - Input Architecture:Token Embeddings(semantics label) + Segment Embeddings(sentence label) + Position Embeddings(postion label)

### 3.1 Pre-training BERT
- Task #1: **Masked LM**
  - standard conditional LMs can only be trained *left-to-right* or *right-to-left*
  - **Methods**: mask 15% input tokens at random, 
  **final hidden vectors** fed into an output **softmax** over the vocabulary
  - **Output**: *predict the masked words*
  - **Issue**: a mismatch between pre-training and fine-tuning
    - for the 15% masked token
      - 80%: [MASKED] token 
      - 10%: random token replace
      - 10%: unchanged
- Task #2: **Next Sentence Prediction** (NSP)
  - relationship between two sentences not directly captured by LM
  - **Methods**: a binarized next sentence prediction task,
  sentences A and B pair, 
    - 50% of the time B is actual next sentence (labeled as IsNext), 
    and 50% is a random sentence (labeled as NotNext).
  - Compared to the prior task: BERT transfer all parameters to initialize end-task model parameters.
  - beneficial to both QA and NLI.
- **Pre-training data**
  - datasets
    - BooksCorpus: 800M words
    - English Wikipedia: 2,500M words
  - tips: to extract **long contiguous sequences**, 
  use a **document-level corpus** rather than a shuffled *sentence-level corpus*

### 3.2 Fine-tuning BERT
- **Methods**: self-attention mechanism
  - encoding a concatenated text pair
  - calculate **bidirectional** cross-attention between two sentences.
- plug in the task-specific inputs and outputs into BERT 
and **finetune** all the parameters end-to-end.
- **Input**:
  1. sentence pairs in paraphrasing
  2. hypothesis-premise pairs in entailment
  3. question-passage pairs in question answering
  4. a degenerate text-$\emptyset$ pair in text classification or sequence tagging
- **Output**:
  - token fed into *output layer* for **token-level** tasks
    - sequence tagging or question answering
  - special tokens [CLS] fed into an output layer for **classification**
    - entailment or sentiment analysis

## 4 Experiments
### 4.1 GLUE
- **new parameters introduced**: classification layer weights $W \in \mathbb{R}^{K \times H}$
  - $K$ is the number of labels
  - $H$ is the hidden size
  - **classification Gradient Descent**: $\log\left(softmax(CW^{T}) \right)$
- **Train**
  - batch size: 32
  - Epoch: 3 (little bit small?)
  - learning rate: choose best 5
- **Issue**: $\text{BERT}_{\text{large}}$ sometimes unstable on small datasets(over-fitting)
- **Result**: 4.5% and 7.0% respective average accuracy improvement over SOTA
- $\text{BERT}_{\text{large}}$ significantly outperforms $\text{BERT}_{\text{base}}$ across all tasks,
especially those with very little training data.
### 4.2 SQuAD v1.
- **Method**: packed input question and related passage
  - question use $A$ embedding
  - passage use $B$ embedding
  - start vector: $S \in \mathbb{R}^{H}$
  - end vector: $E \in \mathbb{R}^{H}$
  - Probability of word $i$(vocab index) in *start*:
    $$P_{i} = \frac{e^{S \cdot T_{i}}}{\sum_{j}e^{S \cdot T_{j}}}$$
  - Probability of word $i$(vocab index) in *end*:
    $$P_{i} = \frac{e^{E \cdot T_{i}}}{\sum_{j}e^{E \cdot T_{j}}}$$
  - Score in **candidate span** from start to end
    $$S \cdot T_{i} + E \cdot T_{j}$$ 
    maximum scoring span where $j>i$ as prediction
- **Training Objective**
  - log-likelihoods of the correct **start and end positions**
- **Train**
  - batch size: 32
  - Epoch: 3 (little bit small?)
  - learning rate: 5e-5

### 4.3 SQuAD v2.0
- no short answer exists in the provided paragraph, making the problem more realistic.
- **Method**: treat questions that do not have an answer as having an answer 
span with start and end at the [CLS] token.
  - score of null answer: 
    $$s_{\text{null}} = S \cdot C + E \cdot C$$
  - score of answer:
    $$\hat{S}_{i,j} =\max_{j \geq i} S \cdot T_i+E \cdot T_i $$
    when $\hat{S}_{i,j} > s_{\text{null}} + \tau$
    where $\tau$ is a hyper paramter
- **Train**
  - batch size: 48
  - Epoch: 2
  - learning rate: 5e-5

### 4.4 SWAG
- Dataset: 113k sentence-pair completion examples
- **new parameters $V$ introduced**: score for each choice 
  - $$softmax(V \cdot C)$$
- **Train**
  - batch size:16
  - Epoch: 3
  - learning rate: 2e-5

## 5 Ablation Studies
### 5.1 Effect of Pre-training Tasks
- No NSP: A **bidirectional** model which is trained using the “masked LM” (MLM) 
but without the “next sentence prediction” (NSP) task
  - hurts performance significantly on QNLI, MNLI, and SQuAD 1.1.
- LTR & No NSP: A standard Left-to-Right (LTR) LM
  - LTR model performs worse than the MLM model on all tasks
### 5.2 Effect of Model Size
- **Conlusion**:larger models **better** across all four datasets
- Far-known:  increasing the model size will lead to continual improvements on **large-scale tasks**
- **News**: scaling to extreme model sizes also leads to large improvements on **very small scale tasks**, 
  with model been sufficiently pre-trained.
  - examples: 
    1.  Peters et al. (2018b) add pre-trained bi-LM size from 2 to 4
    2.  Melamud et al. (2016) increasing hidden dimension size from 200 to 600
  - **Hypothesis**: when the model is *fine-tuned on the odwnstream tasks* 
  and uses only  small number of randomly initialized additional parameters, 
  the task-specific models can benefit from the **larger, more expressive pre-trained representations** even 
  when downstream task data is very small.
### 5.3 Feature-based Approach with BERT
- **Feature-based Task**: WordPiece model and formulate as a tagging task
- **Input**: the representation of the first sub-token as the input to the token-level classifier 
- **Methods**: extracting the activations from **one or more layers** without fine-tuning any parameters of BERT. 
- **Result**: best performing method concatenates the token representations from the **top four hidden layers** of the pre-trained Transformer
- **Conlusion**: BERT is effective for both **finetuning and feature-based** approaches.

## 6 Conclusion
- rich, **unsupervised** pre-training is an integral
part of many language understanding systems
- low-resource tasks to benefit from **deep unidirectional** architectures.