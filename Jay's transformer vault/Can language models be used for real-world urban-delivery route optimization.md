---
date: 2025-01-09
tags:
  - optimization
  - Word2Vec
---
# 1. 背景与研究问题
## 1.1 研究背景

- 目前的城市配送路径优化面临的挑战（如复杂的现实场景、无法捕捉驾驶员偏好等）。
- 传统优化方法（如TSP）的局限性。
- 提出语言模型在跨领域研究中的潜力。
## 1.2 研究问题

- 语言模型能否用于解决实际城市配送中的路径优化问题？
- 如何将语言模型与运筹学（TSP模型）结合，既学习隐式知识，又优化路径？

# 2. 方法论

	按照文章的叙述，drivers倾向先完成每个zone的配送，每个zone的配送顺序受到driver的行为模式的影响，因此运用词向量的相似性，学习到配送的路径。而在每个zone的内部，司机倾向于采取配送路程最短的策略，因此可以使用传统的tsp方法，得到每个zone内部的最短路径。
	混合模型由两个阶段构成：语言模型生成配送区域序列 + TSP优化区域内路径。

## 2.1 数据处理

- 从历史配送数据中提取配送路径序列，每条路径包括配送站和多个区域。
- 使用区域ID构造“配送行为句子”（delivery behavior sentences），每个句子的格式如下：
$$ s = (z_0, z_1, \dots, z_M)$$
	z0：配送站.
	z1,z2,…,zM​：配送区域的顺序。
## 2.2 [Word2Vec]([深入浅出Word2Vec原理解析 - 知乎](https://zhuanlan.zhihu.com/p/114538417))

### 2.2.1 构建配送路径的联合概率

通过历史数据，构建配送区域序列的联合概率 $P(z_0, z_1, \dots, z_M)$，Bayes公式如下：
$$P\left(z_0, z_1, \ldots, z_M\right)=\prod_{m=0}^M P\left(z_m \mid z_0, z_1, \ldots, z_{m-1}\right)$$
只要计算出条件概率即可计算出配送区域序列的的概率

### 2.2.2 条件概率的估计

考虑 $p\left(z_m \mid z_1, \ldots, z_{m-1}\right)$ 的近似计算。利用Bayes公式，有：

$$
p\left(z_m \mid z_1, \ldots, z_{m-1}\right)=\frac{p\left(z_1, \ldots, z_m\right)}{p\left(z_1, \ldots, z_{m-1}\right)}
$$


根据大数定理 ，当语料库足够大时，$p\left(z_m \mid z_1, \ldots, z_{m-1}\right)$ 可以近似地用频率表示为：

$$
p\left(z_m \mid z_1, \ldots, z_{m-1}\right) \approx \frac{\operatorname{count}\left(z_1, \ldots, z_m\right)}{\operatorname{count}\left(z_1, \ldots, z_{m-1}\right)}
$$

其中， $\operatorname{count}\left(z_1, \ldots, z_m\right)$ 表示词串 $z_1, \ldots, z_m$ 在语料中出现的次数， $\operatorname{count}\left(z_1, \ldots, z_{m-1}\right)$ 表示词串 $z_1, \ldots, z_{m-1}$ 在语料中出现的次数。可想而知，当 m 很大时， $\operatorname{count}\left(z_1, \ldots, z_m\right)$ 和 $\operatorname{count}\left(z_1, \ldots, z_{m-1}\right)$ 的统计将会多么的耗时。
在原文中，$\operatorname{count}\left(z_1, \ldots, z_m\right)$是计算$z_1, \ldots, z_{m-1}, z$($z$代表节点序列中的任意节点)

### 2.2.3 马尔可夫假设（N-gram模型）

为了降低计算复杂度，引入马尔可夫假设，即只考虑最近的 $k$ 个区域：

$$
P\left(z_m \mid z_0, z_1, \ldots, z_{m-1}\right) \approx P\left(z_m \mid z_{m-k}, \ldots, z_{m-1}\right)
$$
### 2.2.4 目标函数

对于统计语言模型而言，利用最大似然，可把目标函数设为：

$$
\prod_{z \in C} p(z \mid \operatorname{Context}(z))
$$

其中，C表示语料（Corpus），Context $(\mathrm{z})$表示词 $z$ 的上下文，即 $z$ 周边的词的集合。当Context $(\mathrm{z})$为空时，就取 $p(z \mid \operatorname{Context}(z))=p(z)$ 。特别地，对于n－gram模型，就有 $\operatorname{Context}\left(z_i=z_{i-n}, \ldots, z_{i-1}\right)$ 。

Word2Vec是轻量级的神经网络，其模型仅仅包括输入层、隐藏层和输出层，模型框架根据输入输出的不同，主要包括CBOW和Skip-gram模型。 CBOW的方式是在知道词$z_m$的上下文$z_{m−k}, z_{m−k+1},z_m, \ldots, z_{m+k-1}, z_{m+k}$的情况下预测当前词$z_m$.而Skip-gram是在知道了词$z_m$的情况下,对词$z_m$的上下文$z_{m−k}, z_{m−k+1},z_m, \ldots, z_{m+k-1}, z_{m+k}$进行预测。

![[Pasted image 20250109162337.png]]
- Skip-gram模型通常采用一个简单的神经网络架构，包括一个输入层、一个隐藏层和一个输出层。
- 输入层：将文本中的每个词转换为one-hot编码形式。
- 隐藏层：利用权重矩阵W将one-hot编码映射到低维空间，即词向量。
- 输出层：使用softmax函数计算概率分布，预测出上下文词。
![[Pasted image 20250109162659.png]]
原文采用的是Skip-gram模型，定义最大化区域间共现概率的目标函数为：
$$
\mathcal{L}(\theta)=\prod_{m=0}^M \prod_{-k \leq i \leq k, i \neq 0} P\left(z_{m+i} \mid z_m ; \theta\right)
$$
### 2.2.5 损失函数

转化为负对数似然损失：
$$\mathcal{L}(\theta)^{\prime}=-\frac{1}{M} \sum_{m=0}^M \sum_{-k \leq i \leq k, i \neq 0} \log P\left(z_{m+i} \mid z_m ; \theta\right)$$
$$
P\left(z_{m+i} \mid z_m\right)=\frac{\exp \left(v_{z_{m+i}}^T v_{z_m}\right)}{\sum_{z \in D} \exp \left(v_z^T v_{z_m}\right)}
$$
- $v_{z_m}$ ：区域 $z_m$ 的词向量。
- 使用 softmax 函数计算上下文区域的概率。
点积$\left(v_{z_{m+i}}^T v_{z_m}\right)$反映了两个区域之间的相似性，值越大表示越相似。
softmax函数用于将词向量的点积（相似性）转化为概率分布

### 2.2.6 推断配送区域序列：定制链式反应算法
![[Pasted image 20250109161627.png]]
- **输入**：
    - 配送站和所有配送区域的词向量。
- **逻辑**：
    - 从配送站$v_0$开始，逐步选择与当前区域词向量最相似的下一个区域。
    - 相似性使用 **余弦相似度** 计算：$\operatorname{similarity}\left(v_1, v_2\right)=\frac{v_1 \cdot v_2}{\left\|v_1\right\|\left\|v_2\right\|}$
- **输出**：
    - 最终生成配送区域的顺序。
## 2.3 步骤一：语言模型的整体流程
### **步骤 1：编码（Encoding）**

**目标**：将训练集中所有的配送区域ID和配送站ID转换为 **独热编码（one-hot encoding）** 向量。

- **独热编码**：
    - 对于具有 N 个不同区域和配送站的集合，每个ID被表示为一个长度为 N 的向量。
    - 在对应于该ID的位置上取值为1，其余位置为0。
    - 例如，若总共有5个区域和配送站，ID为3的区域的独热编码为 [0,0,0,1,0]。

**作用**：

- 将离散的、类别型的数据（区域ID）转换为神经网络可处理的数值型向量。
- 为后续的神经网络训练提供输入格式。

---

### **步骤 2：构建样本（Constructing samples）**

**目标**：以 **skip-gram** 模式构建训练样本，形成“特征-标签”对。

- **Skip-gram 模型**：
    - 在自然语言处理中，skip-gram 模型的目标是给定一个中心词（target word），预测其周围的上下文词（context words）。
    - 在本模型中，中心词对应于当前的配送区域 $z_m$​，上下文词对应于其前后 k 个配送区域。

**具体操作**：

- **特征（Feature）**：
    
    - 输入特征是当前区域 $z_m$​ 的独热编码向量。
    - feature $=$ one_hot $\left(z_m\right)$
- **标签（Label）**：
    
    - 标签是当前区域前后 k 个上下文区域的独热编码向量的拼接。
    - label $=$ concat $\left(\right.$ one $\_$hot $\left(z_{m-k}\right), \ldots$, one $\_$hot $\left(z_{m-1}\right)$, one $\_$hot $\left(z_{m+1}\right), \ldots$, one_hot $\left.\left(z_{m+k}\right)\right)$
    - **解释**：
        - $\text{concat}(\cdot)$：表示将多个向量进行拼接，形成一个更长的向量。
        - $z_{m−k}, z_{m−k+1},z_m, \ldots, z_{m+k-1}, z_{m+k}$ 是当前区域 $z_m$​ 的前后 k 个区域。
        - 这样，每个训练样本的输入是当前区域，标签是其上下文区域。

---

### **步骤 3：训练（Training）**

![[Pasted image 20250109170324.png]]

	由于one-hot编码的稀疏性以及无法计算词与词之间的相似性等缺点，所以我们希望通过一个稠密的向量来表示词语，即将词语映射到低维的向量表示，我们把这种映射叫做词嵌入(word embedding)。
	
**目标**：基于定义的损失函数，使用构建的训练样本更新神经网络的权重矩阵 W 和 W′。

- **神经网络结构**：
    
    - **输入层**：
        - 输入为独热编码的特征向量$\text{one\_hot}(z_i)$，维度为 O（总的区域和配送站数量）。
    - **隐藏层**：
        - 维度为 H，即词向量的维度（embedding size）。
        - 使用权重矩阵 $W \in \mathbb{R}^{H \times O}$ 将输入层映射到隐藏层： $$W\times \text{one\_hot}(z_i)$$
        - $v_{z_i}$ 是区域 $z_i$ 的词向量，维度为 H。
        - 矩阵乘法实际上相当于从 W 中提取第 i 列作为词向量 $v_{z_i}$。
    - **输出层**：
        - 维度为 O。
        - 权重矩阵 $W' \in \mathbb{R}^{O \times H}$ 将隐藏层映射到输出层。$$ u_{zi}​​=W^′⋅v_{zi}​​$$
        - $u_{zi}$​​ 是预测的上下文得分向量，维度为 O。
    - **激活函数**：
        - 输出层通常通过 Softmax 函数将得分转化为概率分布： $$P\left(z_{m+i} \mid z_m\right)=\frac{\exp \left(v_{z_{m+i}}^T v_{z_m}\right)}{\sum_{z \in D} \exp \left(v_z^T v_{z_m}\right)}$$
- **训练过程**：
    
    - **损失函数**：通常使用 **负对数似然损失函数（Negative Log-Likelihood Loss）**
    - **优化器**：使用梯度下降或其变体（如 SGD、Adam）来更新权重矩阵 W 和 W′。
    - **更新规则**：基于训练样本，不断调整权重，使模型能够准确预测上下文区域。

---

### **步骤 4：学习结果（Learning outcomes）**

**目标**：在训练完成后，得到权重矩阵W，获取每个区域的词向量表示。

- **词向量的计算**：
    
    - 对于第 i 个区域，其词向量 $v_{z_i}$（若是Ox1的列向量） 通过将其独热编码向量与权重矩阵 W（HxO） 相乘得到。
    - **公式**： $W\times \text{one\_hot}(z_i)$
        - $\text{one\_hot}(z_i)$ 是区域 $z_i$ 的独热编码向量，维度为 O。
        - $W\in \mathbb{R}^{H \times O}$ 是输入层到隐藏层的权重矩阵。
        - W 的列向量 W[:,i]是直接学习得到的每个区域的词嵌入，代表了区域的低维表示。
- **矩阵结构：
	- 行数为词向量的维度 H。
    - 列数为总的区域和配送站数量 O。
    - 每个区域的词向量对应于权重矩阵 W 的一列。

**作用**：

- 词向量 $v_{z_i}$​​ 捕捉了区域之间的语义关系，即在配送路径中常一起出现的区域，其词向量在向量空间中更接近。
- 这些词向量可用于后续的配送区域序列预测和路径优化。

## 2.4 步骤二：TSP模型

在语言模型生成区域序列后，使用 **TSP模型** 优化区域内的具体路径。