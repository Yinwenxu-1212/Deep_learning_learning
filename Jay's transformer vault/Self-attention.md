---
date: 2024-12-26
tags:
  - deep-learning
  - attention
---
# ATTENTION模型
## 1. 输入与输出

- **输入：**
 一堆向量
- **输出：**
1. 每一个向量都有一个label（输入输出数目一样->Sequence labeling）
2. 整个序列只有一个label
3. 模型自己决定了输出的label的数量（seq2seq)

## 2. Attention机制的本质思想
	从大量信息中，有选择地筛选出少量重要信息，并且聚焦到这些重要的信息上，忽略大多数不重要的信息。
	聚焦的过程体现在权重系数的计算上，权重越大，越聚焦于其对应的Value值上，即权重代表了信息的重要性，而Value是其对应的信息。

## 3. Sequence Labeling
### 3.1 机制

![[Pasted image 20241226192742.png]]
**Self-attention 处理的是整个序列的信息，得到考虑一整个序列的信息；FC(Fully Connected Network) 处理的是局部的信息。**
![[{7734D87E-552D-410B-98ED-2E53174495B5}.png]]
**self-attention可以不断叠加，与FC交替出现**

### 3.2 具体计算的步骤
 
**1. 找出在序列中的相关的向量，用α代表两个向量的关联性，根据Query与Key计算两者的相似性或相关性（这里用的是两个向量的点积）**
![[Pasted image 20241226194052.png]]![[Pasted image 20241226194617.png]]![[Pasted image 20241226195030.png]]
**α1左乘Wq，剩余的阿尔法左乘Wk，分别得到q1与k2，k3，k4·······一般来说α1也需要左乘Wq，得到k1。将q1与k1、k2、k3、k4······点乘可以得到相关系数α1,1 α1,2 α1,3（可以叫做attention score） ······ 

**2. 通过Soft-max机制，对相关系数进行归一化处理，得到α1,i'**
![[Pasted image 20241227183415.png]]

**3. 根据α1,i'抽取出序列中重要的信息，即根据权重系数对Value进行加权求和**
![[Pasted image 20241226200414.png]]
**将αi左乘Wv，可以得到vi，vi与α1,i'相乘，再将每一个乘积相加得到b1，vi越大，越接近抽取出来的结果b1**

### 3.3 矩阵形式
$$
Q = W^q * I
$$
$$
K = W^k * I
$$
$$
V = W^v * I
$$
$$
Q = [q^1, q^2, q^3, q^4], 
K = [k^1, k^2, k^3, k^4],
$$
$$
V = [v^1, v^2, v^3, v^4],
I = [\alpha^1, \alpha^2, \alpha^3, \alpha^4] 
$$
$$
\begin{bmatrix}  \alpha_{1,1}&  \alpha_{2,1}&  \alpha_{3,1}&\alpha_{4,1}  \\  \alpha_{1,2}&  \alpha_{2,2}&  \alpha_{3,2}&\alpha_{4,2} \\  \alpha_{1,3}&  \alpha_{2,3}&  \alpha_{3,3}&\alpha_{4,3} \\  \alpha_{1,4}&  \alpha_{2,4}&  \alpha_{3,4}&\alpha_{4,4}\end{bmatrix}=\begin{bmatrix}k_1^T \\ k_2^T\\ k_3^T\\k_4^T\end{bmatrix}*\begin{bmatrix}  q_1&  q_2& q_3&q_4\end{bmatrix}
$$
$$
A = K^T*Q
$$
$$
A\to A'
$$
$$
O = V*A'
$$
**Wq,Wk,Wv需要通过学习找出来**
$$
{\color{Red} 如果Q = I * W^q；K = I * W^k；V = I * W^v，则}
$$
![[Pasted image 20241227185331.png]]
```python
def attention(query, key, value, mask=None, dropout=None):     """     实现注意力机制     """
	# query，key,value：注意力的三个输入张量
	# mask：掩码张量
	# dropout：传入的Dropout实例化对象
	# 获取query张量的最后一维大小，即查询向量的嵌入维度d_k
	d_k = query.size(-1)  
	# 计算相似度分数
	# 将key的最后两维进行转置，使其从(batch_size, seq_len_k, d_k)变为(batch_size, d_k, seq_len_k)
	# matmul即作两个张量矩阵相乘  
	scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)   
	# 应用掩码     
	if mask is not None: 
	# 将 scores 中对应 mask == 0 的位置填充为 `-1e9`（一个非常小的值，接近负无穷）
	# 这是因为 softmax 的输出会接近 0，对于负无穷值几乎不会产生注意力。        
		scores = scores.masked_fill(mask == 0, -1e9)
	# 计算注意力权重
	# 对scores的最后一维进行softmax操作        
	p_attn = torch.softmax(scores, dim=-1)  
	# 应用 Dropout
	# Dropout 是一种正则化方法，用于随机将部分注意力权重置零，避免模型过拟合。     
	if dropout is not None:         
		p_attn = dropout(p_attn) 
	# 加权求和得到输出       
	output = torch.matmul(p_attn, value) 
	# output：最终的注意力加权输出，形状为(batch_size, seq_len_q, d_v)`     
	# p_attn：注意力权重，形状为 (batch_size, seq_len_q, seq_len_k)，表示每个查询对每个键的注意力分布。
	return output, p_attn
```
### 3.4 Multi-head Self-attention
![[Pasted image 20241226210040.png]]
$$
{\color{Red} 但是注意到并没有考虑位置，因此引入e^i，用e^i+\alpha^i表示位置} 
$$

### 3.5 将self-attention应用于Graph
![[Pasted image 20241226212113.png]]
只需要计算有edge相连接的节点之间的相关系数

## 进一步的学习

- Long Range Arena: A Benchmark for Efficient Transformers https://arxiv.org/abs/2011.04006
- Efficient Transformers: A Survey https://arxiv.org/abs/2009.06732

