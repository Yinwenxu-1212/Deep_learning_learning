---
date: 2025-01-04
tags:
  - transformer
  - attention
  - deep-learning
---
# 1. Sequence to sequence(Seq2seq)

![[Pasted image 20250104165733.png]]
# 2. Encoder

	输入一排向量，输出另外一排向量

![[Pasted image 20250104170028.png]]
## transformer的Encoder中的Block
![[Pasted image 20250104171248.png]]
## 2.1 Embedding层
	文本嵌入层，用一个低维稠密的向量表示一个对象，通常有One-Hot Encoding、Word Embedding。
	作用是将文本中词汇的数字表示转变为向量表示，在高维空间中捕捉词汇间的关系。
1. One-Hot Encoding
	用 **One-Hot** 形式编码成序列向量。向量长度是预定义的词汇表中拥有的单词量，向量在这一维中的值只有一个位置是1，其余都是0，1对应的位置就是词汇表中表示这个单词的地方。
2. Word Embedding
	设计一个可学习的权重矩阵 W，将词向量与这个矩阵进行点乘，即得到新的表示结果。
```python
class Embeddings(nn.Module):
	# vocab代表词汇表中的单词量（词表的大小），one-hot 编码后词向量的长度就是这个值
	# d_model代表权重矩阵的列数（词嵌入的维数），通常为512，就是要将词向量的维度从vocab编码到d_model
	# 维度（dimension）表示张量在特定方向上的大小，而这一方向上的数据通常可以看作不同的属性或特征。
    def __init__(self, d_model, vocab):
	    # 继承nn.Module的初始化函数
        super(Embeddings, self).__init__()
		# 调用nn中预定义层Embedding，获得一个词嵌入对象self.lut
		# lut 就是嵌入矩阵（lookup table），它是模型从索引到嵌入向量的映射。
        self.lut = nn.Embedding(vocab, d_model)
        # 将d.model传入类中
        self.d_model = d_model

    def forward(self, x):
	    # 前向传播
	    # x：代表输入进模型的文本通过词汇映射后的数字张量
	    # 调用 lut(x) 时，x 是输入的索引张量。lut(x) 会根据 x 中的每个索引，从嵌入矩阵中提取对应的向量
	    # 将x传给self.lut并与根号下self.d_model相乘作为结果返回
        return self.lut(x) * math.sqrt(self.d_model)

# input 是一个形状为 (batch_size=2, seq_len=4) 的二维张量
d_model = 512
vocab = 1000
x = Variable(torch.LongTensor([1,2,4,5],[4,3,2,9]))
emb = Embeddings(d_model, vocab)
embr = emb(x)
```
## 2.2 Position Encoding

	在transformer的编码器结构中，并没有针对词汇位置信息的处理，因此需要在Embedding层后加入位置编码器。
	做法是预定义一个函数，通过函数计算出位置信息。pos代表的是词在句子中的位置，d是词向量的维度（通常经过 word embedding 后是512），2i代表的是d中的偶数维度，2i + 1则代表的是奇数维度，这种计算方式使得每一维都对应一个正弦曲线。

$$
\begin{gathered}
P E_{(p o s, 2 i)}=\sin \left(p o s / 10000^{2 i / d}\right) \\
P E_{(p o s, 2 i+1)}=\cos \left(\text { pos } / 10000^{2 i / d}\right)
\end{gathered}
$$
![[Pasted image 20250105174109.png]]
	由公式可知，**每一维 i 都对应不同周期的正余弦曲线**： i=0 时是周期为 2π 的 sin 函数， i=1 时是周期为 2π 的cos 函数······对于不同的两个位置 pos1 和 pos2 ，若它们在某一维 i 上有相同的编码值，则说明这两个位置的差值等于该维所在曲线的周期，即 |pos1−pos2|=Ti 。而对于另一个维度 j(j≠i) ，由于 Tj≠Ti ，因此 pos1 和 pos2 在这个维度 j 上的编码值就不会相等，对于其它任意 k∈{0,1,2,..,d−1};k≠i 也是如此。
	综上可知，这种编码方式保证了**不同位置在所有维度上不会被编码到完全一样的值**，从而使每个位置都获得独一无二的编码。
```python
class PositionalEncoding(nn.Module):
	# d_model:词嵌入的维度
	# dropout：置0比率，防止过拟合的策略
	# max_len:每个句子的最大长度
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
		# 实例化nn中预定义的Dropout层，并将dropout传入其中，获得对象self.dropout
		# p表示置零的概率,用于控制Dropout的强度（神经元失效的比例）。
        self.dropout = nn.Dropout(p=dropout) 
        # 初始化一个位置编码矩阵，是一个零阵 
        # 每一个单词都代表一个行，d_model是词向量的维数
        pe = torch.zeros(max_len, d_model)  
        # 初始化绝对位置矩阵
        # 生成位置索引 [0, 1, 2, ..., max_len-1]
        # 因为参数传的是1，即向量是一个max_len x 1的矩阵
        position = torch.arange(0, max_len).unsqueeze(1)
        # 即计算10000^(-2i/d)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
  -(math.log(10000.0) / d_model))  # d_model即公式中的d
		# 偶数维度
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数维度
        pe[:, 1::2] = torch.cos(position * div_term)
        # 得到了位置编码矩阵pe，但只是一个二维的矩阵
        # 在 Transformer 中，输入通常是一个三维张量，形状为(batch_size, seq_len, d_model)
        # batch_size: 批量大小。
		# seq_len: 序列长度（可以理解为行数）。
		# d_model: 嵌入维度（可以理解为列数）。
        # 位置编码的形状需要与输入张量匹配，因此需要通过 unsqueeze(0) 将位置编码的形状从 (max_len, d_model) 扩展为 (1, max_len, d_model)。
        pe = pe.unsqueeze(0)
        # 将pe注册为模块的 buffer，表示它是模型的一部分，但不需要更新梯度
        # 注册为buffer后，可以在模型保存后，重新加载时，将这个位置编码器和模型参数加载进来
        self.register_buffer('pe', pe)

    def forward(self, x):
	    # x：是文本序列的词嵌入张量，形状为 (batch_size, seq_len, d_model)，是Embedding层的输出
	    # pe的编码太长了，将第二个维度（max_len）缩小为x句子同等的长度，即从self.pe中提取与输入序列长度（max_len）相匹配的部分
	    # x.size(1): 返回输入序列的长度
	    # 原向量加上计算出的位置信息才是最终的embedding
        x = x + Variable(self.pe[:, :x.size(1)],
					    requires_grad=False)
		# 运用self.dropout对象进行‘丢弃’操作，并返回结果
        return self.dropout(x)
        
# 加入超参数        
d_model = 512
dropout = 0.1
max_len = 60
x = embr
pe = PositionalEncoding(d_model, dropout, max_len)
pe_result = pe(x)
print("pe_result:", pre_result)

# 可视化词汇向量中特征的分布曲线
# 实例化 PositionalEncoding 类对象 
pe = PositionalEncoding(d_model=20, dropout=0) 
# 输入全零张量 (batch_size=1, seq_len=100, d_model=20) 
x = torch.zeros(1, 100, 20) 
y = pe(x) 
# 输出位置编码，形状为 (1, 100, 20) 
# 可视化维度 4 到 7 的位置编码 
plt.figure(figsize=(15, 5)) 
positions = np.arange(100) 
# 序列位置 
for dim in range(4, 8): 
	# 维度 4 到 7
	# x轴用position显示位置索引
	# y轴显示位置编码值 
	plt.plot(positions, y[0, :, dim].detach().numpy(),label=f"dim {dim}") 
# 添加图例和标题 
plt.legend() 
plt.title("Visualization of Positional Encoding") plt.xlabel("Position in Sequence") 
plt.ylabel("Encoding Value") 
plt.show()
```
## 2.3 多头注意力机制

单头注意力机制看这篇文章[[Self-attention]]
	多头自注意力就是“多个头的自注意力”，通常“头数”（通俗理解为个数）设为8，这样就有8个自注意力的输出，最后将它们拼接起来再经过一个线性变换即可。可以让每个注意力机制去优化每个词汇的不同特征部分，从而均衡同一种注意力机制可能产生的偏差，让词义拥有来自更多元的表达，提升模型的效果。
#### **多头注意力机制的核心思想**

1. **输入数据**
    
    - 输入由查询向量（Query, Q）、键向量（Key, K）和值向量（Value, V）组成。
    - 每个位置都有对应的 Q、K 和 V。
2. **多头拆分**
    
    - 将嵌入维度（`embedding_dim`）拆分成 `head` 个部分，每部分称为一个注意力头（Head）。
    - 每个头独立计算注意力机制。
3. **并行注意力计算**
    
    - 每个头分别计算查询和键之间的相似度（点积）。
    - 使用 softmax 归一化分数，得到注意力权重。
    - 用注意力权重对值向量加权求和，得到每个头的输出。
4. **结果合并**
    
    - 将每个头的输出拼接，恢复到原始嵌入维度。
    - 通过一个线性层对结果进行变换，生成最终输出。
![[Pasted image 20250106162518.png]]
```python
import copy
# 实现克隆函数，因为在多头注意力机制下，需要用到多个结构相同的线性层
# 需要使用clone函数将他们一同初始化到一个网络层列表对象中
def clones(module, N):
	'''创建目标网络层的深拷贝，确保每个拷贝是独立的。'''
	# module：代表要克隆的目标网络层
	# N：将module克隆N个
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 实现多头注意力机制的类
class MultiHeadedAttention(nn.Module):
	def __init__(self, head, embedding_dim, dropout=0.1):
		# head：代表几个头的参数
		# embedding_dim：代表词嵌入的维度
		# dropout：进行Dropout操作时，置零的比率
		super(MultiHeadAttention, self).__init__()

	# 判断多头的数量head需要整除词嵌入的维度embedding_dim
	assert embedding_dim % head == 0
	# 得到每个头获得的词向量的维度
	self.d_k = embedding_dim//head
	self.head = head
	self.embedding_dim = embedding_dim
	# 获得线性层，要获得4个(如上图所示），分别是Q,K,V以及最终的输出线性层
	self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
	# 初始化注意力向量
	self.attn = None
	# 初始化dropout对象
	self.dropout = nn.Dropout(p=dropout)

	def forward(self, query, key, value, mask=None):
		# query,key,value是注意力机制的三个输入张量，mask代表掩码张量
		# 判断是否使用掩码张量
		if mask is not None:
			# 使用squeeze将掩码张量进行维度扩充，代表多头中的第n个头
			mask = mask.unsqueeze(1)

		# 得到batch_size，代表有多少个样本
		batch_size = query.size(0)

		# 首先使用zip将网络层和输入数据连接在一起，运用for循环，将QKV分别传入线性层中
		# 模型的输出利用view和transpose进行维度和形状的改变
		# .view()将线性变换的结果拆分为head个头，每个头的嵌入维度为d_k
		# batch_size: 保留批量大小的维度
		# -1: 表示序列长度（seq_len），自动推导
		# .transpose(1,2):将seq_len和head的维度交换，便于多头注意力机制的计算，即让序列长度与词向量的维度相邻，便于找到词义与序列位置的关系
		query, key, value = \
			[model(x).view(batch_size, -1, self.head, self.d_k).transpose(1,2)  
			for model,x in zip(self.linears, (query,key,value))]
		# 将每个头的输出传入到注意力层，计算每个头的点积注意力
		# x: 注意力加权后的输出，形状为(batch_size, head, seq_len, d_k)
		# self.attn: 注意力权重，形状为(batch_size, head, seq_len, seq_len)
		x, self.attn = attention(query, key, value, mask = mask, dropout=self.dropout)
		# 恢复原始形状
		# .contiguous()保证张量在内存中的存储是连续的
		# view的作用将多头的结果拼接起来，恢复到原始嵌入维度
		x = x.transpose(1,2).contiguous().view(batch_size,-1,self.head*self.d_k)
		# 最后将x输入线性层列表中的最后一个线性层中进行处理，得到最终的多头注意力结构输出
		return self.linears[-1](x)

# 实例化参数
head = 8
embedding_dim = 512
dropout = 0.2

# 输入参数的初始化
query = key = value = pe_result
mask = torch.zeros(8,4,4)
mha = MultiHeadedAttention(head, embedding_dim, dropout)
mha_result = mha(query, key, value, mask)
```
## 2.4 Add & Norm

	即residual + Layer Norm，归一化操作和残差连接

1. 将输入a与经过Self-attention Model的输出b相加
2. 将a + b 进行Layer Norm，即将xi减去均值m，再除以标准差
3. 将通过标准化的a + b与其经过FC（全连接神经网络）的结果相加
4. 再通过一次Layer Norm得到最终的输出
![[Pasted image 20250104170931.png]]

$$
\operatorname{LayerNorm}(x)=\frac{x-\mu}{\sigma+\epsilon} \cdot a+b
$$
- $\mu$: 均值。
- $\sigma$: 标准差。
- $\epsilon$: 一个小数值，防止分母为零。
- $a$ 和 $b$: 可学习的缩放参数（`a2` 和 `b2`）。
```python
# 规范化层（Layer Norm）
class LayerNorm(nn.Module):
	def __init__(self, features, eps=1e-6):
		# features：词嵌入的维度
		# eps：足够小的正数，在规范化公示的分母中出现，防止分母为零
		super(LayerNorm, self).__init__()

		# 初始化两个参数张量
		# a2初始化为全1，表示不改变规范化后的缩放比例
		# b2初始化为全0，表示不改变规范化后的偏移量
		# 用Parameter进行封装，代表也是模型中的参数
		self.a2 = nn.Parameter(torch.ones(features))
		self.b2 = nn.Parameter(torch.zeros(features))

		# 把eps传入类中
		self.eps = eps

	def forward(self, x):
		# 对输入x的最后一个维度计算均值和标准差
		# keepdim=True确保均值的形状与x保持一致，便于后续操作。
		mean = x.mean(-1, keepdim = True)
		std = x.std(-1, keepdim = True)
		# *代表点乘
		return self.a2*(x - mean)/(std+self.eps) + self.b2

# 实例化参数
features = d_model = 512
eps = 1e-6

x = ff_result
ln = LayerNorm(features, eps)
```
![[Pasted image 20250107135848.png]]
```python
# 子层连接结构（残差连接）
# 使用SublayerConnection来实现子层连接结构的类
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        """它输入参数有两个, size以及dropout"""
        # size：词嵌入维度的大小
        # dropout：对模型结构中的节点数进行随机抑制的比率，又因为节点被抑制等效就是该节点的输出都是0，因此也可以把dropout看作是对输出矩阵的随机置0的比率
        super(SublayerConnection, self).__init__()
        # 实例化了规范化对象self.norm
        self.norm = LayerNorm(size)
        # 又使用nn中预定义的droupout实例化一个self.dropout对象.
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        """前向逻辑函数中, 接收上一个层或者子层的输入作为第一个参数，
           将该子层连接中的子层函数作为第二个参数"""
        # x：代表上一层传入的张量
        # sublayer：子层的具体操作（如多头注意力机制或前馈网络）
        # 首先对输出进行规范化，然后将结果传给子层处理，之后再对子层进行dropout操作，
        # 随机停止一些网络中神经元的作用，来防止过拟合. 最后还有一个add操作，
        # 因为存在跳跃连接，所以是将输入x与dropout后的子层输出结果相加作为最终的子层连接输出
        return x + self.dropout(sublayer(self.norm(x)))

# 实例化参数
size = d_model = 512
head = 8
dropout = 0.2

x = pe_result
mask = torch.zeros(8,4,4)
self.attn = MultiHeadedAttention(head, d_model)

# 输入的 `query`, `key`, `value` 都是同一个张量 `x`，表示自注意力机制
# 定义了一个匿名函数（lambda 函数），表示多头自注意力机制的计算逻辑。
sublayer = lambda x:self_attn(x, x, x, mask) 

sc = SublayerConnection(size, dropout)
sc_result = sc(x, sublayer)
```
## 2.5 Feed Forward Network（FFN）前馈神经网络

	实质上就是两个全连接层，并且其中一个带ReLU激活，两层中间有Dropout。输入维度和输出维度相同，便于与残差连接（Residual Connection）对接。
	作用：考虑到注意力机制可能对复杂过程的拟合程度不够，增加两层网络来增强模型的能力
	
$$
\operatorname{FFN}(x)=\max \left(0, x W_1+b_1\right) W_2+b_2
$$
可以参考[[ATTENTION, LEARN TO SOLVE ROUTING PROBLEMS!]]

```python
# 通过PositionwiseFeedForward来实现前馈全连接层、
class PositionwiseFeedForward(nn.Module):
	def __init__(self, d_model, d_ff, dropout=0.1):
		# d_model：词嵌入的维度，同时也是第一个线性层的输入维度和第二个线性层的输出维度
		# d_ff：线性变换维度，第一个线性层的输出维度，也是第二个线性层的输入维度
		super(PositionwiseFeedForward, self).__init__()
		# 定义两层全连接的线性层
		# 第一个线性层，输入维度是d_model，输出维度是d_ff
		self.w1 = nn.Linear(d_model, d_ff)
		# 第二个线性层，输入维度是d_ff，输出维度是d_model
		self.w2 = nn.Linear(d_ff, d_model)
		# 实例化dropout
		self.dropout = nn.Dropout(p=dropout)
		
	def forward(self, x):
		''' 输入参数为x，代表来自上一层的输出（通常是多头注意力模块的结果）'''
		# 首先经过第一个线性层，然后运用Functional中的relu函数进行激活
		# 再使用dropout随机置零
		# 最后通过第二个线性层w2，返回最终的结果
		return self.w2(self.dropout(F.relu(self.w1(x))))

# 实例化参数
d_model = 512
d_ff = 64
dropout = 0.2

x = mha_result
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
ff_result = ff(x)
```

## 2.6 编码器层

	作为编码器的组成单元，每个编码器层完成一次对输入的特征提取过程

![[Pasted image 20250107151721.png]]

```python
# 使用EncoderLayer类实现编码器层
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
		# size：词嵌入维度的大小
		# self_attn：传入多头自注意力子层实例化对象,
		# feed_froward：传入前馈全连接层实例化对象
		# dropout：置0比率
        super(EncoderLayer, self).__init__()

        # 将self_attn和feed_forward传入其中.
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        # 编码器层中有两个子层连接结构, 所以使用clones函数进行克隆
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # 把size传入其中
        self.size = size

    def forward(self, x, mask):
        # x：代表上一层的传入张量
		# mask：代表掩码张量
		# 首先通过第一个子层连接结构，其中包含多头自注意力子层
		# 然后通过第二个子层连接结构，其中包含前馈全连接子层
		# 最后返回结果
		# 通过 `lambda` 封装，可以将动态参数（如 `query`, `key`, `value` 和 `mask`）固定，而只需要传入x
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

# 实例化参数
size = d_model = 512
head = 8
d_ff = 64
x = pe_result
dropout = 0.2

self_attn = MultiHeadedAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
mask = torch.zeros(8,4,4)

el = EncoderLayer(size, self_attn, ff, dropout)
el_result = el(x, mask)
```

## 2.7 编码器

	用于对输入进行指定的特征提取过程，由N个编码器层堆叠形成
	编码器的输出就是Transformer中编码器的特征提取表示，成为解码器输入的一部分

```python
# 使用Encoder类来实现编码器
class Encoder(nn.Module):
	def __init__(self, layer, N):
        """初始化函数的两个参数分别代表编码器层和编码器层的个数"""
        super(Encoder, self).__init__()
        # 首先使用clones函数克隆N个编码器层放在self.layers中
        self.layers = clones(layer, N)
        # 再初始化一个规范化层, 它将用在编码器的最后面.
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """forward函数的输入和编码器层相同, x代表上一层的输出, mask代表掩码张量"""
        # 首先就是对我们克隆的编码器层进行循环，每次都会得到一个新的x，
        # 这个循环的过程，就相当于输出的x经过了N个编码器层的处理.
        # 最后再通过规范化层的对象self.norm进行处理，最后返回结果.
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# 实例化参数
size = 512
head = 8
d_model = 512
d_ff = 64
# 每个编码器层的权重应当是独立的，否则模型的表达能力会受到限制，因此使用深度拷贝
c = copy.deepcopy
attn = MultiHeadedAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
dropout = 0.2
layer = EncoderLayer(size, c(attn), c(ff), dropout)

# 编码器中的编码器层的数量N
N = 8
mask = torch.zeros(8,4,4)

en = Encoder(layer, N)
en_result = en(x, mask)
```
# 3. Decoder

	实质是产生输出

![[Pasted image 20250105113649.png]]
## 3.1 Autoregressive（AT）

![[Pasted image 20250105114218.png]]
	Decoder的输出会作为下一阶段的输入
	Encoder-Decoder对比
![[Pasted image 20250105115726.png]]

	由N个解码器层堆叠⽽成
	每个解码器层由三个⼦层连接结构组成
	第⼀个⼦层连接结构包括⼀个多头⾃注意⼒⼦层和规范化层以及⼀个残差连接
	第⼆个⼦层连接结构包括⼀个多头注意⼒⼦层和规范化层以及⼀个残差连接
	第三个⼦层连接结构包括⼀个前馈全连接⼦层和规范化层以及⼀个残差连接
### 3.1.1 Masked Muti-Head Attention

	掩膜（mask）的使用，目的是忽略某些位置，不计算与其相关的注意力权重。因为 Decoder 是要对序列进行解码预测，所以你不能提前看到要预测的内容，你应当根据当前及之前已解码/预测的内容来推算即将预测的内容。于是，这个mask就是用来遮住后面将要预测的部分。
	
![[Pasted image 20250105141836.png]]
	计算b1时只考虑q1与k1之间的关系，计算b2时只考虑q2与k1，q2与k2的关系，计算b3时只考虑q3与k1，q3与k2，q3与k3之间的关系，计算b4时需要考虑q4与所有的k之间的关系。
```python
# 构建掩码张量的函数
# subsequent_mask的作用是生成向后遮掩（subsequent masking）的掩码张量，用于防止 Transformer 的解码器在生成第t个时间步时访问未来时间步（t+1及之后）的信息。
def subsequent_mask(size):
	""" 生成向后遮掩的掩码张量，参数size是掩码张良最后两个维度的大小，它的最后两维形成一个方阵"""
	# 在函数中，首先定义掩码张量的形状
	# size:解码器序列的长度，形成一个方阵
	attn_shape = (1, size, size)

	# np.ones生成一个全为1的张量，形状为(1, size, size)
	# np.triu(..., k=1):提取上三角部分，`k=1` 表示从对角线上的第 1 个元素开始保留上三角(不包含主对角线上的元素)，其他位置置为 0
	# 为了节省时间，将其中的数据类型变为无符号8位整型unit8
	subsequent_mask = np.triu(np.ones(attn_shape), k=1).astyle('uint8')
	# 将numpy类型转化为torch中的tensor，内部做一个1-的操作
	# 实际是做了三角阵的反转，subsequent_mask中的每个元素都会被1减
	# 如果是0，subsequent_mask中的该位置由0变成1
	# 如果是1，subsequent_mask中的该位置由1变为0
	# 1代表被遮掩，0代表没有被遮掩（可以自己定义）
	return torch.from_numpy(1-subsequent_mask)
```
### 3.1.2 Cross Attention

	计算逻辑与encoder中一致，不过K、V矩阵使用 Encoder 的编码信息矩阵C进行计算，而Q：如果是 Decoder 的第一层，则使用（已解码的）输入序列（最开始则是起始字符）；而对于后面的层，则是前面层的输出。

Decoder第一层运作过程
![[Pasted image 20250105150714.png]]
Decoder第二层及以后的运作过程
![[Pasted image 20250105150917.png]]
### 3.1.3 Feed Forward

	与encoder一致

### 3.1.4 Output Generator

	实质就是个线性层，将解码的序列映射回原来的空间维度，然后经过softmax（或log-softmax）生成预测概率。

## 3.2 Non-autoregressive (NAT)

![[Pasted image 20250105144529.png]]
	NAT决定输出seq长度的方法：
- 另一个predictor来预测出输出的长度
- 输出很长的序列，并且忽略END之后的tokens（词元）
## 3.3 解码器层

	是解码器的组成单元，每个解码器层根据给定的输入向目标方向进行特征提取操作。

![[Pasted image 20250107162440.png]]
```python
# 使用DecoderLayer的类实现解码器层
class DecoderLayer(nn.Module):
	def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
		# size：词嵌入的维度
		# self_attn：多头自注意力对象
		# src_attn：常规的多头注意力对象(Q!=K=V)
		# feed_forward：前馈全连接层对象
		# dropout：置零比率
        super(DecoderLayer, self).__init__()
        # 在初始化函数中， 主要就是将这些输入传到类中
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward

        # 用clones函数克隆三个子层连接对象.
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
		# memory：来自编码器层的语义存储张量
		# source_mask：源数据掩码张量
		# target_mask：目标数据掩码张量
		
        # 将memory表示成m方便之后使用
        m = memory

		# 多头自注意力子层（Self-Attention）:
		# 负责捕捉目标序列中不同位置之间的关系。
        # 将x传入第一个子层结构，第一个子层结构的输入分别是x和self-attn函数，因为是自注意力机制，所以Q,K,V都是x，
        # 最后一个参数是目标数据掩码张量，这时要对目标数据进行遮掩，因为此时模型可能还没有生成任何目标数据。使用目标掩码（`target_mask`）屏蔽未来位置，确保生成序列的因果性。
        # 比如在解码器准备生成第一个字符或词汇时，我们其实已经传入了第一个字符以便计算损失，
        # 但是我们不希望在生成第一个字符时模型能利用这个信息，因此我们会将其遮掩，同样生成第二个字符或词汇时，
        # 模型只能使用第一个字符或词汇信息，第二个字符以及之后的信息都不允许被模型使用.
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))

		# 交互注意力子层（Source-Attention）:
		# 负责目标序列和源序列之间的信息交互。
		# 从源序列的编码器输出中提取有意义的信息，用于生成目标序列。
        # 进入第二个子层，这个子层中常规的注意力机制，q是输入x; k，v是编码层输出memory，
        # 同样也传入source_mask，但是进行源数据遮掩的原因并非是抑制信息泄漏，而是遮蔽掉对结果没有意义的字符而产生的注意力值，
        # 以此提升模型效果和训练速度. 这样就完成了第二个子层的处理.
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))

        # 前馈全连接子层（Feed Forward）:
        # 提供非线性特征提取，增强特征表达能力
        return self.sublayer[2](x, self.feed_forward)

# 调用验证
# 类的实例化参数与解码器层类似, 相比多出了src_attn, 但是和self_attn是同一个类.
head = 8
size = 512
d_model = 512
d_ff = 64
dropout = 0.2
# 这里简化让src_attn与self_attn相同
self_attn = src_attn = MultiHeadedAttention(head, d_model, dropout)

# 前馈全连接层也和之前相同
ff = PositionwiseFeedForward(d_model, d_ff, dropout)

# x是来自目标数据的词嵌入表示, 但形式和源数据的词嵌入表示相同, 这里使用pe_result充当.
x = pe_result

# memory是来自编码器的输出
memory = en_result

# 实际中source_mask和target_mask并不相同, 这里为了方便计算使他们都为mask
mask = Variable(torch.zeros(8, 4, 4))
source_mask = target_mask = mask

dl = DecoderLayer(size, self_attn, src_attn, ff, dropout)
dl_result = dl(x, memory, source_mask, target_mask)
```

## 3.4 解码器

	根据编码器的结果以及上⼀次预测的结果, 对下⼀次可能出现的'值'进⾏特征表示

```python
# 使用类Decoder来实现解码器
class Decoder(nn.Module):
	def __init__(self, layer, N):
		# layer：解码器层
		# N：解码器层的个数
        super(Decoder, self).__init__()

        # 首先使用clones方法克隆了N个layer，然后实例化了一个规范化层.
        # 因为数据走过了所有的解码器层后最后要做规范化处理.
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
		# x：目标数据的嵌入表示
		# memory：编码器的输出
		# source_mask：源数据的掩码张量
		# target_mask：目标数据的掩码张量
		
        # 对每个层进行循环，变量x通过每一个层的处理
        # 得出最后的结果，再进行一次规范化返回即可.
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)

# 调用验证
# 分别是解码器层layer和解码器层的个数N
size = 512
d_model = 512
head = 8
d_ff = 64
dropout = 0.2
c = copy.deepcopy
attn = MultiHeadedAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
N = 8

# 输入参数与解码器层的输入参数相同
x = pe_result
memory = en_result
mask = Variable(torch.zeros(8, 4, 4))
source_mask = target_mask = mask

de = Decoder(layer, N)
de_result = de(x, memory, source_mask, target_mask)
```

## 3.5 输出生成器

![[Pasted image 20250107164340.png]]

	线性层：通过对上⼀步的线性变化得到指定维度的输出, 也就是转换维度的作⽤.
	softmax层：使最后⼀维的向量中的数字缩放到0-1的概率值域内, 并满⾜他们的和为1.

```python
# nn.functional工具包装载了网络层中那些只进行计算, 而没有参数的层
import torch.nn.functional as F

# 将线性层和softmax计算层一起实现, 因为二者的共同目标是生成最后的结构
# 因此把类的名字叫做Generator, 生成器类

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
	    # d_model：词嵌入维度
	    # vocab_size：词表大小
        super(Generator, self).__init__()

        # 线性层（nn.Linear），负责将输入张量从d_model映射到 vocab_size
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x：是上一层（通常是解码器输出）的张量，形状为 (batch_size, seq_len, d_model)
        # 首先使用上一步得到的self.project对x进行线性变化,
        # 再对最后一维（dim=-1）进行归一化，生成词汇表的概率分布
        # 输出是对数形式的概率分布，用于与交叉熵损失函数（`CrossEntropyLoss`）配合使用
        return F.log_softmax(self.project(x), dim=-1)

# 调用验证
# 词嵌入维度是512维
d_model = 512
# 词表大小是1000
vocab_size = 1000

# 输入x是上一层网络的输出, 我们使用来自解码器层的输出
x = de_result

gen = Generator(d_model, vocab_size)
gen_result = gen(x)
```
# 4. Model 
## 4.1 编码器-解码器结构

```python
# 使用EncoderDecoder类来实现编码器-解码器结构
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
		# encoder：编码器对象
		# decoder：解码器对象
		# source_embed：源数据嵌入函数
		# target_embed：目标数据嵌入函数
		# generator：输出部分的类别生成器对象
        super(EncoderDecoder, self).__init__()

        # 将参数传入到类中
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):

        # source：源数据
        # target：目标数据
        # source_mask：源数据掩码张量
        # target_mask：目标数据掩码张量
        # 在函数中, 将source, source_mask传入编码函数, 得到结果后,
        # 与source_mask，target，和target_mask一同传给解码函数.
        return self.decode(self.encode(source, source_mask), source_mask,target, target_mask)

    def encode(self, source, source_mask):

        """编码函数, 以source和source_mask为参数"""
        # 使用src_embed对source做处理, 然后和source_mask一起传给self.encoder
        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
    
        """解码函数, 以memory即编码器的输出, source_mask, target, target_mask为参数"""
        # 使用tgt_embed对target做处理, 然后和source_mask, target_mask, memory一起传给self.decoder
        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)

# 调用验证
vocab_size = 1000
d_model = 512
encoder = en
decoder = de
source_embed = nn.Embedding(vocab_size, d_model)
target_embed = nn.Embedding(vocab_size, d_model)
generator = gen

# 假设源数据与目标数据相同, 实际中并不相同
source = target = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
# 假设src_mask与tgt_mask相同，实际中并不相同
source_mask = target_mask = Variable(torch.zeros(8, 4, 4))
```
## 4.2 Transformer模型构建过程

```python
def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=0.1):
    """该函数用来构建模型"""
    # source_vocab：源数据特征(词汇)总数
    # target_vocab：目标数据特征(词汇)总数
    # N：编码器和解码器堆叠数
    # d_model：词向量映射维度
    # d_ff：前馈全连接网络中变换矩阵的维度
    # head：多头注意力结构中的多头数
    # dropout：置零比率
    
    # 首先得到一个深度拷贝命令，接下来很多结构都需要进行深度拷贝，
    # 来保证他们彼此之间相互独立，不受干扰.
    c = copy.deepcopy

    # 实例化了多头注意力类，得到对象attn
    attn = MultiHeadedAttention(head, d_model)

    # 实例化前馈全连接类，得到对象ff
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    # 实例化位置编码类，得到对象position
    position = PositionalEncoding(d_model, dropout)

    # 根据结构图, 最外层是EncoderDecoder，在EncoderDecoder中，
    # 分别是编码器层，解码器层，源数据Embedding层和位置编码组成的有序结构，
    # 目标数据Embedding层和位置编码组成的有序结构，以及类别生成器层.
    # 在编码器层中有两个子层，分别是attention子层以及前馈全连接子层，
    # 在解码器层中有三个子层，分别是两个attention子层以及前馈全连接层.
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab))

    # 初始化模型中的参数，比如线性层中的变换矩阵
    # 这里一但判断参数的维度大于1，则会将其初始化成一个服从均匀分布的矩阵，
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    # 最终返回一个构建好的模型对象
    return model

#  调用验证
source_vocab = 11
target_vocab = 11
N = 6
# 其他参数都使用默认值

if __name__ == '__main__':
    res = make_model(source_vocab, target_vocab, N)
    print(res)
```
# 5. Training

![[Pasted image 20250105152443.png]]
	ground truth：数据集的真实标签
	teacher forcing：将目标序列输入给解码器
# 6. Tips
## 6.1 Copy Mechanism

	直接复制输入的一部分作为输出
	pointer network
## 6.2 Guided Attention 

	对于一些输入输出需要单调的，可以应用guided attention
## 6.3 [Beam Search]([十分钟读懂Beam Search 1：基础 - 知乎](https://zhuanlan.zhihu.com/p/114669778))

	需要有创造力的任务，一般不能用Beam Search

![[Pasted image 20250105154753.png]]
## 6.4 Scheduled Sampling
