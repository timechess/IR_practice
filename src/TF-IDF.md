# TF-IDF

在本章中我们将主要介绍经典的TF-IDF算法。这与本书的核心内容之间相差较远，作为过渡章节，主要目的是让读者练习Python编程，为后续实践进行准备。

## 文本向量化

在上一章中我们强调，找到文本恰当的数学表示是信息检索的关键。我们已经知道，向量之间的相似度计算是容易的，对于 $x,y\in \mathbb{R}^{n}$，其余弦相似度可计算如下：

$$
sim(x, y)=\frac{x\cdot y}{\|x\|\cdot \|y\|}
$$

而文本的相似度如果能转换为向量的相似度，那么一切就变得顺理成章了。我们只需要找到一种合理的方法，在文本与向量之间建立映射，使向量能够尽可能地体现文本的特征。这一过程类似于机器学习中的**特征工程**，选取的特征很大程度上影响最后算法的效果。

**TF-IDF**（Term Frequency-Inverse Document Frequency）算法计算单词在文档库中的权重，权重越大说明该单词更能体现文档的独有特征，能够用于区分不同的文档。将文档中的词映射到其TF-IDF值后，我们自然就得到了文本的向量表示。暂且不讨论仅用单词的特征进行文本向量化的优劣，我们先介绍该算法的细节。

## TF

首先我们还是定义一些符号，用 $\mathcal{D}=\{d_i\}_{i=1}^{N}$ 表示文档库，$\{w_{i}\}_{i}^{M}$ 为出现过的所有单词，$n_{ij}$ 表示单词 $w_{i}$ 在文档 $d_{j}$ 中出现的次数。

TF即词频，为单词在文档中出现的频率。注意，每个单词在每个文档中的词频不同，这与接下来要说的IDF不同。单词 $w_{i}$ 在文档 $d_{j}$ 中的词频用数学公式表达即：

$$
TF_{ij} = \frac{n_{ij}}{\sum\limits_{k=1}^{M}n_{kj}}
$$

一般而言，词频越高的单词越能体现文档的特征，即高频词总是更重要。但是，仅凭借文档内的单词频率来判断这个单词在用于文档区分上的效果有如下几个问题：

- 文档中频率最高的词可能并不包含多少信息，比如中文中的“的”、“了”，英文中的"the"、"is"等。
- 如果语料库中每个文档的高频词都差不多，比如在小领域的专业文献库中，大量出现相同的专业术语，那么这样筛选出的高频词并不能很好地区分语料库中的文献。
- 对于长文本，每个单词的词频都极小，数值上差异不大，同时在计算机具体实现时可能出现下溢等问题。

因此，光靠词频来判断单词权重是远远不够的。为解决上述的部分问题，IDF即逆文档频率被引入。

## IDF

IDF源于一个简单的思想，即在一个文档中出现但很少在其他文档中出现的词更能体现该文档的特征。基于这个思想，导出IDF的公式就相对自然了。

$$
IDF_{i}=-\ln\frac{\sum\limits_{j=1}^{N}I(n_{ij}\ge 1)+1}{N}
$$

其中加入对数运算是考虑到数值稳定性，如果一个词只在一个文档中出现，那么其IDF值会极高。分子加一是为了防止出现 $\ln 0$。在这一计算公式下，一个单词在越多的文档中出现，其IDF值越小。

注意到与TF不同，每个单词在不同文档中的IDF值是相同的。当然，这都是在固定文档库的前提下。如果文档库变化，那么单词的IDF值也会变化。

最终我们计算出的TF-IDF值实际上是 $TF\times IDF$。

## Python实现

理解了公式之后，我们要做的就是把公式翻译成代码，让计算机能够高效地执行。我们首先来生成测试用的文档。得益于LLM，我们能够让LLM帮忙生成样例，而不需要手动构造。

```Python
documents = [
	"The quick brown fox jumps over the lazy dog at dawn",
	"Machine learning algorithms analyze data patterns effectively",
	"Sunset paints vibrant colors across the evening sky daily",
	"Fresh coffee beans are roasted and ground every morning",
	"Birds migrate thousands of miles during seasonal changes",
	"Digital transformation reshapes modern business operations completely",
	"Rainforests produce oxygen and absorb carbon dioxide naturally",
	"Athletes train rigorously for international competitions annually",
	"Quantum computing promises revolutionary breakthroughs in technology",
	"Ancient civilizations built remarkable architectural wonders worldwide",
	"Ocean currents influence global climate patterns significantly",
	"Vegetables contain essential vitamins for healthy body functions",
	"Renewable energy solutions combat climate change effectively",
	"Children develop language skills through interactive play activities",
	"Space exploration expands humanity's understanding of the universe",
	"Classical music enhances concentration and reduces stress levels",
	"Volcanic eruptions create fertile soil for agriculture over time",
	"Cybersecurity measures protect sensitive digital information proactively",
	"Penguins huddle together to survive Antarctic winter conditions",
	"Urban gardening promotes sustainable food production in cities"
]
```

对于一个严肃的自然语言处理项目而言，我们需要对以上文本进行标准化，包括但不限于大小写统一、词性标准化、去除停用词等。但由于本次实验仅作为样例，故我们只进行大小写的统一。

```Python
documents = [doc.lower() for doc in documents]
```

下面计算TF与IDF的函数留给读者作为练习，注意TF只需传入文档，而IDF需要传入整个文档库。

```python
def tf(doc, word):
    """Calculate the TF of a given word in the document.

    Args:
        doc: A string representing the document.
        word: The given word.
    Returns:
        The TF of the word in the document.
    """
    raise NotImplementedError

def idf(word, corpus):
    """Calculate the IDF of a given word in the corpus.

    Args:
        word: The given word.
        corpus: A list of documents(string).
    Returns:
        The IDF of the word in the corpus.
    """
    raise NotImplementedError
```

这两个函数都可以仅用一两行代码完成，初学者也可以轻松解决。如果是有C语言基础的读者，在饱受C中复杂的字符串操作折磨后，想必会很喜欢Python中的字符串。

```admonish hint
Life is short, I use Python!
```