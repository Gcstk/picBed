# 杂记

## 一、一些较好的解释性文章

### 1.1机器学习相关

#### 1.1.1PCA主成分分析

[【机器学习】降维——PCA（非常详细） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/77151308)

#### 1.1.2NER 命名实体识别

[命名实体识别（NER）综述 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/56085975)

#### 1.1.3准确率、精确率、召回率

[通俗解释机器学习中的召回率、精确率、准确率 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/93586831)

TP（True Positives）、TN（True Negatives）、FP（False Positives）和FN（False Negatives）是评估分类模型性能时常用的术语。

- TP（True Positives）：真正例，表示模型将正类样本正确地预测为正类。
- TN（True Negatives）：真负例，表示模型将负类样本正确地预测为负类。
- FP（False Positives）：假正例，表示模型将负类样本错误地预测为正类。
- FN（False Negatives）：假负例，表示模型将正类样本错误地预测为负类。

![img](https://pic4.zhimg.com/80/v2-9a2d6ead21593c692304b295ceb4ed9f_1440w.webp)

**F1值**：中和了精确率和召回率的指标
$$
F_1 = \frac{2 P R}{P+R}
$$
**Spearman相关系数**：Spearman相关系数衡量的是两个等级变量之间的统计依赖性，它主要关注的是变量之间的单调关系，而不是精确的值。Spearman相关系数检查的是当一个变量改变时，另一个变量是否也会在同一方向上改变。它用于衡量等级（顺序）之间的关系，而不关心这些变化的具体幅度。

## 1.2 神经网络相关

### 1.2.1 阈值相关问题

#### 1.2.1.1余弦相似度阈值的确定

[常见文本相似度计算方法简介 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/88938220)

[一文看懂机器学习指标：准确率、精准率、召回率、F1、ROC曲线、AUC曲线 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/93107394)











## 1.3 基础知识

### 1.3.1 熵与交叉熵

[一文搞懂熵(Entropy),交叉熵(Cross-Entropy) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/149186719)

[Evaluation Metrics for Language Modeling (thegradient.pub)](https://thegradient.pub/understanding-evaluation-metrics-for-language-models/)

> 熵的定义：无损编码事件信息的最小平均编码长度。
>
> H(P) = - Σ P(x) * log(P(x)) 

> 交叉熵的定义：H(P,Q) = - Σ P(x) * log(Q(x)) 
>
> 语言模型中，P通常是真实数据的分布，Q是模型的预测分布。因为根据熵的定义是最小的编码长度，交叉熵H(P,Q)必定大于熵H(P) 。因此当数据真实分布与模型概率预测分布一致时，即有H(P,Q)==H(P) ，

[交叉熵损失函数原理详解 - 简书 (jianshu.com)](https://www.jianshu.com/p/269ad3103c41)







#### 1.3.2





## 1.4 LLM

### 1.4.1 参考性文章

[【LLM】从零开始训练大模型 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/636270877)



### 1.4.2 STF



### 1.4.3 RM（reward model）



### 1.4.4 RLHF

参考资料：https://arxiv.org/abs/2203.02155

使用方法：



## 二、CoSENT

### 1.1 参考资料

[CoSENT（一）：比Sentence-BERT更有效的句向量方案 - 科学空间](https://kexue.fm/archives/8847)

[CoSENT（二）：特征式匹配与交互式匹配有多大差距？ - 科学空间](https://kexue.fm/archives/8860)

### 1.2









## 三、Triton

### 1.1 参考资料

[官方文档](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html)

[Triton Inference Server 原理入门之框架篇](https://www.bilibili.com/video/BV1KS4y1v7zd/?spm_id_from=333.337.search-card.all.click&vd_source=008ad44715922007849c860da387f908)



### 1.2 推理性能优化

> 方法一：使用dynamic_batchsize

> 方法二：使用onnx模型或者tensorRT模型并设置optimizer

