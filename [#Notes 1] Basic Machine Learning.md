# [#Notes 1]: Basic Machine Learning

## 损失函数

### 0-1损失函数

> 不连续且导数为0，常用连续可微的损失函数替代

### 平方损失函数

$$
\cal L(y,f(x;\theta)) = \frac{1}{2}(y-f(x;\theta))^2
$$

> 不适用于分类问题：
>

### 交叉熵损失函数

用于衡量两个概率分布的差异

对于多分类问题，设标签的真实分布为$\pmb{y}$,模型预测分布为$f(\pmb{x};\theta)$之间的交叉熵为：

$\cal {L}(\pmb{y},f(\pmb {x};\theta)) = -\pmb{y}^{T}~log~f(\pmb{x};\theta)=-\sum_{c=1}^{C}y_clog~f_c(\pmb{x};\theta)$



## Theorem

### PAC学习理论——可能近似正确学习理论

- 泛化错误：

  $\cal G_{\cal D} (f) = \cal R(f) - \cal R_{\cal D}^{emp}(f)$

  > 当训练集$|\cal D | $趋向无穷大，泛化误差趋向0
  
- PAC可学习

  > 算法能够在多项式时间内从合理数量的训练数据中学到近似争取的解$f(x)$

  - 近似正确：泛化误差小于一定界限
  - 可能：学习算法$\cal A$可能以$1-\delta$概率学习到近似正确的假设

  ![image-20201017111451632](C:\Users\Nector\AppData\Roaming\Typora\typora-user-images\image-20201017111451632.png)

  > - $\epsilon,\delta$与样本数量$N$和假设空间$\cal F$相关

  - 固定$\delta,\epsilon$可计算出需要的样本数量

![image-20201017111704949](C:\Users\Nector\AppData\Roaming\Typora\typora-user-images\image-20201017111704949.png)

### 归纳偏置——先验

指在机器学习算法中对学习问题做出的一些假设，这些假设成为归纳偏置



