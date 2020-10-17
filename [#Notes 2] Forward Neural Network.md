# [#Notes 2]: Feedforward Neural Network

## 神经元

![image-20201017113623944](C:\Users\Nector\AppData\Roaming\Typora\typora-user-images\image-20201017113623944.png)

### 神经元活性值：

$a = f(z)$

### 激活函数：

非线性函数$f(·)$,称为激活函数

<u>激活函数应满足的特点：</u>

- 连续可导$\to$可利用优化方法学习网络参数
- 激活函数和导数要简单$\to$提高网络运算效率
- 激活函数的导数值要在一个合适的区间内$\to$太大或太小混影响训练的效率和稳定性

#### Sigmoid函数

> **饱和：**
>
> 对于函数$f(x)$,$x\to -\infty $, $f'(x) \to 0$,则称为**左饱和**；$x\to +\infty $, $f'(x) \to 0$,则称为**右饱和**
>
> 同时满足左饱和、右饱和成为**两端饱和**

##### Logistic函数

$$
\sigma(x) = \frac{1}{1+e^{-x}}
$$

> - "挤压"函数：
>   - $0$附近近似为线性函数
>   - 两端对输入进行抑制,$0$附近$\to0$,$1$附近$\to1$
> - 连续可导
>   - $\sigma'(x)=\sigma(x)(1-\sigma(x))$

##### Tanh函数

$$
tanh(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}} = 2\sigma(2x)-1
$$

> - 值域为$(-1,1)$
>
> - 输出为Zero-Centered
>
>   > ​	非零中心化的输出会使得最后一层的神经元的输入发生偏置，梯度下降的收敛速度降低

![image-20201017130754418](C:\Users\Nector\AppData\Roaming\Typora\typora-user-images\image-20201017130754418.png)

##### Hard-Logistic函数

$$
hard-logistic(x) = max(min(0.25x+0.5,1),0)
$$

![image-20201017131648651](C:\Users\Nector\AppData\Roaming\Typora\typora-user-images\image-20201017131648651.png)

##### Hard-tanh函数

$$
hard-tanh(x)=max(min(x,1),-1)
$$

![image-20201017131658333](C:\Users\Nector\AppData\Roaming\Typora\typora-user-images\image-20201017131658333.png)

#### ReLU函数

$$
ReLU(x)=max(0,x)
$$

> **优点：**
>
> - 计算操作简单：加、乘和比较
> - 生物学合理性：
>   - 单侧抑制
>   - 宽兴奋边界
>   - 稀疏性：50%神经元处于激活状态
> - 优化方面的优点：
>   - 左饱和
>   - $x>0$导数为$1$，缓解梯度消失问题，加速梯度下降的收敛速度
>
> **缺点：**
>
> - 输出非Zero-centered，最后一层神经网络引入偏置偏移，影响梯度下降的效率
> - 神经元训练时容易"死亡"——Dying ReLU Problem

##### 带泄露的ReLU

![image-20201017132544907](C:\Users\Nector\AppData\Roaming\Typora\typora-user-images\image-20201017132544907.png)

> $\gamma$为$x<0$时的梯度，使得神经元未激活时也有梯度可以更新参数，避免永远不被激活的问题

##### 带参数的ReLU

对于第$i$个神经元：

![image-20201017132834654](C:\Users\Nector\AppData\Roaming\Typora\typora-user-images\image-20201017132834654.png)

> $\gamma\ge 0$是超参数，决定$x\le0$时的饱和曲线，调整期输出均值在$0$附近

### Maxout单元

<u>特点：</u>接受上层神经元的全部原始输出——输入是向量$\pmb{x}=[x_1;x_2;...;x_D]$

$z_k=\pmb{\omega_k^Tx}+b_k$

  $\pmb{w_k}=[w_{k,1},...,w_{k,D}]^T$
$$
maxout(\pmb{x})=max_{k\in[1,K]}(z_k)
$$

> - 整体学习输入到输出之间的非线性映射关系
> - 任意凸函数的分段线性近似
> - 在有限点上不可微

## 网络结构

### 前馈神经网络（FNN）——多层感知器（MLP）

- **记号：**

   $\pmb{W^{(l)}}$: 第$l-1$到第$l$层的权重矩阵

  $\pmb{b}^{(l)}$:第$l-1$层到第$l$层的偏置

  $\pmb{z}^{(l)}$:第$l$层神经元的净输入——净活性值

  $\pmb{a}^{(l)}$:第$l$层神经元的输出——活性值

- **计算公式：**

  $\pmb{z^{(l)}} = \pmb{W^{(l)}}f_{l-1}(z^{(l-1)})+\pmb{b^{(l)}}$

- 整个神经网络的输出可以记为$\phi(\pmb{x;W,b})$

> <u>通用近似定理：</u>
>
> 由线性输出层和至少一个使用"挤压"性质的激活函数的隐藏层组成的前馈神经网络，可以任意精度近似一个在$D$维实数空间中的有界闭集函数
>
> - 并未说明如何找到这样的网络
> - 不知道这样的网络是否最优

- 特征抽取：将样本的原始特征向量$\pmb{x}$转换到更有效的特征向量$\phi\pmb{(x)}$

  > 多层前馈网络可以看作一种特征转换方法：$\pmb{x}\to\phi(\pmb{x})$

- 将特征抽取的结果$\phi(\pmb{x})$作为分类器$g(·)$的输入

  > $\hat y=g(\phi(\pmb{x};\theta)$

- 分类器:

  - 二分类：Logistic函数

    最后一层直接输出$y=1$的条件概率

  - 多分类：

    网络最后一层多个神经元，每一层的输出可作为每个类的条件概率

### 反向传播算法

![image-20201017161845116](C:\Users\Nector\AppData\Roaming\Typora\typora-user-images\image-20201017161845116.png)

第$l$层的误差项由第$l+1$层的误差项导出——成为误差的反向传播（**BackPropagation，BP**）

> 含义：第$l$层的一个神经元的误差项是所有与该神经元相连的第$l+1$层的神经元的误差项的权重和

- **关于参数矩阵的计算公式：**

  ![image-20201017164517592](C:\Users\Nector\AppData\Roaming\Typora\typora-user-images\image-20201017164517592.png)

- **关于偏移矩阵的计算公式：**

  ![image-20201017164551326](C:\Users\Nector\AppData\Roaming\Typora\typora-user-images\image-20201017164551326.png)



#### 前馈神经网络训练过程：

1. 计算每一层的净输入$\pmb{z}^{(l)}$和激活值$\pmb{a}^{(l)}$

   > $\pmb{z}^{(l)}=\pmb{W}^{(l)}\pmb{a}^{(l-1)}+\pmb{b}^{(l)}$
   >
   > $\pmb{a}^{(l)}=f_l(\pmb{z}^{(l)})$

2. 反向传播计算每一层的误差项$\delta^{(l)}$
3. 计算每一层的偏导数并更新参数

![image-20201017165255239](C:\Users\Nector\AppData\Roaming\Typora\typora-user-images\image-20201017165255239.png)

#### 自动计算梯度：

- **自动微分：**
  - 计算图（Computational Graph）
  - 




















