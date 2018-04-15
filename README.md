# ML Notes
# 莫烦 Python Tensorflow 笔记

## **神经网络如何从经验中学习?**

就是对比正确答案和错误答案之间的区别，然后把这个区别反向的传递回去，对每个相应的神经元进行一点点的改变。那么下一次在训练的时候就可以用已经改进一点点的神经元去得到稍微准确一点的结果。

## **为什么需要激励函数：**

解决现实生活中不能用线性方程概括的问题

只有可微分的激励函数在误差反向传递的时候才能将误差传递回去。???

## 激励函数怎么选择

在网络层数比较少时，随意选择。在卷积神经网络中使用relu, 在循环神经网络中使用relu或tanh

softmax通常是用来做classification的



## 如何加速神经网络训练

- Stochastic Gradient Descent (SGD)(一批一批喂)
- Momentum(给醉汉一个下坡，利用惯性)
- AdaGrad(给醉汉一双鞋，走错方向脚就会疼)
- RMSProp（结合惯性和那双鞋）
- Adam

## 线性回归和分类算法的区别

线性回归是连续的关系

分类算法是离散的

## MNIST数据集

数据中包含55000张训练图片，每张图片的分辨率是28×28，所以我们的训练网络输入应该是28×28=784个像素数据。

MNIST数据集有包含训练数据(trainning data)和测试数据(test data)

## 过拟合(Overfitting)

自负=过拟合

**解决方法：**

- 增加数据量

- L1,L2 regularization(正则化)，
    $y = w\cdot x$

    $L1: cost = (W\cdot x - y_{real})^2 + abs(w)$

    $L2: cost = (W\cdot x - y_{real})^2 + (w)^2$

- Dropout regularization

## (卷积神经网络)CNN

每次做卷积的时候神经层会无意丢失一些东西: 解决方法:(池化)pooling

常用的CNN结构:

<img src="https://morvanzhou.github.io/static/results/ML-intro/cnn6.png">

patch/kernel 有用到一个参数叫做STRIDE:每次跨多少个像素点

抽离信息的方式叫做PADDING

- valid padding (裁剪一点点)
- same padding(和原来一样大)

POOLING

- maxima pooling 

- average pooling
 