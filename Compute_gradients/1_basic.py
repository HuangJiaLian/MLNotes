'''
@Description: 
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-09-06 16:19:44
@LastEditors: Jack Huang
@LastEditTime: 2019-09-06 17:41:34
'''

import tensorflow as tf 

x = tf.Variable(initial_value=5., dtype='float32')
w = tf.Variable(initial_value=1., dtype='float32')
y = w*(x**2)

opt = tf.train.GradientDescentOptimizer(0.1)
grad = opt.compute_gradients(y, [w,x])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(grad))

# 结果[(25.0, 1.0), (10.0, 5.0)]的解释如下:
# y'(x) = 2x*w => [10,5]: 当 x=５时，对应偏导值
# y'(w) = x**2 => [25,1]: 当 w=1时，对应的偏导值 
