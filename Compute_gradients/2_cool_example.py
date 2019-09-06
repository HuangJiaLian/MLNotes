'''
@Description: 
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-09-06 16:26:10
@LastEditors: Jack Huang
@LastEditTime: 2019-09-06 17:59:18
'''
# build network
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 

def add_layer(inputs, in_size, out_size, actication_function = None):
	Weights = tf.Variable(tf.random_normal([in_size,out_size]))
	biases = tf.Variable(tf.zeros([1,out_size]) + 0.1) # Because the recommend initial
												# value of biases != 0; so add 0.1
	Wx_plus_b = tf.matmul(inputs, Weights) + biases
	
	if actication_function is None:
		outputs = Wx_plus_b
	else:
		outputs = actication_function(Wx_plus_b)
	return outputs

# create data
# 300 elenments from -1 to 1 
x_data = np.linspace(-1,1,300)[:, np.newaxis] 
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1]) # * rows, 1 col
ys = tf.placeholder(tf.float32, [None, 1]) # * rows, 1 col


# define hidden layer and output layer
l1 = add_layer(xs, 1, 3, actication_function = tf.nn.sigmoid)
prediction = add_layer(l1, 3, 1, actication_function = None)




loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
opt = tf.train.GradientDescentOptimizer(0.1)
grad = opt.compute_gradients(loss,tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
train_step = opt.minimize(loss)


init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

for i in range (2000):
	sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
	if i % 20 == 0:
		print('Loss:', sess.run(loss,feed_dict={xs:x_data, ys:y_data}))
		try:
			ax.lines.remove(lines[0])
		except Exception :
			pass
		prediction_value = sess.run(prediction, feed_dict={xs:x_data, ys:y_data})
		lines = ax.plot(x_data, prediction_value, 'r-', lw = 4)
		plt.pause(0.1)
    
    # 用来测试求偏导
	if i  == 1999:
		print('Trainable Variables:', tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
		print('Grad:', sess.run(grad,feed_dict={xs:x_data, ys:y_data}))




##################
# 可以被训练的参数:  
##################
# # 对应 w11,w12, w13
# [<tf.Variable 'Variable:0' shape=(1, 3) dtype=float32_ref>,  
# # 对应 b11, b12, b13
#  <tf.Variable 'Variable_1:0' shape=(1, 3) dtype=float32_ref>, 
# # 对应　w21, w22, w23
#  <tf.Variable 'Variable_2:0' shape=(3, 1) dtype=float32_ref>, 
# # 对应　b21
#  <tf.Variable 'Variable_3:0' shape=(1, 1) dtype=float32_ref>]


###################
# 结果解释：
###################
# [w11,w12,w13] = [1.4377401, 3.1283407, -1.8470509]时的偏导值为[-0.00658626, -0.00904603,  0.00350215]

# # 对应　w11, w12, w13
# [　　　 (array([[-0.00658626, -0.00904603,  0.00350215]], dtype=float32), array([[ 1.4377401,  3.1283407, -1.8470509]], dtype=float32)), 
# # 对应  b11, b12, b13　　　
# 　　　　(array([[ 0.00469836, -0.00399115, -0.00315102]], dtype=float32), array([[-0.58592653,  1.7151704 ,  0.7725139 ]], dtype=float32)), 
# # 对应 w21, w22, w23 
# 	   (array([[-0.00720509],
#        [ 0.00878299],
#        [ 0.00243923]], dtype=float32), 
#        array([[ 1.6078004],
#        [-1.8724854],
#        [-0.7588247]], dtype=float32)), 
# # 对应 b21
#        (array([[-0.00723374]], dtype=float32), array([[1.1012503]], dtype=float32))]
