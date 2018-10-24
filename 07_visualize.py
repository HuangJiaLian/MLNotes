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
l1 = add_layer(xs, 1, 10, actication_function = tf.nn.relu)
prediction = add_layer(l1, 10, 1, actication_function = None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), 
				reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

for i in range (1000):
	# ??? tensorflow know update the x_data, y_data at each train ???
	sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
	if i % 20 == 0:
		print(sess.run(loss,feed_dict={xs:x_data, ys:y_data}))
		try:
			ax.lines.remove(lines[0])
		except Exception :
			pass
		prediction_value = sess.run(prediction, feed_dict={xs:x_data, ys:y_data})
		lines = ax.plot(x_data, prediction_value, 'r-', lw = 4)
		plt.pause(0.1)

