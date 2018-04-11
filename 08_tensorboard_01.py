# build network
import tensorflow as tf
import numpy as np 


def add_layer(inputs, in_size, out_size, actication_function = None):
	with tf.name_scope('layer'):
		with tf.name_scope('weights'):
			Weights = tf.Variable(tf.random_normal([in_size,out_size]), name='w')
		with tf.name_scope('biases'):
			biases = tf.Variable(tf.zeros([1,out_size]) + 0.1) # Because the recommend initial
													# value of biases != 0; so add 0.1
		with tf.name_scope('Wx_plus_b'):
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

with tf.name_scope('inputs'):
	xs = tf.placeholder(tf.float32, [None, 1], name='x_input') # * rows, 1 col
	ys = tf.placeholder(tf.float32, [None, 1], name='y_input') # * rows, 1 col


# define hidden layer and output layer
l1 = add_layer(xs, 1, 10, actication_function = tf.nn.relu)
prediction = add_layer(l1, 10, 1, actication_function = None)

with tf.name_scope('loss'):
	# the error between prediction and real data 
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), 
				reduction_indices=[1]))
with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()


sess = tf.Session()
# writer = tf.train.SummaryWriter("logs/", sess.graph)
writer = tf.summary.FileWriter("logs/", sess.graph)

sess.run(init)

for i in range (1000):
	# ??? tensorflow know update the x_data, y_data at each train ???
	sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
	if i % 50 == 0:
		print(sess.run(loss,feed_dict={xs:x_data, ys:y_data}))

