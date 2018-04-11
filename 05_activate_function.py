# add layer 
import tensorflow as tf

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
