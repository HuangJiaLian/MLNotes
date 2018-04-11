# The main idea of tensorflow:
# Define the functions
#  then use Session to control how to use those function


import tensorflow as tf

state = tf.Variable(0, name = "counter")
# print(state.name)
one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.initialize_all_variables() # must have if you have defined variable(s)

with tf.Session() as sess:
	sess.run(init) 
	for _ in range(3):
		sess.run(update)
		print(sess.run(state))
