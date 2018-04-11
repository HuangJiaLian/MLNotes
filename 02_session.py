import tensorflow as tf

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
					  [2]])
product = tf.matmul(matrix1, matrix2) # matrix multiply np.dot(m1,m2)

# # method 1
# sess = tf.Session() # Session is an object , so  need in uppercase form
# result = sess.run(product)
# print(result)
# sess.close()

# method 2
with tf.Session() as sess: # no need sess.close when using "with ... as ... " structure
	result2 = sess.run(product)
	print(result2)
 
