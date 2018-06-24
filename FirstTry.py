#!/usr/local/bin/python3.6

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

xrange  = 10
nper = 4
nx = xrange * nper

batch_size = 4

xdata = np.arange(xrange*nper)/nper

# print(xdata.size)
# print(nx)
# print(len(xdata))
# assert nx == len(xdata)
# exit()

ydata = xdata + 5 * np.sin(xdata)

ny = len(ydata)

xdata = np.reshape(xdata,(nx,1))
ydata = np.reshape(ydata,(ny,1))

# plt.scatter(xdata,ydata)
# plt.show()

xx = tf.placeholder(tf.float32, shape=(nx,1))
yy = tf.placeholder(tf.float32, shape=(ny,1))

# with tf.variable_scope('linreg'):

# linear transform y = A * x + b

A1 = tf.get_variable('A1',(1,1),
          initializer = tf.random_normal_initializer)
b1 = tf.get_variable('b1',(1,1),
          initializer=tf.constant_initializer(0.0))

y1 = tf.matmul(xx,A1) + b1

# loss = tf.reduce_sum((yy-y1)**2/ny)
loss   = tf.reduce_sum(tf.abs(yy-y1)/ny)

opt_operation = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for _ in range(5000):
    # indices = np.random.choice(nx,batch_size)
    # xbatch,ybatch = xdata[xindices], ydata[indices]
    #
    _, loss_val = sess.run([opt_operation, loss],
                  feed_dict={xx:xdata,yy:ydata})

    currweights, currbias, currloss = sess.run(
      [A1,b1,loss],{xx:xdata,yy:ydata})

    print('weight: %s  bias: %s  loss: %s '%(currweights,currbias,currloss))

plt.scatter(xdata,ydata)
plt.scatter(xdata,currweights*xdata+currbias)
plt.show()


