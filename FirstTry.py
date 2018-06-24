#!/usr/local/bin/python3.6

xrange     = 10
nper       = 4
nsample    = 5
sampleshift = [0,12,50,-23.5, 13.3];
# batch_size = 4

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

nx = xrange * nper

xdata = np.arange(xrange*nper)/nper

xdata = np.reshape(xdata,(nx,1));

xdata = xdata + sampleshift;

# print(xdata.shape)
# exit()

ydata = xdata + 5 * np.sin(xdata)

ny = len(ydata)

xx = tf.placeholder(tf.float32, shape=(nx,nsample))
yy = tf.placeholder(tf.float32, shape=(ny,nsample))

# with tf.variable_scope('linreg'):

# linear transform y = A * x + b

A1 = tf.get_variable('A1',(ny,nx),
          initializer = tf.random_normal_initializer)
b1 = tf.get_variable('b1',(1,1),
          initializer=tf.constant_initializer(0.0))

y1 = tf.matmul(A1,xx) + b1

# print(y1.shape)
# print(nx,ny)
# exit()

# # loss = tf.reduce_sum((yy-y1)**2)/ny
# loss   = tf.reduce_sum(tf.abs(yy-y1))/ny

# ysum = tf.reduce_sum((yy-y1)**2)
# asum = tf.reduce_sum(A1**2)

ysum = tf.reduce_sum(abs(yy-y1))
asum = tf.reduce_sum(abs(A1))

loss = ysum / asum
opt_operation = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for _ in range(500):
    # indices = np.random.choice(nx,batch_size)
    # xbatch,ybatch = xdata[xindices], ydata[indices]
    #
    _, loss_val = sess.run([opt_operation, loss],
                  feed_dict={xx:xdata,yy:ydata})

    A1_, b1_, y1_, loss_ = sess.run(
      [A1,b1,y1,loss],{xx:xdata,yy:ydata})

#     print('weight: %s  bias: %s  loss: %s '
#       %(currweights,currbias,currloss))
    print('loss: %s '%(loss_))

plt.scatter(xdata,ydata)
plt.scatter(xdata,y1_)
plt.show()


