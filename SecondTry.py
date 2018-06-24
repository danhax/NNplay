#!/usr/local/bin/python3.6

xrange     = 10
sampledev  = 30
nper       = 4
nsample    = 5
batch_size = 4

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sampleshift = np.random.normal(np.zeros([1,nsample]),sampledev)

nx = xrange * nper

xdata = np.arange(xrange*nper)/nper

xdata = np.reshape(xdata,(nx,1));

xdata = xdata + sampleshift;

# print(xdata[:,0].shape)
# print(xdata.shape)
# exit()

def sinfunc(xvals,shift,lin,fac):
  yvals = shift + lin * xvals / xrange + fac * np.sin(xvals)
  return yvals

# ydata = xdata + xrange / 2 * np.sin(xdata)

ydata = sinfunc(xdata,3,1,2)

ny = len(ydata)

xx = tf.placeholder(tf.float32, shape=(nx,batch_size))
yy = tf.placeholder(tf.float32, shape=(ny,batch_size))

XX = tf.placeholder(tf.float32, shape=(nx,nsample))
YY = tf.placeholder(tf.float32, shape=(nx,nsample))

# with tf.variable_scope('linreg'):

# linear transform y = A * x + b

A1 = tf.get_variable('A1',(ny,nx),
          initializer = tf.random_normal_initializer)
b1 = tf.get_variable('b1',(ny,1),
          initializer=tf.constant_initializer(0.0))

y1 = tf.matmul(A1,xx) + b1
Y1 = tf.matmul(A1,XX) + b1

# print(y1.shape)
# print(nx,ny)
# exit()

# # loss = tf.reduce_sum((yy-y1)**2)/ny
# loss   = tf.reduce_sum(tf.abs(yy-y1))/ny

# ysum = tf.reduce_sum((yy-y1)**2)
# Ysum = tf.reduce_sum((YY-Y1)**2)
# asum = tf.reduce_sum(A1**2)

ysum = tf.reduce_sum(abs(yy-y1))
Ysum = tf.reduce_sum(abs(YY-Y1))
asum = tf.reduce_sum(abs(A1))

asum = 1;

loss = ysum * asum
LOSS = Ysum * asum

opt_operation = tf.train.AdamOptimizer().minimize(loss)
OPT_operation = tf.train.AdamOptimizer().minimize(LOSS)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for _ in range(5000):
    indices = np.random.choice(nsample,batch_size)

    xbatch,ybatch = xdata[:,indices], ydata[:,indices]
#    _, loss_ = sess.run([opt_operation,loss],
#                     feed_dict={xx:xbatch,yy:ybatch})

    _, loss_ = sess.run([OPT_operation,LOSS],
                     feed_dict={XX:xdata,YY:ydata})

    Y1_, = sess.run([Y1],{XX:xdata})

#     print('weight: %s  bias: %s  loss: %s '
#       %(currweights,currbias,currloss))
    print('loss: %s '%(loss_))

plt.scatter(xdata,ydata)
plt.scatter(xdata,Y1_)
plt.show()


