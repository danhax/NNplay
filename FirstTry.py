#!/usr/local/bin/python3.6

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

xrange  = 10
nper = 4
ndat = xrange * nper

batch_size = 4

xdata = np.arange(xrange*nper)/nper

# print(xdata.size)
# print(ndat)
# print(len(xdata))
# assert ndat == len(xdata)
# exit()

ydata = xdata + 5 * np.sin(xdata)

xdata = np.reshape(xdata,(ndat,1))
ydata = np.reshape(ydata,(ndat,1))

print(xdata.size)

# plt.scatter(xdata,ydata)
# plt.show()

xx = tf.placeholder(tf.float32, shape=(ndat,1))
yy = tf.placeholder(tf.float32, shape=(ndat,1))

# with tf.variable_scope('linreg'):

# linear transform y = A * x + b

A1 = tf.get_variable('A1',(1,1),
          initializer = tf.random_normal_initializer)
b1 = tf.get_variable('b1',(1,),
          initializer=tf.constant_initializer(0.0))

y1 = tf.matmul(xx,A1) + b1

# loss = tf.reduce_sum((yy-y1)**2/ndat)
loss   = tf.reduce_sum(tf.abs(yy-y1)/ndat)

opt_operation = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for _ in range(1000):
    indices = np.random.choice(ndat,batch_size)
    xbatch,ybatch = xdata[indices], ydata[indices]
    _, loss_val = sess.run([opt_operation, loss],
                  feed_dict={xx:xdata,yy:ydata})

    currweights, currbias, currloss = sess.run(
      [A1,b1,loss],{xx:xdata,yy:ydata})

    print('weight: %s  bias: %s  loss: %s '%(currweights,currbias,currloss))

plt.scatter(xdata,ydata)
plt.scatter(xdata,currweights*xdata+currbias)
plt.show()


