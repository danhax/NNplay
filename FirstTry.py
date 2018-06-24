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

ydata = xdata + 20 * np.sin(xdata/10)

xdata = np.reshape(xdata,(ndat,1))
ydata = np.reshape(ydata,(ndat,1))

print(xdata.size)

# plt.scatter(xdata,ydata)
# plt.show()

xx = tf.placeholder(tf.float32, shape=(ndat,1))
yy = tf.placeholder(tf.float32, shape=(ndat,1))

# with tf.variable_scope('linreg'):

weights = tf.get_variable('weights',(1,1),
          initializer = tf.random_normal_initializer)
bias = tf.get_variable('bias',(1,),
          initializer=tf.constant_initializer(0.0))

ypred = tf.matmul(xx,weights) + bias
# loss = tf.reduce_sum((yy-ypred)**2/ndat)
loss   = tf.reduce_sum(tf.abs(yy-ypred)/ndat)

# opt = tf.train.AdamOptimizer()
# opt_operation = opt.minimize(loss)

opt_operation = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  _, loss_val = sess.run([opt_operation,loss],feed_dict={xx:xdata,yy:ydata})

  print(loss_val)

  for _ in range(5000):
    indices = np.random.choice(ndat,batch_size)
    xbatch,ybatch = xdata[indices], ydata[indices]
    _, loss_val = sess.run([opt_operation, loss], feed_dict={xx:xdata,yy:ydata})

    #  print(loss_val)
    
    currweights, currbias, currloss = sess.run([weights,bias,loss],{xx:xdata,yy:ydata})

    print('weight: %s  bias: %s  loss: %s '%(currweights,currbias,currloss))

plt.scatter(xdata,ydata)
plt.scatter(xdata,currweights*xdata+currbias)
plt.show()


