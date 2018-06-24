#!/usr/local/bin/python3.6

xrange     = 4
nper       = 10
nsample    = 12

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def myquad(x):
  return 10 * np.power(x/xrange,2)

def mysin(x):
  return 10 * np.sin(6.28*x/xrange)

funclist = [mysin,myquad]

###########################

nfunc = len(funclist)

functype = np.random.choice(nfunc,nsample)
fdata    = np.zeros([nfunc,nsample])
for isp in range(nsample):
  fdata[functype[isp],isp] = 1

xshift   = np.random.normal(np.ones(nsample))
yshift   = np.random.normal(np.ones(nsample))
xfac     = np.random.normal(np.ones(nsample))
yfac     = np.random.normal(np.ones(nsample))

nx = xrange * nper

xdata = np.arange(xrange*nper)/nper
xdata = np.reshape(xdata,(nx,1));

# print(xdata[:,0].shape)
# print(xdata.shape)
# exit()

def yfunc(ifuncs,xvals,xshift,yshift,xfac,yfac):
  nsample = ifuncs.size
  assert np.all(np.array(
    [xshift.size,yshift.size,xfac.size,yfac.size]) == nsample)
  assert np.all(ifuncs >= 0) and np.all(ifuncs < nfunc)
  yvals = np.zeros([nx,nsample])
  for isp in range(nsample):
    yvals[:,isp] = \
      yshift[isp] + yfac[isp] * funclist[ifuncs[isp]](
      xshift[isp] + xfac[isp] * xvals[:,0])
  return yvals

ydata = yfunc(functype,xdata,xshift,yshift,xfac,yfac)

FF = tf.placeholder(tf.float32, shape=(nfunc,nsample))

YY = tf.placeholder(tf.float32, shape=(nx,nsample))

A1 = tf.get_variable('A1',(nfunc,nx),
          initializer = tf.random_normal_initializer)
B1 = tf.get_variable('b1',(nfunc,1),
          initializer=tf.constant_initializer(0.0))

F1 = tf.sigmoid(tf.matmul(A1,YY) + B1)

# error for each sample
Fpart = tf.reshape(tf.reduce_sum((FF-F1)**2,axis=0),(1,nsample))

Fsum = tf.reduce_sum(Fpart)

LOSS = Fsum;

# opt_operation = tf.train.AdamOptimizer().minimize(loss)
OPT_operation = tf.train.AdamOptimizer().minimize(LOSS)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for _ in range(500):
#    indices = np.random.choice(nsample,batch_size)
#    xbatch,ybatch = xdata[:,indices], ydata[:,indices]

    _, loss_, Fpart_, F1_ = sess.run(
      [OPT_operation,LOSS,Fpart,F1],
      feed_dict={YY:ydata,FF:fdata})

    print('loss: %s '%(loss_))

sindex = np.arange(nsample);
sindex = np.reshape(sindex,(1,nsample))

print(np.transpose(np.concatenate((Fpart_,functype[sindex]))))

#print(Fpart_)
#print(functype[sindex])

print()

print(np.transpose(F1_))

plt.scatter(sindex,Fpart_)
plt.show()
plt.scatter(functype[sindex],Fpart_)
plt.show()
