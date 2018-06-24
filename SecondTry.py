#!/usr/local/bin/python3.6

xrange     = 4
nper       = 10
nsample    = 30

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# positive-valued functions

def myquad(x):
  return np.power(x/xrange,2)

def mysin(x):
  return np.sin(16.28*x/xrange)**2

funclist = [mysin,myquad]

###########################

nfunc = len(funclist)

# positive-valued function

def yfunc(ifuncs,xvals,xshift,xfac,yfac):
  nsample = ifuncs.size
  assert np.all(np.array(
    [xshift.size,xfac.size,yfac.size]) == nsample)
  assert np.all(ifuncs >= 0) and np.all(ifuncs < nfunc)
  yvals = np.zeros([nx,nsample])
  for isp in range(nsample):
    yvals[:,isp] = \
      yfac[isp]**2 * funclist[ifuncs[isp]](xshift[isp] +
      xfac[isp] * xvals[:,0])
  return yvals

functype = np.random.choice(nfunc,nsample)
fdata    = np.zeros([nfunc,nsample])
for isp in range(nsample):
  fdata[functype[isp],isp] = 1

# x-values at which to evaluate the functions

nx = xrange * nper
xdata = np.arange(xrange*nper)/nper
xdata = np.reshape(xdata,(nx,1));

# get the positive-valued functions ydata(nx,nsample)
#   with random parameters

xshift   = np.random.normal(np.zeros(nsample))
xfac     = np.random.normal(np.zeros(nsample))
yfac     = np.random.normal(np.zeros(nsample))

ydata = yfunc(functype,xdata,xshift,xfac,yfac)

for isp in range(nsample):
  plt.scatter(xdata,ydata[:,isp])
plt.show()

# output to fit
FF = tf.placeholder(tf.float32, shape=(nfunc,nsample))

# input
YY = tf.placeholder(tf.float32, shape=(nx,nsample))

W1 = tf.get_variable('W1',(nfunc,nx),
          initializer = tf.random_normal_initializer)
B1 = tf.get_variable('b1',(nfunc,1),
          initializer=tf.constant_initializer(0.0))

F1 = tf.sigmoid(tf.matmul(W1**2,YY) + B1)

# error for each sample
Fpart = tf.reshape(tf.reduce_sum((FF-F1)**2,axis=0),(1,nsample))

Fsum = tf.reduce_sum(Fpart)

LOSS = Fsum;

OPT_operation = tf.train.AdamOptimizer().minimize(LOSS)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for _ in range(500):
    
    _, loss_, Fpart_, F1_ = sess.run(
      [OPT_operation,LOSS,Fpart,F1],
      feed_dict={YY:ydata,FF:fdata})

    print('loss: %s '%(loss_))

sindex = np.arange(nsample);
sindex = np.reshape(sindex,(1,nsample))

print('errors and actual')
print(np.transpose(np.concatenate((Fpart_,functype[sindex]))))

print()

print('output')
print(np.transpose(F1_))

plt.scatter(sindex,Fpart_)
plt.show()
plt.scatter(functype[sindex],Fpart_)
plt.show()
