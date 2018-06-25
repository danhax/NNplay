#!/usr/local/bin/python3.6

xrange     = 4
nper       = 10
ntrain    = 1000
ntest     = 20

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# positive-valued functions

def myquad(x):
  return x/xrange + np.power(x/xrange,2)

def mysin(x):
  return np.sin(16.28*x/xrange)**2

funclist = [mysin,myquad]

###########################

nfunc = len(funclist)

# x-values at which to evaluate the functions

nx = xrange * nper
xdata = np.arange(xrange*nper)/nper
xdata = np.reshape(xdata,(nx,1));

# get the positive-valued functions ydata(nx,ntrain)
#   with random parameters

def yfunc(ifuncs,xvals,xshift,xfac,yfac):
  # positive-valued function
  #
  ninput = ifuncs.size
  assert np.all(np.array(
    [xshift.size,xfac.size,yfac.size]) == ninput)
  assert np.all(ifuncs >= 0) and np.all(ifuncs < nfunc)
  yvals = np.zeros([nx,ninput])
  for isp in range(ninput):
    yvals[:,isp] = \
      yfac[isp]**2 * funclist[ifuncs[isp]](xshift[isp] +
      xfac[isp] * xvals[:,0])
  return yvals

def getdata(functype):
  numvecs = functype.size
  xshift = np.random.normal(np.zeros(numvecs))
  xfac   = np.random.normal(np.zeros(numvecs))
  yfac   = np.random.normal(np.zeros(numvecs))
  ydata  = yfunc(functype,xdata,xshift,xfac,yfac)
  return ydata

functype_train = np.random.choice(nfunc,ntrain)
functype_test  = np.random.choice(nfunc,ntest)

ftrain_actual          = np.zeros([nfunc,ntrain])
for ivec in range(ntrain):
  ftrain_actual[functype_train[ivec],ivec] = 1

ftest_actual          = np.zeros([nfunc,ntest])
for ivec in range(ntest):
  ftest_actual[functype_test[ivec],ivec] = 1

ytrain_actual = getdata(functype_train)
ytest_actual  = getdata(functype_test)

# for ivec in range(ntrain):
#   plt.scatter(xdata,ytrain_actual[:,ivec])
# plt.show()



##### DO TENSORFLOW

# output to fit for training
Ftrain_fit = tf.placeholder(tf.float32, shape=(nfunc,ntrain))
# input
Ytrain_fit = tf.placeholder(tf.float32, shape=(nx,ntrain))
Ytest_fit  = tf.placeholder(tf.float32, shape=(nx,ntest))

# # sample weights and biases
#
# w0train = tf.get_variable('w0train',(1,ntrain),
#              initializer = tf.random_normal_initializer)
# b0train = tf.get_variable('b0train',(1,ntrain),
#              initializer=tf.constant_initializer(0.0))
# w0test  = tf.get_variable('w0test',(1,ntest),
#              initializer = tf.random_normal_initializer)
# b0test  = tf.get_variable('b0test',(1,ntest),
#              initializer=tf.constant_initializer(0.0))

# NN weights and biases

W = tf.get_variable('W',(nfunc,nx),
              initializer = tf.random_normal_initializer)
B = tf.get_variable('B',(nfunc,1),
              initializer=tf.constant_initializer(0.0))

def myNNfunc(YY,W,B,w0,b0):
  # YY(nx,nvec)
  
  # Y0 = YY * w0 + b0;

  # nvec = YY.shape[1]
  # ymean = tf.reshape(tf.reduce_sum(YY,axis=0),(1,nvec)) / nx
  # ydev =  tf.sqrt(tf.reshape(tf.reduce_sum((YY-ymean)**2,axis=0),(1,nvec)) / nx)
  # Y0 = ( YY - ymean ) / ydev

  Y0 = YY

  Q = tf.matmul(W,Y0) + B
  F = tf.sigmoid(Q)
  return F


######      TRAIN    ######

Ftrain_NN = myNNfunc(Ytrain_fit,W,B,1,0)

# error for each sample
train_lossper = tf.reshape(
  tf.reduce_sum((Ftrain_fit-Ftrain_NN)**2,axis=0),(1,ntrain))
# summed over samples
train_LOSS = tf.reduce_sum(train_lossper);
OPT_train = tf.train.AdamOptimizer().minimize(train_LOSS)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for _ in range(20000):
    
    _, train_loss_, train_lossper_, Ftrain_NN_, Wtrain_, Btrain_ = sess.run(
      [OPT_train,train_LOSS,train_lossper,Ftrain_NN,W,B],
      feed_dict={Ytrain_fit:ytrain_actual,Ftrain_fit:ftrain_actual})
    print('loss: %s '%(train_loss_))

trainindex = np.arange(ntrain);
trainindex = np.reshape(trainindex,(1,ntrain))

besttrain = np.reshape(np.argmax(Ftrain_NN_,axis=0),(1,ntrain))

print('TRAIN: actual, bestguess, error')
print(np.array2string(np.transpose(np.concatenate(
  (functype_train[trainindex], besttrain, train_lossper_))),
  formatter={'float_kind':lambda x: '%.4f' % x}))

plt.scatter(trainindex,train_lossper_)
plt.show()
plt.scatter(functype_train[trainindex],train_lossper_)
plt.show()


######      TEST    ######


Wtrain_ = np.double(Wtrain_)
Btrain_ = np.double(Btrain_)

Ftest_NN = myNNfunc(ytest_actual,Wtrain_,Btrain_,1,0)

# exit()

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  Ftest_NN_ = sess.run(Ftest_NN)

Ftest_NN_ = np.array(Ftest_NN_)

testindex = np.arange(ntest);
testindex = np.reshape(testindex,(1,ntest))

test_errorper = np.reshape(
  np.sum((ftest_actual-Ftest_NN_)**2,axis=0),(1,ntest))

besttest = np.reshape(np.argmax(Ftest_NN_,axis=0),(1,ntest))

print('TEST: actual, bestguess,error')
print(np.array2string(np.transpose(np.concatenate(
  (functype_test[testindex], besttest, test_errorper))),
  formatter={'float_kind':lambda x: '%.4f' % x}))

plt.scatter(testindex,test_errorper)
plt.show()
plt.scatter(functype_test[testindex],test_errorper)
plt.show()
