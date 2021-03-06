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
Ftrain_fit = tf.placeholder(tf.float64, shape=(nfunc,ntrain))
# input
Ytrain_fit = tf.placeholder(tf.float64, shape=(nx,ntrain))
Ytest_fit  = tf.placeholder(tf.float64, shape=(nx,ntest))

# NN weights and biases

W = tf.get_variable('W',(nfunc,nx),
              dtype=tf.float64,
              initializer = tf.random_normal_initializer)
B = tf.get_variable('B',(nfunc,1),
              dtype=tf.float64,
              initializer=tf.constant_initializer(0.0))

##############

npolywts = 3;

def myNNfunc(YY,W,B,ww):
  # YY(nx,nvec)
  # W(nfunc,nx)  B(nfunc,1)
  # ww(npolywts,nvec)

  nvec = YY.shape[1];
  Y0 = tf.reshape(ww[0,:],[1,nvec]) + \
       tf.reshape(ww[1,:],[1,nvec]) * YY + \
       tf.reshape(ww[2,:],[1,nvec]) * YY**2

  Q1 = tf.matmul(W,Y0) + B
  F1 = tf.sigmoid(Q1)
  return F1

#### SAMPLE TRAIN PARAMS

# sample polynomial weights

doSampleFit = True

if doSampleFit:
  wtrain = tf.get_variable('wtrain',(npolywts,ntrain),
           dtype=tf.float64,
           initializer = tf.random_normal_initializer)
  wtest  = tf.get_variable('wtest',(npolywts,ntest),
           dtype=tf.float64,
           initializer = tf.random_normal_initializer)
else:
  wtrain = np.array([[0],[1],[0]])*np.ones([1,ntrain])
  wtest  = np.array([[0],[1],[0]])*np.ones([1,ntest])

# print(wtrain[0,:].reshape(1,ntrain).shape)

######      TRAIN    ######

Ftrain_NN = myNNfunc(Ytrain_fit,W,B,wtrain)

# error for each sample
train_lossper = tf.reshape(
  tf.reduce_sum((Ftrain_fit-Ftrain_NN)**2,axis=0),(1,ntrain))
# summed over samples
train_LOSS = tf.reduce_sum(train_lossper);
OPT_train = tf.train.AdamOptimizer().minimize(train_LOSS)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for _ in range(2000):
    
    _, train_loss_, train_lossper_, Ftrain_NN_, Wtrain_, Btrain_ = sess.run(
      [OPT_train,train_LOSS,train_lossper,Ftrain_NN,W,B]
      ,feed_dict={Ytrain_fit:ytrain_actual,Ftrain_fit:ftrain_actual})
    print('loss: %s '%(train_loss_))

Wtrain_ = np.double(Wtrain_)
Btrain_ = np.double(Btrain_)

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


if not doSampleFit:

  Ftest_NN = myNNfunc(ytest_actual,Wtrain_,Btrain_,wtest)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    Ftest_NN_ = sess.run(Ftest_NN)

else:

  Ftest_NN = myNNfunc(Ytest_fit,Wtrain_,Btrain_,wtest)
  # error for each sample
  test_lossper = tf.reshape(
    tf.reduce_sum((Ftest_NN-1)**2*Ftest_NN**2,axis=0),(1,ntest))
  # summed over samples
  test_LOSS = tf.reduce_sum(test_lossper);
  OPT_test = tf.train.AdamOptimizer().minimize(test_LOSS)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for _ in range(2000):
    
      _, test_loss_, test_lossper_, Ftest_NN_ = sess.run(
        [OPT_test,test_LOSS,test_lossper,Ftest_NN]
        ,feed_dict={Ytest_fit:ytest_actual})
      print('loss: %s '%(test_loss_))

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
