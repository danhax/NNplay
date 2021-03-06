#!/usr/local/bin/python3.6

#######   settings     #######

xrange     = 4
nper       = 10
ntrain    = 500
ntest     = 20

convlen = 20
convnum = 5

nTrainSteps = 100000

##############################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

###### x-values at which to evaluate the functions

nx = xrange * nper
xdata = np.arange(xrange*nper)/nper
xdata = np.reshape(xdata,(nx,1));

######  positive-valued functions

def myquad(x):
  return x/xrange + np.power(x/xrange,2)

def mysin(x):
  return np.sin(16.28*x/xrange)**2

funclist = [mysin,myquad]

nfunc = len(funclist)

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

######  random functions for train and test

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

#####   OKAY.  convolution size:

nc = nx + 1 - convlen;

#####   DO TENSORFLOW

def myconv(inp,C):

  # want NWC inp(nvec, inlen, 1)    nbatch=nvec  width=inlen
  #          C   convlen,1,nconv)
  # input
  #   inp(nx,nvec)
  #   C(convlen,convnum)
  # output
  #   outp(nc,nvec,convnum)

  nvec = inp.shape[1]

  inp=tf.reshape(tf.transpose(inp),(nvec,nx,1))
  C  =tf.reshape(C,                (convlen,1,convnum))

  outp = tf.nn.conv1d(inp,C,1,'VALID')
  # have outp(nvec, nc, convnum)
  
  outp = tf.transpose(outp,perm=[1,0,2])
  # now have outp(nc,nvec,convnum)

  return outp

# output to fit for training
Ftrain_fit = tf.placeholder(tf.float64, shape=(nfunc,ntrain))

# input
Ytrain_fit = tf.placeholder(tf.float64, shape=(nx,ntrain))

# NN weights and biases

# weights, linear transformation W(nfunc,nc) for each convnum
#
C = tf.get_variable('C',(convlen,convnum),
              dtype=tf.float64,
              initializer = tf.random_normal_initializer)
W = tf.get_variable('W',(nfunc,convnum),
              dtype=tf.float64,
              initializer = tf.random_normal_initializer)
D = tf.get_variable('D',(nc,nfunc),
              dtype=tf.float64,
              initializer = tf.random_normal_initializer)
B = tf.get_variable('B',(nfunc,1),
              dtype=tf.float64,
              initializer=tf.constant_initializer(0.0))

##############

def myNNfunc(YY,C,W,D,B):
  # YY(nx,nvec)
  # C(convlen,convnum)    convolve YY
  # W(nfunc,convnum)      linear transform
  # D(nc,nfunc)           pixel contraction
  # B(nfunc,1)            constant term
  # out(nfunc,nvec)     0 to 1
  
  nvec = YY.shape[1]

  # Y0(nc,nvec,convnum)
  Y0 = myconv(YY,C)

  # Q1(nc,nvec,nfunc)
  Q1 = ( tf.tensordot(Y0,tf.transpose(W),axes=((2),(0)))
       )  #  + tf.reshape(B,(1,1,nfunc))

  # N1(nvec,nfunc)       norm-squared
  D = tf.reshape(D,(nc,1,nfunc));
  N1 = tf.reshape(tf.reduce_sum(Q1*D,axis=0),(nvec,nfunc))

  # N1(nfunc,nvec)       norm-squared
  N1 = tf.transpose(N1)

  # F1 = 1/(1+1/N1)
  F1 = tf.sigmoid( N1 + B )
  return F1

######      TRAIN    ######

Ftrain_NN = myNNfunc(Ytrain_fit,C,W,D,B)

# error for each sample
train_lossper = tf.reshape(
  tf.reduce_sum(tf.abs(Ftrain_fit-Ftrain_NN),axis=0),(1,ntrain))
#  tf.reduce_sum((Ftrain_fit-Ftrain_NN)**2,axis=0),(1,ntrain))

# summed over samples
train_LOSS = tf.reduce_sum(train_lossper);
OPT_train = tf.train.AdamOptimizer().minimize(train_LOSS)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for _ in range(nTrainSteps):
    
    _, train_loss_, train_lossper_, \
      Ftrain_NN_, C_, W_, D_, B_ = sess.run(
      [OPT_train,train_LOSS,train_lossper,Ftrain_NN,C,W,D,B]
      ,feed_dict={Ytrain_fit:ytrain_actual,Ftrain_fit:ftrain_actual})
    print('loss: %s '%(train_loss_))

C_ = np.array(C_)
W_ = np.array(W_)
D_ = np.array(D_)
B_ = np.array(B_)

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

Ftest_NN = myNNfunc(ytest_actual,C_,W_,D_,B_)
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
