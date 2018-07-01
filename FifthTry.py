#!/usr/local/bin/python3.6

#######   settings     #######

xrange     = 4
nper       = 10
ntrain    = 2000
ntest     = 2000

nTrainSteps = 10000

clen = 39

##############################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

###### x-values at which to evaluate the functions

nx    = xrange * nper
xdata = np.arange(xrange*nper)/nper
xdata = np.reshape(xdata,(nx,1))

nc = nx + 1 - clen;

######  positive-valued functions

def myquad(x):
  return x/xrange + np.power(x/xrange,2)

def mysin(x):
  return np.sin(16.28*x/xrange)**2

funclist = [mysin,myquad]
nfunc    = len(funclist)

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

# output to fit for training
Ftrain_fit = tf.placeholder(tf.float64, shape=(nfunc,ntrain))

# input
Ytrain_fit = tf.placeholder(tf.float64, shape=(nx,ntrain))

functype_train = np.random.choice(nfunc,ntrain)
functype_test  = np.random.choice(nfunc,ntest)

functype_train = np.reshape(functype_train,(ntrain,))

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

#####   TENSORFLOW functions:

def myconv(inp,C):

  # want NWC inp(nvec, inlen, 1)    nbatch=nvec  width=inlen
  #          C   clen,1,cnum)
  # input
  #   inp(nx,nvec)
  #   C(clen,cnum)
  # output
  #   outp(nc,nvec,cnum)

  nvec = inp.shape[1]

  inp=tf.reshape(tf.transpose(inp),(nvec,nx,1))
  C  =tf.reshape(C,                (clen,1,cnum))

  outp = tf.nn.conv1d(inp,C,1,'VALID')
  # have outp(nvec, nc, cnum)
  
  outp = tf.transpose(outp,perm=[1,0,2])
  # now have outp(nc,nvec,cnum)

  return outp

def myNNfunc(Im,C,T,W,cnum,mnum,pnum):
  # Im(nx,nvec)
  # C(clen,cnum)         convolve Im
  # T(mnum**cnum,pnum)
  # W(pnum,nfunc)   linear transform polynomials to response functions
  #                          nterm = mnum**cnum
  # out(nfunc,nvec)      0 to 1
  
  nvec = Im.shape[1]

  Im = Im - tf.reduce_mean(Im,axis=(0));
  Im = Im / tf.sqrt(tf.reduce_sum(Im**2,axis=0))

  # Conved(nc,nvec,cnum)
  Conved = myconv(Im,C)

  Terms = tf.reduce_sum(
    tf.reshape(Conved,(nc,nvec,cnum,1)) ** \
    tf.reshape(MonPwr,(1,1,cnum,nterm)), axis=(2))

  # Terms(nc,nvec,nterm)
  Terms = tf.reshape( Terms, (nc,nvec,nterm) )

  # T(nterm,pnum)  ->  Poly(nc,nvec,pnum)
  Poly = tf.tensordot(Terms,T,axes=((2),(0)))

  Poly = tf.nn.relu(Poly)

  # W(pnum,nfunc)  ->  Func(nc,nvec,nfunc)

  Func = tf.tensordot(Poly,W,axes=((2),(0)))  

  Sigmoid = tf.nn.softmax(Func,2)

  Max = tf.reshape(tf.reduce_max(Sigmoid,axis=0),(nvec,nfunc))

  # Max(nfunc,nvec)
  Max = tf.transpose(Max)

  return Max

#
# NN PARAMETERS TO TRAIN
#

cnum = 3            # number of convolutions
mnum = 3            # highest power each convolution plus one
pnum = 4            # number of polynomials before relu

nterm = mnum**cnum  # total number of polynomial terms

# MonPwr(cnum,nterm)
MonPwr = np.mod( np.floor(
  np.reshape( np.arange(nterm),     (1,nterm) ) /
  np.reshape( mnum**np.arange(cnum), (cnum,1) ) ), mnum)

C = tf.get_variable('C',(clen,cnum),
              dtype=tf.float64,
              initializer = tf.random_normal_initializer)

T = tf.get_variable('T',(nterm,pnum),
              dtype=tf.float64,
              initializer = tf.random_normal_initializer)

W = tf.get_variable('W',(pnum,nfunc),
              dtype=tf.float64,
              initializer = tf.random_normal_initializer)

# tt = np.zeros((nterm,1))
# tt[0,0] = 1
# Winit = tt * np.ones(nfunc)
# W = tf.get_variable('W',dtype=tf.float64,
#        initializer = tf.constant(Winit))
# #        initializer = tf.zeros((nterm,nfunc),dtype=tf.float64))


######      TRAIN    ######

Ftrain_NN = myNNfunc(Ytrain_fit,C,T,W,cnum,mnum,pnum)

if 1==1:
  # error for each sample
  train_lossper = tf.reshape(
    tf.reduce_sum(tf.abs(Ftrain_fit-Ftrain_NN),axis=0),(1,ntrain))
  #  tf.reduce_sum((Ftrain_fit-Ftrain_NN)**2,axis=0),(1,ntrain))
  # summed over samples
  train_LOSS = tf.reduce_mean(train_lossper);
else:
  train_lossper = tf.zeros((1,ntrain))
  train_LOSS = tf.losses.sparse_softmax_cross_entropy(
    functype_train,tf.transpose(Ftrain_NN))

OPT_train = tf.train.AdamOptimizer().minimize(train_LOSS)

# OPT_train = tf.train.GradientDescentOptimizer(
#   learning_rate=0.01).minimize(train_LOSS)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for istep in range(nTrainSteps):
    
    _, train_loss_, train_lossper_, \
      Ftrain_NN_, C_, T_, W_ = sess.run(
      [OPT_train,train_LOSS,train_lossper,Ftrain_NN,C,T,W]
      ,feed_dict={Ytrain_fit:ytrain_actual,Ftrain_fit:ftrain_actual})
    if np.mod(istep,100)==0:
      print('loss: %s  step %i of %i'%(train_loss_,istep,nTrainSteps))

C_ = np.array(C_)
W_ = np.array(W_)

trainindex = np.arange(ntrain);
trainindex = np.reshape(trainindex,(1,ntrain))

besttrain = np.reshape(np.argmax(Ftrain_NN_,axis=0),(1,ntrain))

print('TRAIN: actual, bestguess, error')
print(np.array2string(np.transpose(np.concatenate(
  (functype_train[trainindex], besttrain, train_lossper_))),
  formatter={'float_kind':lambda x: '%.4f' % x}))

# plt.scatter(trainindex,train_lossper_)
# plt.show()
# plt.scatter(functype_train[trainindex],train_lossper_)
# plt.show()

######      TEST    ######

Ftest_NN = myNNfunc(ytest_actual,C_,T_,W_,cnum,mnum,pnum)
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

noog = np.sum([functype_train[trainindex] != besttrain]) / ntrain
print('TRAIN error rate: ',noog)
noog = np.sum([functype_test[testindex] != besttest]) / ntest
print(' TEST error rate: ',noog)

plt.scatter(testindex,test_errorper)
plt.show()
plt.scatter(functype_test[testindex],test_errorper)
plt.show()


