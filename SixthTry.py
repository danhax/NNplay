#!/usr/local/bin/python3.6

#######   settings     #######

xrange     = 4
nper       = 10
ntrain    = 5000
ntest     = 5000

nTrainSteps = 1000

clen = 40

###

NUMC = 2           # number of convolutions
NUMP = 20          # number of polynomials before relu

startC = NUMC
startP = NUMP

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

def myrand(x):
  return np.exp(np.random.normal(x*0))

DORAND = False
if DORAND:
  funclist = [mysin,myquad,myrand]
else:
  funclist = [mysin,myquad]
nfunc    = len(funclist)
if DORAND:
  nfunctest = nfunc-1
else:
  nfunctest = nfunc

# get the positive-valued functions ydata(nx,ntrain)
#   with random parameters

def yfunc(ifuncs,xvals,xshift,xfac,yshift,yfac):
  # positive-valued function
  #
  ninput = ifuncs.size
  assert np.all(np.array(
    [xshift.size,xfac.size,yfac.size]) == ninput)
  assert np.all(ifuncs >= 0) and np.all(ifuncs < nfunc)
  yvals = np.zeros([nx,ninput])
  for isp in range(ninput):
    yvals[:,isp] = yshift[isp]**2 + \
      yfac[isp]**2 * funclist[ifuncs[isp]](xshift[isp] +
      xfac[isp] * xvals[:,0])
  return yvals

def getdata(functype):
  numvecs = functype.size
  xshift = np.random.normal(np.zeros(numvecs))*xrange
  xfac   = np.random.normal(np.zeros(numvecs))
  yshift = np.random.normal(np.zeros(numvecs))*10
  yfac   = np.random.normal(np.zeros(numvecs))
  ydata  = yfunc(functype,xdata,xshift,xfac,yshift,yfac)
  return ydata

######  random functions for train and test

# output to fit for training
Ftrain_fit = tf.placeholder(tf.float64, shape=(nfunc,ntrain))

# input
Ytrain_fit = tf.placeholder(tf.float64, shape=(nx,ntrain))

functype_train = np.random.choice(nfunc,ntrain)

functype_test  = np.random.choice(nfunctest,ntest)

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
  cnum = C.shape[1]

  inp=tf.reshape(tf.transpose(inp),(nvec,nx,1))
  C  =tf.reshape(C,                (clen,1,cnum))

  outp = tf.nn.conv1d(inp,C,1,'VALID')
  # have outp(nvec, nc, cnum)
  
  outp = tf.transpose(outp,perm=[1,0,2])
  # now have outp(nc,nvec,cnum)

  return outp

def myNNfunc(Im,inC,inT,inW,cnum,pnum):
  # Im(nx,nvec)
  # C(clen,cnum)         convolve Im
  # T(cnum,pnum)
  # W(pnum,nfunc)   linear transform polynomials to response functions
  # out(nfunc,nvec)      0 to 1
  
  nvec = Im.shape[1]

  Im = Im - tf.reduce_mean(Im,axis=(0));
  Im = Im / tf.sqrt(tf.reduce_mean(Im**2,axis=0))
  
  # Conved(nc,nvec,cnum)
  Conved = myconv(Im,inC)
  
  Terms = Conved;
  
  # T(cnum,pnum)  ->  Poly(nc,nvec,pnum)
  Poly = tf.tensordot(Terms,inT,axes=((2),(0)))

  Poly = tf.nn.relu(Poly)

  # Poly = tf.abs(Poly)
  
  # W(pnum,nfunc)  ->  Func(nc,nvec,nfunc)

  Func = tf.tensordot(Poly,inW,axes=((2),(0)))  

  Sigmoid = tf.nn.softmax(Func,2)

  Max = tf.reshape(tf.reduce_max(Sigmoid,axis=0),(nvec,nfunc))

  # Max(nfunc,nvec)
  Max = tf.transpose(Max)

  return Max

#
# NN PARAMETERS TO TRAIN
#       C, T, W
#

def DOIT(cnum,pnum,Cinit,Tinit,Winit):
  # Cinit(clen,cnum)
  # Tinit(cnum,pnum)
  # Winit(pnum,nfunc)

  C = tf.Variable(Cinit)
  T = tf.Variable(Tinit)
  W = tf.Variable(Winit)

  ######      TRAIN    ######

  Ftrain_NN = myNNfunc(Ytrain_fit,C,T,W,cnum,pnum)

  if 1==1:
    # error for each sample
    train_lossper = tf.reshape(
    #  tf.reduce_sum(tf.abs(Ftrain_fit-Ftrain_NN),axis=0),(1,ntrain))
      tf.reduce_sum((Ftrain_fit-Ftrain_NN)**2,axis=0),(1,ntrain))
    # summed over samples
    train_LOSS = tf.reduce_mean(train_lossper);
  else:
    train_lossper = tf.zeros((1,ntrain))
    train_LOSS = tf.losses.sparse_softmax_cross_entropy(
      functype_train,tf.transpose(Ftrain_NN))

  OPT_train = tf.train.AdamOptimizer(
    # learning_rate=0.001,beta1=0.9,beta2=0.9,epsilon=1e-8
    ).minimize(train_LOSS)

  # OPT_train = tf.train.GradientDescentOptimizer(
  #   learning_rate=0.01).minimize(train_LOSS)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_loss_ = sess.run(
      train_LOSS,
      feed_dict={Ytrain_fit:ytrain_actual,Ftrain_fit:ftrain_actual})
    print('LOSS: %s  step %i of %i'%(train_loss_,0,nTrainSteps))
    
    for istep in range(nTrainSteps):

      _, train_loss_, train_lossper_, \
        Ftrain_NN_, C_, T_, W_ = sess.run(
        [OPT_train,train_LOSS,train_lossper,Ftrain_NN,C,T,W]
        ,feed_dict={Ytrain_fit:ytrain_actual,Ftrain_fit:ftrain_actual})
      if np.mod(istep,100)==0:
        print('loss: %s  step %i of %i'%(train_loss_,istep,nTrainSteps))

  trainindex = np.arange(ntrain);
  trainindex = np.reshape(trainindex,(1,ntrain))

  besttrain = np.reshape(np.argmax(Ftrain_NN_,axis=0),(1,ntrain))

  ######      TEST    ######

  Ftest_NN = myNNfunc(ytest_actual,C_,T_,W_,cnum,pnum)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    Ftest_NN_ = sess.run(Ftest_NN)

  Ftest_NN_ = np.array(Ftest_NN_)

  testindex = np.arange(ntest);
  testindex = np.reshape(testindex,(1,ntest))

  # (nfunc,nvec)
  test_errorper = np.reshape(
    np.sum((ftest_actual-Ftest_NN_)**2,axis=0),(1,ntest))

  besttest = np.reshape(np.argmax(Ftest_NN_[0:nfunctest,:],axis=0),(1,ntest))

  noog = np.sum([functype_train[trainindex] != besttrain]) / ntrain
  print('TRAIN error rate: ',noog)
  noog = np.sum([functype_test[testindex] != besttest]) / ntest
  print(' TEST error rate: ',noog)

  plt.scatter(testindex,test_errorper)
  plt.show()
  plt.scatter(functype_test[testindex],test_errorper)
  plt.show()
  input('press enter')

  return C_, T_, W_

cnum = startC
pnum = startP

Cinit = np.random.normal(np.zeros((clen,cnum)))
Tinit = np.random.normal(np.zeros((cnum,pnum)))
Winit = np.random.normal(np.zeros((pnum,nfunc)))

doflag = True
while doflag :
    
  Cfinal, Tfinal, Wfinal = DOIT(cnum,pnum,Cinit,Tinit,Winit)

  doflag = cnum < NUMC or pnum < NUMP

  cprev = cnum
  pprev = pnum
  
  cnum = np.min((cnum+1,NUMC))
  pnum = np.min((pnum+1,NUMP))

  Cinit = np.zeros((clen,cnum))
  Tinit = np.zeros((cnum,pnum))
  Winit = np.zeros((pnum,nfunc))
  
  Cinit = np.random.normal(np.zeros((clen,cnum)))

  Cinit[:,0:cprev] = Cfinal
  Winit[0:pprev,:] = Wfinal

  Tinit[0:cprev,0:pprev] = Tfinal
  Tinit[0:cprev,pprev:pnum] = np.random.normal(np.zeros((cprev,pnum-pprev)))
  
exit()




