#!/usr/local/bin/python3.6

##############################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

############# FUNCTIONS ################

######  positive-valued functions

def myquad(x):
  return x/xrange + np.power(x/xrange,2)

def mysin(x):
  return np.sin(16.28*x/xrange)**2

def myexp(x):
  return np.exp(2*x/xrange)

def myrand(x):
  return np.exp(np.random.normal(x*0))

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

#####   TENSORFLOW functions:

def myconv(inp,C):

  # input
  #   inp(nvec,nx)
  #   C(clen,cnum)
  # output
  #   outp(nc,nvec,cnum)

  cnum = C.shape[1]

  nvec = inp.shape[0]
  nx   = inp.shape[1]
  inp=tf.reshape(inp,(nvec,nx,1))
  C  =tf.reshape(C,  (clen,1,cnum))

  outp = tf.nn.conv1d(inp,C,1,'VALID')
  # have outp(nvec, nc, cnum)
  
  return outp

def myNNfunc(Im,inC,inT,inW):
  # Im(nvec,nx)
  # C(clen,cnum)         convolve Im
  # T(cnum,pnum)
  # W(pnum,nfunc)   linear transform polynomials to response functions
  # out(nfunc,nvec)      0 to 1
  
  nvec = Im.shape[0]
  nx = Im.shape[1]
  clen = inC.shape[0]
  cnum = inC.shape[1]
  pnum = inT.shape[1]
  nfunc = inW.shape[1]
  nc = nx + 1 - clen
  
  Im = Im - tf.reshape(tf.reduce_mean(Im,axis=(1)),(nvec,1))
  Im = Im / tf.reshape(tf.sqrt(tf.reduce_mean(Im**2,axis=1)),(nvec,1))

  imList = [Im]
  
  # Conved(nvec,nc,cnum)
  Conved = myconv(Im,inC)
  
  Terms = Conved;
  
  # T(cnum,pnum)  ->  Poly(nvec,nc,pnum)
  Poly = tf.tensordot(Terms,inT,axes=((2),(0)))

  Poly = tf.nn.relu(Poly)

  # W(pnum,nfunc)  ->  Func(nvec,nc,nfunc)
  Func = tf.tensordot(Poly,inW,axes=((2),(0)))  

  Func = tf.reshape(Func,(nvec,nfunc*nc))
  Sigmoid = tf.nn.softmax(Func,1)
  Sigmoid = tf.reshape(Sigmoid,(nvec,nc,nfunc))
  Max = tf.reshape(tf.reduce_sum(Sigmoid,axis=1),(nvec,nfunc))

  # Max(nfunc,nvec)
  Max = tf.transpose(Max)

  return Max

def DOIT(cnum,pnum,Cinit,Tinit,Winit):
  # Cinit(clen,cnum)
  # Tinit(cnum,pnum)
  # Winit(pnum,nfunc)

  C = tf.Variable(Cinit)
  T = tf.Variable(Tinit)
  W = tf.Variable(Winit)

  ######      TRAIN    ######

  Ftrain_NN = myNNfunc(Y_fit,C,T,W)

  # error for each sample
  train_lossper = tf.reshape(
  #  tf.reduce_sum(tf.abs(F_fit-Ftrain_NN),axis=0),(1,nvec))
    tf.reduce_sum((F_fit-Ftrain_NN)**2/2,axis=0),(1,nvec))
  # summed over samples
  train_LOSS = tf.reduce_mean(train_lossper);

  OPT = tf.train.AdamOptimizer(
    learning_rate=0.0002,beta1=0.9,beta2=0.99,
    ).minimize(train_LOSS)

  # OPT = tf.train.GradientDescentOptimizer(
  #   learning_rate=0.001).minimize(train_LOSS)
  
  # OPT = tf.train.RMSPropOptimizer(
  #   learning_rate=0.0003).minimize(train_LOSS)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for istep in range(nTrainSteps+1):

      _, train_loss_, train_lossper_, \
        Ftrain_NN_, C_, T_, W_ = sess.run(
        [OPT,train_LOSS,train_lossper,Ftrain_NN,C,T,W]
        ,feed_dict={Y_fit:ytrain_actual,F_fit:ftrain_actual})
      if np.mod(istep,100)==0:
        test_loss_, test_errorper, Ftest_NN_ = sess.run(
          [train_LOSS, train_lossper, Ftrain_NN]
          ,feed_dict={Y_fit:ytest_actual,F_fit:ftest_actual})
        
        besttrain = np.reshape(np.argmax(Ftrain_NN_,axis=0),(1,nvec))
        besttest = np.reshape(np.argmax(Ftest_NN_[0:nfunctest,:],axis=0),(1,nvec))
        train_error = np.sum([functype_train != besttrain]) / nvec
        test_error = np.sum([functype_test != besttest]) / nvec
        
        print(' step %i of %i  loss %.7s %.7s  errorRate %.7s %.7s'%(
          istep,nTrainSteps,train_loss_,test_loss_,train_error,test_error))


  plt.scatter(np.arange(nvec),test_errorper)
  plt.show()
  plt.scatter(functype_test,test_errorper)
  plt.show()
  input('#press enter')

  return C_, T_, W_

######  END FUNCTIONS #########


#######   settings     #######

xrange     = 4
nper       = 10
nvec       = 5000

nTrainSteps = 5000

clen = 12

###

NUMC = 4           # number of convolutions
NUMP = 20          # number of polynomials before relu

startC = NUMC
startP = NUMP

###### x-values at which to evaluate the functions

nx    = xrange * nper
xdata = np.arange(xrange*nper)/nper
xdata = np.reshape(xdata,(nx,1))

nc = nx + 1 - clen;

# get the positive-valued functions ydata(nx,nvec)
#   with random parameters

DORAND = False
if DORAND:
  funclist = [mysin,myquad,myexp,myrand]
else:
  funclist = [mysin,myquad,myexp]
nfunc    = len(funclist)
if DORAND:
  nfunctest = nfunc-1
else:
  nfunctest = nfunc


######  random functions for train and test

# output to fit for training
F_fit = tf.placeholder(tf.float64, shape=(nfunc,nvec))

# input
Y_fit = tf.placeholder(tf.float64, shape=(nvec,nx))

functype_train = np.random.choice(nfunc,nvec)

functype_test  = np.random.choice(nfunctest,nvec)

ftrain_actual          = np.zeros([nfunc,nvec])
for ivec in range(nvec):
  ftrain_actual[functype_train[ivec],ivec] = 1

ftest_actual          = np.zeros([nfunc,nvec])
for ivec in range(nvec):
  ftest_actual[functype_test[ivec],ivec] = 1

ytrain_actual = getdata(functype_train)
ytest_actual  = getdata(functype_test)

ytrain_actual = np.transpose(ytrain_actual)
ytest_actual = np.transpose(ytest_actual)

# for ivec in range(nvec):
#   plt.scatter(xdata,ytrain_actual[:,ivec])
# plt.show()

#
# NN PARAMETERS TO TRAIN
#       C, T, W
#

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
  # Winit = np.random.normal(np.zeros((pnum,nfunc)))

  Cinit[:,0:cprev] = Cfinal
  Winit[0:pprev,:] = Wfinal

  Tinit[0:cprev,0:pprev] = Tfinal
  # Tinit[0:cprev,pprev:pnum] = np.random.normal(np.zeros((cprev,pnum-pprev)))
  Tinit[:,pprev:pnum] = np.random.normal(np.zeros((cnum,pnum-pprev)))
  
exit()




