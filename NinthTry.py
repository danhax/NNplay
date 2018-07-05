#!/usr/local/bin/python3.6

##############################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

######### GLOBAL #########

xrange     = 4

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

def yfunc(funclist,ifuncs,xvals,xshift,xfac,yshift,yfac):
  # positive-valued function
  #
  nx = xvals.size
  ninput = ifuncs.size
  nfunc = ifuncs.size
  assert np.all(np.array(
    [xshift.size,xfac.size,yfac.size]) == ninput)
  assert np.all(ifuncs >= 0) and np.all(ifuncs < nfunc)
  yvals = np.zeros([nx,ninput])
  for isp in range(ninput):
    yvals[:,isp] = yshift[isp]**2 + \
      yfac[isp]**2 * funclist[ifuncs[isp]](xshift[isp] +
      xfac[isp] * xvals[:,0])
  return yvals

def getdata(xdata,functype,funclist):
  numvecs = functype.size
  
  xshift = np.random.normal(np.zeros(numvecs))*xrange
  xfac   = np.random.normal(np.zeros(numvecs))
  yshift = np.random.normal(np.zeros(numvecs))*10
  yfac   = np.random.normal(np.zeros(numvecs))
  ydata  = yfunc(funclist,functype,xdata,xshift,xfac,yshift,yfac)
  return ydata

#####   TENSORFLOW functions:

def myconv(inpinp,C):
  # input
  #   inp(nvec,nx)
  #   C(clen,cnum)
  # output
  #   outp(nc,nvec,cnum)

  inp = inpinp;  #  tf.Variable(inpinp)
  
  clen = C.shape[0]
  cnum = C.shape[1]

  nvec = inp.shape[0]
  nx   = inp.shape[1]
  inp=tf.reshape(inp,(nvec,nx,1))
  Q  =tf.reshape(C,  (clen,1,cnum))

  outp = tf.nn.conv1d(inp,Q,1,'VALID')
  # have outp(nvec, nc, cnum)
  
  return outp

def downsize(Im,nvec,nx,ndown):
  assert nvec == Im.shape[0]
  assert nx == Im.shape[1]
  nq = nx - ndown;               # must be odd
  assert np.mod(nq,2) == 1
  if ndown == 0 :
    Jm = Im  # tf.Variable(Im)
  else :
    nqh = int(np.round((nq-1)/2))
    imult = np.identity(nx)
    imult = np.concatenate(
      (imult[:,0:nqh],imult[:,(nqh+ndown):nx]),axis=1)
    imult = tf.cast(imult,tf.complex64)
    Jm = tf.cast(Im,tf.complex64)
    Jm = tf.transpose(tf.fft(tf.transpose(Jm)))
    Jm = tf.matmul(Jm,imult)
    Jm = tf.transpose(tf.ifft(tf.transpose(Jm)))
    Jm = tf.cast(Jm,tf.float64)
  return Jm

def downsizeYY(Im,nvec,nx,qnum,ndown):
  imList = []
  for iq in range(qnum):
    imList = imList + [downsize(Im,nvec,nx,iq*ndown)]
  return imList

def downsizeXX(inIm,nvec,nx,qnum,ndown):
  Im = inIm   # tf.Variable(inIm)
  assert nvec == Im.shape[0]
  assert nx == Im.shape[1]
  if qnum > 1 :
    ImFt = tf.cast(Im,tf.complex64)
    ImFt = tf.transpose(tf.fft(tf.transpose(ImFt)))
  for iq in range(qnum):
    nq = nx - iq*ndown;               # must be odd
    assert np.mod(nq,2) == 1
    if iq == 0 :
      ImList = [Im]
    else :
      nqh = int(np.round((nq-1)/2))
      imult = np.identity(nx)
      imult = np.concatenate(
        (imult[:,0:nqh],imult[:,(nqh+iq*ndown):nx]),axis=1)
      imult = tf.cast(imult,tf.complex64)
      Jm = tf.matmul(ImFt,imult)
      Km = tf.transpose(tf.ifft(tf.transpose(Jm)))
      ImList = ImList + [ tf.cast(Km,tf.float64) ]
  return ImList

def upsize(Im,nvec,nx,ndown):
  ncopy = Im.shape[2]
  Jm = []
  for icopy in range(ncopy):
    Jm = Jm + [tf.reshape(upsize0(Im[:,:,icopy],nvec,nx,ndown),(nvec,nx,1))]
  Km = tf.concat(Jm,2)
  return Km

def upsize0(Im,nvec,nc,ndown):
  assert nvec == Im.shape[0]
  nq = nc - ndown;               # must be odd
  assert nq == Im.shape[1]
  assert np.mod(nq,2) == 1
  if ndown == 0 :
    Km = Im   # tf.Variable(Im)
  else :
    nqh = int(np.round((nq-1)/2))
    imult = np.identity(nc)
    imult = np.concatenate(
      (imult[:,0:nqh],imult[:,(nqh+ndown):nc]),axis=1)
    imult = tf.cast(np.transpose(imult),tf.complex64)
    Jm = tf.cast(Im,tf.complex64)
    Jm = tf.transpose(tf.fft(tf.transpose(Jm)))
    Jm = tf.matmul(Jm,imult)
    Jm = tf.transpose(tf.ifft(tf.transpose(Jm)))
    Km = tf.cast(Jm,tf.float64)
  return Km

def upsizeXX(ImList,nvec,nc,qnum,ndown):
  # 
  assert qnum == len(ImList)
  # each image in imlist dimension (nc,nvec,cnum)
  ncopy = ImList[0].shape[2]
  OutList = []
  for iq in range(qnum) :
    Im = ImList[iq]   # tf.Variable(ImList[iq])
    Jm = []
    for icopy in range(ncopy):
      Jm = Jm + [tf.reshape(upsize0(Im[:,:,icopy],nvec,nc,iq*ndown),(nvec,nc,1))]
    Km = tf.concat(Jm,2)
    OutList = OutList + [Km]
  return OutList

def myNNfunc(Im,inC,inT,inW,nvec,nx,clen,qstep):
  # Im(nvec,nx)
  # C(clen,cnum)         convolve Im
  # T(cnum,pnum)
  # W(qnum,pnum,nfunc)   linear transform polynomials to response functions
  # out(nfunc,nvec)      0 to 1

  assert nvec == Im.shape[0]
  assert nx == Im.shape[1]
  assert clen == inC.shape[0]
  cnum = inC.shape[1]
  pnum = inT.shape[1]
  nfunc = inW.shape[2]
  qnum = inW.shape[0]

  assert pnum == inW.shape[1]
  
  nc = nx + 1 - clen

  Jm = Im - tf.reshape(tf.reduce_mean(Im,axis=(1)),(nvec,1))
  Jm = Jm / tf.reshape(tf.sqrt(tf.reduce_mean(Jm**2,axis=1)),(nvec,1))

  ndown = np.mod(nx,2) + 1 + 2*qstep

  mindim = nc - (qnum-1)*ndown

  assert  mindim          >  0
  assert  np.mod(nx,2)    == 1
  assert  np.mod(ndown,2) == 0

  if 1==1 :
    imList = []
    for iq in range(qnum):
      imList = imList + [downsize(Jm,nvec,nx,iq*ndown)]
      
  else:
    imList = downsizeXX(Jm,nvec,nx,qnum,ndown)

  if 1==1 :
    Conved =[]
    for iq in range(qnum):
      # Conved(nvec,nc,cnum)
      thisconv = myconv(imList[iq],inC)
      Conved = Conved + [ upsize(thisconv, nvec,nc,iq*ndown) ]
      #
      # Conved = Conved + [ tf.reshape( upsize(
      #    thisconv, nvec,nc,iq*ndown), (nvec,nc,1,cnum))]
  else:
    thisconv =[]
    for iq in range(qnum) :
      thisconv = thisconv + [myconv(imList[iq],inC)]
    # have Conved[iq] dimension (nvec,nc,cnum)
    Conved = upsizeXX(thisconv, nvec,nc,qnum,ndown)

  Reshaped = []
  for iq in range(qnum) :
    Reshaped = Reshaped + [tf.reshape( Conved[iq], (nvec,nc,1,cnum))]

  # Terms(nvec,nc,qnum,cnum)
  Terms = tf.concat(Reshaped,2)
  
  # T(cnum,pnum)  ->  Poly(nvec,nc,qnum,pnum)
  Poly = tf.tensordot(Terms,inT,axes=((3),(0)))
  if 1==0:
    Poly = tf.reduce_sum(Poly,axis=(2),keepdims=True)
    qnum = 1

  Poly = tf.nn.relu(Poly)

  # W(qnum,pnum,nfunc)  ->  Func(nvec,nc,nfunc)
  Func = tf.tensordot(Poly,inW,axes=((2,3),(0,1)))  

  Func = tf.reshape(Func,(nvec,nfunc*nc))
  Sigmoid = tf.nn.softmax(Func,1)
  Sigmoid = tf.reshape(Sigmoid,(nvec,nc,nfunc))
  Max = tf.reshape(tf.reduce_sum(Sigmoid,axis=(1)),(nvec,nfunc))

  # Max(nfunc,nvec)
  Max = tf.transpose(Max)

  return Max

def DOIT(nTrainSteps,cnum,pnum,qnum,Cinit,Tinit,Winit,
         Y_fit,ytrain_actual,ytest_actual,
         F_fit,ftrain_actual,ftest_actual,
         functype_train,functype_test,nvec,nx,clen,qstep):
  # Cinit(clen,cnum)
  # Tinit(cnum,pnum)
  # Winit(qnum,qnum,nfunc)

  nvec = functype_train.size
  nfunc = Winit.shape[2]
  
  C = tf.Variable(Cinit)
  T = tf.Variable(Tinit)
  W = tf.Variable(Winit)

  ######      TRAIN    ######

  F_NN = myNNfunc(Y_fit,C,T,W,nvec,nx,clen,qstep)
  
  # error for each sample
  t_lossper = tf.reshape(
  #  tf.reduce_sum(tf.abs(F_fit-F_NN),axis=0),(1,nvec))
    tf.reduce_sum((F_fit-F_NN)**2/2,axis=0),(1,nvec))
  # summed over samples
  t_LOSS = tf.reduce_mean(t_lossper);

  OPT = tf.train.AdamOptimizer(
    learning_rate=0.0002,beta1=0.9,beta2=0.99,
    ).minimize(t_LOSS)

  # OPT = tf.train.GradientDescentOptimizer(
  #   learning_rate=0.001).minimize(t_LOSS)
  
  # OPT = tf.train.RMSPropOptimizer(
  #   learning_rate=0.0003).minimize(t_LOSS)

  
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for istep in range(nTrainSteps+1):

      _, train_loss, train_lossper, \
        Ftrain_NN, C_, T_, W_ = sess.run(
        [OPT,t_LOSS,t_lossper,F_NN,C,T,W]
        ,feed_dict={Y_fit:ytrain_actual,F_fit:ftrain_actual})

      if np.mod(istep,100)==0:
        test_loss, test_lossper, Ftest_NN_ = sess.run(
          [t_LOSS, t_lossper, F_NN]
          ,feed_dict={Y_fit:ytest_actual,F_fit:ftest_actual})
        
        besttrain = np.reshape(
          np.argmax(Ftrain_NN,axis=0),(1,nvec))
        besttest = np.reshape(
          np.argmax(Ftest_NN_[0:nfunc,:],axis=0),(1,nvec))
        train_error = np.sum(
          [functype_train != besttrain]) / nvec
        test_error = np.sum(
          [functype_test != besttest]) / nvec
        
        print(' step %i of %i  loss %.7s %.7s  errorRate %.7s %.7s'%(
          istep,nTrainSteps,train_loss,
          test_loss,train_error,test_error))


  plt.scatter(np.arange(nvec),test_lossper)
  plt.show()
  plt.scatter(functype_test,test_lossper)
  plt.show()
  input('#press enter')

  return C_, T_, W_

######  END FUNCTIONS #########

def main():
  
  #######   settings     #######

  nTrainSteps = 50000

  nvec       = 5000
  #nper       = 5
  #clen       = 7
  nper       = 10
  clen       = 5
  # clen       = 41

  qnum  = 3
  qstep = 8
  
  NUMC = 2           # number of convolutions
  NUMP = 20          # number of polynomials before relu

  startC = NUMC
  startP = NUMP

  ###### x-values at which to evaluate the functions

  nx    = xrange * nper + 1
  xdata = np.arange(nx)/nper
  xdata = np.reshape(xdata,(nx,1))

  nc = nx + 1 - clen;

  ######  random functions for train and test
  # get the positive-valued functions ydata(nx,nvec)
  #   with random parameters

  # funclist = [mysin,myquad,myexp]
  funclist = [mysin,myquad]
  nfunc    = len(funclist)

  # output
  F_fit = tf.placeholder(tf.float64, shape=(nfunc,nvec))
  # input
  Y_fit = tf.placeholder(tf.float64, shape=(nvec,nx))

  functype_train = np.random.choice(nfunc,nvec)
  functype_test  = np.random.choice(nfunc,nvec)

  ftrain_actual          = np.zeros([nfunc,nvec])
  for ivec in range(nvec):
    ftrain_actual[functype_train[ivec],ivec] = 1

  ftest_actual          = np.zeros([nfunc,nvec])
  for ivec in range(nvec):
    ftest_actual[functype_test[ivec],ivec] = 1

  ytrain_actual = getdata(xdata,functype_train,funclist)
  ytest_actual  = getdata(xdata,functype_test,funclist)
  ytrain_actual = np.transpose(ytrain_actual)
  ytest_actual  = np.transpose(ytest_actual)

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
  Winit = np.random.normal(np.zeros((qnum,pnum,nfunc)))

  doflag = True
  while doflag :

    Cfinal, Tfinal, Wfinal = DOIT(
      nTrainSteps,cnum,pnum,qnum,Cinit,Tinit,Winit,
      Y_fit,ytrain_actual,ytest_actual,
      F_fit,ftrain_actual,ftest_actual,
      functype_train,functype_test,nvec,nx,clen,qstep)
    doflag = cnum < NUMC or pnum < NUMP

    cprev = cnum
    pprev = pnum

    cnum = np.min((cnum+1,NUMC))
    pnum = np.min((pnum+1,NUMP))

    Cinit = np.zeros((clen,cnum))
    Tinit = np.zeros((cnum,pnum))
    Winit = np.zeros((qnum,pnum,nfunc))

    Cinit = np.random.normal(np.zeros((clen,cnum)))
    # Winit = np.random.normal(np.zeros((qnum,pnum,nfunc)))

    Cinit[:,0:cprev] = Cfinal
    Winit[:,0:pprev,:] = Wfinal

    Tinit[0:cprev,0:pprev] = Tfinal
    # Tinit[0:cprev,pprev:pnum] =
    #   np.random.normal(np.zeros((cprev,pnum-pprev)))
    Tinit[:,pprev:pnum] = np.random.normal(np.zeros((cnum,pnum-pprev)))
  
main()
exit()





