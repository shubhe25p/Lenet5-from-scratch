import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


def pad(x,pad):
  x_pad=np.pad(x,((0,0),(pad,pad),(pad,pad),(0,0)),mode='constant',constant_values=(0,0))
  return x_pad

def conv_forward(img, filters, bias, pad, stride):
  m=img.shape[0]
  (m,nh,nw,ch_old)=img.shape
  (n,f,f,ch_old)=filters.shape
  newh=int((nh+2*pad-f)/stride+1)
  neww=int((nw+2*pad-f)/stride+1)
  Z=np.zeros((m,newh,neww,n))
  img=np.pad(img,((0,0),(pad,pad),(pad,pad),(0,0)),mode='constant',constant_values=(0,0))

  for i in range(m):
    img_i=img[i]
    for h in range(newh):

      hstart=stride*h
      hend=hstart+f
      for w in range(neww):
        
        wstart=stride*w
        wend=wstart+f

        for c in range(n):

          img_slice=img_i[hstart:hend,wstart:wend,:]
          Z[i,h,w,c]=np.sum(img_slice*filters[c,:,:,:])+float(bias[:,:,:,c])

          
  return Z

def pooling(X, pool_filter, mode, stride):


  (m,nh,nw,ch_old)=X.shape
  (m,f,f,ch_old)=pool_filter
  newh=int((nh-f)/stride+1)
  neww=int((nw-f)/stride+1)
  A=np.zeros((m,newh,neww,ch_old))
  for i in range(m):
    X_i=X[i]
    for h in range(newh):

      for w in range(neww):

        hstart=stride*h
        hend=hstart+f
        wstart=stride*w
        wend=wstart+f

        for c in range(ch_old):
          
          X_slice=X_i[hstart:hend,wstart:wend,c]
          if(mode=='max'):
            A[i,h,w,c]=np.max(X_slice)
          
          if(mode=='avg'):
            A[i,h,w,c]=np.mean(X_slice)


  return A

def tanh(Z):

  thx=(np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))

  return thx

def softmax(Z):

  sum=np.sum(np.exp(Z))
  sfx=np.exp(Z)/sum

  return sfx

def cnn_forward(X,W1,b1,W2,b2,W3,b3,W4,b4,W5,b5):

  X=pad(X, 2)
  (m,nh,nw,ch)=X.shape
  #1st-layer
  
  Z1=conv_forward(X, W1, b1, stride=1, pad=0)

  Z2=pooling(Z1, (m, 2, 2, 6), mode='avg', stride=2)
  #max-pool with pad
  #2nd-layer

  

  Z3=conv_forward(Z2,W2,b2,stride=1,pad=0)
  Z4=pooling(Z3, (m,2,2,16), mode='avg', stride=2)
  #3rd layer
  
  A0=Z4.reshape(m,-1)
  
  Z5=np.dot( W3.T, A0.T) + b3
  A1=tanh(Z5)
  #4th layer
  
  Z6=np.dot( W4.T, A1) + b4
  A2=tanh(Z6)

  #5th-layer
  
  Z7=np.dot( W5.T, A2) + b5
  A3=softmax(Z7)

  cache={
      "Z1":Z1,
      "W1":W1,
      "b1":b1,
      "Z2":Z2,
      "Z3":Z3,
      "W2":W2,
      "b2":b2,
      "Z4":Z4,
      "A0":A0,
      "W3":W3,
      "b3":b3,
      "Z5":Z5,
      "A1":A1,
      "W4":W4,
      "b4":b4,
      "Z6":Z6,
      "A2":A2,
      "W5":W5,
      "b5":b5,
      "Z7":Z7,
      "X":X
  }

  return A3, cache


def compute_cost(y_pred,y_true):

  cost=-np.sum(y_true*np.log(y_pred))

  return cost

def conv_back(dZ, cache):
  
  (A_prev,W, b, stride, pad)= cache
  
  (m,nhp,nwp,chp)=A_prev.shape
  (ch,f,f,chp)=W.shape
  (m,nh,nw,ch)= dZ.shape
  dA_prev=np.zeros((m,nhp,nwp,chp))
  dW=np.zeros((ch,f,f,chp))
  db=np.zeros((b.shape))

  #A_prev_pad=np.pad(A_prev,((0,0),(pad,pad),(pad,pad),(0,0)),mode='constant',constant_values=(0,0))
  #dA_prev_pad=np.pad(dA_prev,((0,0),(pad,pad),(pad,pad),(0,0)),mode='constant',constant_values=(0,0))


  for i in range(m):
    for h in range(nh):

      for w in range(nw):

        hstart=stride*h
        hend=hstart+f
        wstart=stride*w
        wend=wstart+f
        
        for c in range(ch):
          
          a_slice=A_prev[i,hstart:hend,wstart:wend,:]
          dA_prev[i,hstart:hend,wstart:wend,:]+=W[c,:,:,:]*dZ[i,h,w,c]
          dW[c,:,:,:]+=a_slice*dZ[i,h,w,c]
          db[:,:,:,c]+=dZ[i,h,w,c]
        
  return dA_prev, dW, db


def create_mask(x):
  mask=(x==np.max(x))

  return mask


def distribute_val(da, shape):

  (nh,nw)=shape
  average=float((da)/(nh*nw))

  d=np.full((nh,nw),average)

  return d

def pool_back(dZ, cache, mode):

  (A_prev , stride, f)=cache
  (m,nh_p,nw_p,ch_p)=A_prev.shape
  (n,nh,nw,ch)=dZ.shape

  dA_prev=np.zeros((m,nh_p,nw_p,ch_p))

  for i in range(n):

    A_prev_i=A_prev[i]

    for h in range(nh):

      for w in range(nw):

        hstart=stride*h
        hend=hstart+f
        wstart=stride*w
        wend=wstart+f

        for c in range(ch):


          if(mode=='max'):

            a_slice=A_prev_i[hstart:hend,wstart:wend,:]

            mask=create_mask(a_slice)

            dA_prev[i,hstart:hend,wstart:wend,c]+=mask*dZ[i,h,w,c]
          
          elif(mode=='avg'):

            val=dZ[i,h,w,c]
            dA_prev[i,hstart:hend,wstart:wend,c]+=distribute_val(val,(f,f))


  return dA_prev

def cnn_backprop(A3 , Y, cache):

  A2=cache["A2"]
  W5=cache["W5"]
  Z6=cache["Z6"]
  A1=cache["A1"]
  W4=cache["W4"]
  Z5=cache["Z5"]
  A0=cache["A0"]
  W3=cache["W3"]
  Z3=cache["Z3"]
  Z2=cache["Z2"]
  W2=cache["W2"]
  b2=cache["b2"]
  W1=cache["W1"]
  b1=cache["b1"]
  Z1=cache["Z1"]
  X=cache["X"]

  #FC backprop
  dZ7 = A3 - Y
  dW5 = np.dot( A2, dZ7.T)
  db5 = dZ7
  dA2 = np.dot(W5,dZ7)
  dZ6 = dA2*(1 - np.square(Z6))
  dW4 = np.dot(A1,dZ6.T)
  db4 = dZ6
  dA1 = np.dot(W4,dZ6)
  dZ5 = dA1*(1 - np.square(Z5))
  dW3 = np.dot(A0.T ,dZ5.T)
  db3 = dZ5
  dA0 = np.dot(W3,dZ5)

  dA0=dA0.reshape(10,5,5,16)
  #Pool_back

  dZ3 = pool_back(dA0, cache=(Z3,2,2), mode='avg')

  #Conv-layer
  dZ2, dW2, db2 = conv_back(dZ3, cache=(Z2,W2,b2,1,0))

  #Pool_layer
  dZ1 = pool_back(dZ2, cache=(Z1,2,2), mode='avg')

  #conv_layer
  dX, dW1, db1 = conv_back(dZ1, cache=(X,W1,b1,1,0))

  parameters={
      "dW1":dW1,
      "db1":db1,
      "dW2":dW2,
      "db2":db2,
      "dW3":dW3,
      "db3":db3,
      "dW4":dW4,
      "db4":db4,
      "dW5":dW5,
      "db5":db5,
  }
  return parameters


