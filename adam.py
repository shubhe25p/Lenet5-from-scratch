import numpy as np
import matplotlib.pyplot as plt



def new_func():
    temp=30+np.random.randint(0,10,size=100)
    v_prev=0
    v_new=0
    v=[]
    for i in range(100):
        v_new=0.9*v_prev+0.1*temp[i]
        v.append(v_new)
        v_prev=v_new
    
    print(np.max(v))
new_func()