# coding: utf-8
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from dataset.mnist import load_mnist
from third_layer_net import ThirdLayerNet
import matplotlib.pyplot as plt
# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = ThirdLayerNet(input_size=784, hidden_size1=50,hidden_size2=10,hidden_size3=40, output_size=10)
from third_layer_relu import *
#network = ThirdLayerRelu()

#iters_num = 10000
train_size = x_train.shape[0]
test_size=x_test.shape[0]
batch_size = 153
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []
test_loss_list=[]

iter_per_epoch =max(train_size / batch_size, 1)
iter_per_epoch=int(iter_per_epoch)
i=0

while(True):
    batch_mask = np.random.choice(train_size, batch_size)#10000 153
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    batch_mask2 = np.random.choice(test_size, batch_size)
    x_test_batch=x_test[batch_mask2]
    t_test_batch=t_test[batch_mask2]
    
    # 기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    #grad = network.gradient(x_batch, t_batch) # 오차역전파법 방식(훨씬 빠르다)
    
    # 갱신
    for key in ('W1', 'b1', 'W2', 'b2','W3','b3','W4','b4'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    testloss=network.loss(x_test_batch,t_test_batch)
    test_loss_list.append(testloss)
    i+=1
    #print(i)
    
    if i % iter_per_epoch == 0:
        #print(0)
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        #train_loss_list.append(loss)
        
        print(train_acc, test_acc,loss)
        if loss<=0.01:
             #print(train_acc, test_acc,loss)
             break
plt.figure(1)
plt.plot( train_loss_list, label='train loss ', linestyle='-.')

plt.xlabel("epochs")
plt.ylabel("train loss")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')

plt.figure(2)
plt.plot( test_loss_list, label='test loss ', linestyle='-.')

plt.xlabel("epochs")
plt.ylabel("test loss")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
