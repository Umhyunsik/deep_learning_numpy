import sys, os
import math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import numpy as np 
from itertools import product
s = np.random.uniform(-4,4,11000)
y = np.random.uniform(-4,4,11000)

ordered_pairs = np.column_stack((s, y))
x_train=ordered_pairs[0:10000]#train 문제
x_test=ordered_pairs[10000:11000]#test문제

print(ordered_pairs.shape)
a=[]
for x ,y in ordered_pairs:
    if(x**2+y**2<=2):
        a.append([1,0])
       
    else:
        a.append([0,1])
       

a = np.asarray(a, dtype=np.float32)
t_train=a[0:10000]#train 정답지

t_test=a[10000:11000]#test정답지 
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
#x_test.flatten()
#t_test.flatten()
# coding: utf-8


# 데이터 읽기
from third_layer_relu import *
network = ThirdLayerRelu()


# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0]
test_size=x_test.shape[0]
batch_size = 100  # 미니배치 크기
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []
test_loss_list=[]

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)
i=0
while(True):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    batch_mask2 = np.random.choice(test_size, batch_size)
    x_test_batch=x_test[batch_mask2]
    t_test_batch=t_test[batch_mask2]
    
    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2','W3','b3','W4','b4'):
        network.params[key] -= learning_rate * grad[key]
    
    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    testloss=network.loss(x_test_batch,t_test_batch)
    train_loss_list.append(loss)
    test_loss_list.append(testloss)

    i+=1
    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc),loss,testloss)
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
