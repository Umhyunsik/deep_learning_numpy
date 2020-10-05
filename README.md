# deep_learning_numpy_구현

# 1. Mnist Sigmoid,Relu

파일명 : Mnist.py
  
    두 모델중 선택 
    
    network = ThirdLayerNet(input_size=784, hidden_size1=50,hidden_size2=10,hidden_size3=40, output_size=10)
    #network = ThirdLayerRelu()
  
Sigmoid는 third_layer_net.py 모델사용

순방향으로 짰을 때 은닉층의 노드수가 너무 많아 거의 컴퓨터가 멈춰있듯이 계산 처리가 느림.
따라서 오차 역전파를 했지만, sigmoid의 오차역전파 역시 학습이 진행이 되지않아 처리할 수 없었다.


따라서 third_layer_relu.py 모델 사용

Epoch가 6000번부터 loss가 급격히 떨어지며 학습이 진행되고 오차역전파를 쓰지않으면 수치미분방식이 너무 느리게 진행된다.
network.numerical_gradient 가 순방향 , network.gradient 가 오차역전파 비교하여 test해보면됨.


    # 기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    #grad = network.gradient(x_batch, t_batch) # 오차역전파법 방식(훨씬 빠르다)
    
<img src="https://user-images.githubusercontent.com/22265915/95035655-b69bf280-0700-11eb-8a94-b8c4cc539f17.png" width="50%">
  
### train loss ###

<img src="https://user-images.githubusercontent.com/22265915/95035666-c0bdf100-0700-11eb-9a9f-60db67b0fda5.png" width="50%">

### test loss ###




# 2. Playground classification

<img width="1340" alt="classification_circle" src="https://user-images.githubusercontent.com/22265915/95034847-10e78400-06fe-11eb-9d5f-579d16538e29.png">

## 실제 Playground 원 안, 밖 classification 이 되는지 numpy로 구현 ##

![image](https://user-images.githubusercontent.com/22265915/95036405-cd434900-0702-11eb-80ce-66164afd10fc.png)

linear classification은 가장 왼쪽것을 제외하고 분류하지 못한다.
세번째 원 안,밖을 분류하지 못하기때문에 다층신경망이 해결할 수 있는지에 대한 test 

Sigmoid ,Relu 두 모델 사용

    x , y 좌표 -4 , 4 사이에 랜덤으로 생성
    s = np.random.uniform(-4,4,11000)
    y = np.random.uniform(-4,4,11000)

    ordered_pairs = np.column_stack((s, y))
    x_train=ordered_pairs[0:10000]#train 문제
    x_test=ordered_pairs[10000:11000]#test문제

    print(ordered_pairs.shape)
    a=[]
    
    # 정답데이터 생성
    # 원 안이면 [1,0] 그 외 점 [0,1]
    
    for x ,y in ordered_pairs:
        if(x**2+y**2<=2):
            a.append([1,0])

        else:
            a.append([0,1])


third_layer_net.py 모델사용

Activation을 Sigmoid로 시작해봤지만 exp() 함수를 사용하여 계산속도가 상당히 느리다.
또한 sigmoid는 극단값들에대해 gradient가 0 으로 수렴하여 학습이 진행 되지않았다고 생각한다. 
실제로, playground 에서도 sigmoid로 진행하면 학습의 시간이 정말 오래걸린다.


따라 RELU로 변경
third_layer_relu.py 모델 사용

순방향, 오차역전파 를 사용한결과 오차역전파가 훨씬 빠르게 학습됐다.
이론으로만 봤던 것을 실제 속도차이를 실감하니 놀라움.
아무래도 exp() 함수를 사용하지않고, 0 보다 작은값들에 에대해 gradient가 0이 되는 문제가있지만, 상대적으로 sigmoid의 양끝 값에서 발생하는 것보다는 적다는 것을 고려해봤을때, 속도면에서 더 빠르게 학습된것이 아닐까 생각한다. 

은닉층의 node 개수를 조절해보고 평가했을때 30 , 5 , 10 이 가장 빠르게 학습되는것을 볼 수 있었다.

<img width="700" alt="스크린샷 2020-10-05 오후 12 17 50" src="https://user-images.githubusercontent.com/22265915/95037082-d208fc80-0704-11eb-87fa-015711273e5f.png">

### loss graph ###
위 train 아래 test 


            


       
