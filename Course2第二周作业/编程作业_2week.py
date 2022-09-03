# -*- coding: utf-8 -*-
# @Time : 2022/9/3 10:40
# @Author : zhuyu
# @File : 编程作业_2week.py
# @Project : Python菜鸟教程

"""
1.分割数据集
2.优化梯度下降算法：
    2.1 不使用任何优化算法的梯度下降法(批量梯度下降法)
    2.2 mini-batch梯度下降法
    2.3 momentum梯度下降法
    2.4 Adam算法
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

import opt_utils
import testCase

#plt.rcParams设置图像细节
plt.rcParams["figure.figsize"]=(7,4)
plt.rcParams["image.interpolation"]="nearest"
plt.rcParams["image.cmap"]="gray"

#无任何优化的梯度下降(批量梯度下降法)
def update_parameters_with_gd(parameters,grads,learning_rate):
    """
    批量梯度下降法
    :param parameters:字典 - 包含要更新的参数
    :param grads:  字典 - 包含待更新参数的梯度
    :param learning_rate:  学习率
    :return:
    """
    L=len(parameters)//2 #神经网络层数

    for l in range(L):
        parameters["W"+str(l+1)]=parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]
        parameters["b"+str(l+1)]=parameters["b"+str(l+1)]-learning_rate*grads["db"+str(l+1)]

    return parameters

#测试update_parameters_with_gd
print("-"*30+"测试update_parameters_with_gd"+"-"*30)
parameters,grads,learning_rate=testCase.update_parameters_with_gd_test_case()
parameters=update_parameters_with_gd(parameters,grads,learning_rate)
print("W1 = ",parameters["W1"])
print("b1 = ",parameters["b1"])
print("W2 = ",parameters["W2"])
print("b2 = ",parameters["b2"])

#随机梯度下降算法SGD即batch_size=1
#批量梯度下降算法即batch_size=m

"""
#仅作比较，不运行
#批量梯度下降即梯度下降
X=data_input
Y=labels
parameters=initialize_paramters(layers_dims)
for i in range(0,num_iterations):
    #前向传播
    A,cache=forward_propagation(X,parameters)
    #计算损失
    cost=compute_cost(A,Y)
    #反向传播
    grads=backward_propagation(X,Y,cache)
    #更新参数
    parameters=update_parameters(parameters,grads)
#完整遍历一遍数据集更新1次参数
    
#随机梯度下降算法
X=data_input
Y=labels
parameters=initialize_paramters(layers_dims)
for i in range(0,num_iterations):
    for j in range(m):
        # 前向传播
        A, cache = forward_propagation(X, parameters)
        # 计算损失
        cost = compute_cost(A, Y)
        # 反向传播
        grads = backward_propagation(X, Y, cache)
        # 更新参数
        parameters = update_parameters(parameters, grads)
#完整遍历一遍数据集更新m次参数
"""

#mini-batch梯度下降法
#分割数据集 batch_size 将数据集随机分成不同的小批次，并保证标签对应

def random_mini_batches(X,Y,mini_batch_size=64,seed=0):
    """
    从(X,Y)中创建一个随机的mini_batch列表
    :param X: 输入数据
    :param Y: 标签向量
    :param mini_batch_size: 每个mini_batch的样本大小
    :param seed: 随机种子
    :return:
    """
    np.random.seed(seed) #指定随机种子
    m=X.shape[1]
    mini_batches=[]

    #第一步：打乱顺序
    permutation=list(np.random.permutation(m)) #返回一个长度为m的随机数组，值为[0,m-1]
    # print(permutation)
    shuffled_X=X[:,permutation] #将每一列的数据按permutation的顺序重新排列
    shuffled_Y=Y[:,permutation].reshape((1,m))
    """
    #博主注：
    #如果你不好理解的话请看一下下面的伪代码，看看X和Y是如何根据permutation来打乱顺序的。
    x = np.array([[1,2,3,4,5,6,7,8,9],
                  [9,8,7,6,5,4,3,2,1]])
    y = np.array([[1,0,1,0,1,0,1,0,1]])

    random_mini_batches(x,y)
    permutation= [7, 2, 1, 4, 8, 6, 3, 0, 5]
    shuffled_X= [[8 3 2 5 9 7 4 1 6]
                 [2 7 8 5 1 3 6 9 4]]
    shuffled_Y= [[0 1 0 1 1 1 0 1 0]]
    """

    #第二步：分割
    num_complete_minibatches=math.floor(m/mini_batch_size) #将数据集划分为多少个mini_batch
    for k in range(0,num_complete_minibatches):
        mini_batch_X=shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y=shuffled_Y[:,k*mini_batch_size:(k+1)*mini_batch_size]
        """
        # 博主注：
        # 如果你不好理解的话请单独执行下面的代码，它可以帮你理解一些。
        a = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [9, 8, 7, 6, 5, 4, 3, 2, 1],
                      [1, 2, 3, 4, 5, 6, 7, 8, 9]])
        k = 1
        mini_batch_size = 3
        print(a[:, 1 * 3:(1 + 1) * 3])  # 从第4列到第6列
        '''
        [[4 5 6]
         [6 5 4]
         [4 5 6]]
        '''
        k = 2
        print(a[:, 2 * 3:(2 + 1) * 3])  # 从第7列到第9列
        '''
        [[7 8 9]
         [3 2 1]
         [7 8 9]]
        '''

        # 看一下每一列的数据你可能就会好理解一些
        """
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)

    #如果训练集的大小刚好是mini_batch_size的整数倍，那么这里已经处理完毕
    #如果不是整数倍，接着处理剩下的部分(在Pytorch中可以设置是否保留剩余部分)
    if m%mini_batch_size != 0 :
        #获取剩余部分
        mini_batch_X=shuffled_X[:,mini_batch_size*num_complete_minibatches:]
        mini_batch_Y=shuffled_Y[:,mini_batch_size*num_complete_minibatches:]

        mini_batch=(mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

#测试random_mini_batches
print("-"*30+"测试random_mini_batches"+"-"*30)
X_assess,Y_assess,mini_batch_size= testCase.random_mini_batches_test_case()
mini_batches=random_mini_batches(X_assess,Y_assess,mini_batch_size)
print("X_assess 的维度为",X_assess.shape)
print("第1个mini_batch_X 的维度为",mini_batches[0][0].shape)
print("第1个mini_batch_Y 的维度为",mini_batches[0][1].shape)
print("第2个mini_batch_X 的维度为",mini_batches[1][0].shape)
print("第2个mini_batch_Y 的维度为",mini_batches[1][1].shape)
print("第3个mini_batch_X 的维度为",mini_batches[2][0].shape)
print("第3个mini_batch_Y 的维度为",mini_batches[2][1].shape)

#momentum梯度下降法
def initialize_velocity(parameters):
    """
    初始化速度：velocity是一个字典：
        - key："dW1","db1","dW2","db2"
        - value:与相应的梯度/参数维度相同的值为0的矩阵
    :param parameters: 参数字典
    :return: v - 字典遍历，包含v[dW1]（dW1的速度）,v[db1]（db1的速度）等
    """
    L=len(parameters)//2
    v={}

    for l in range(L):
        v["dW"+str(l+1)]=np.zeros_like(parameters["W"+str(l+1)])
        v["db"+str(l+1)]=np.zeros_like(parameters["b"+str(l+1)])

    return v

#测试initialize_velocity
print("-"*30+"测试initialize_velocity"+"-"*30)
parameters=testCase.initialize_velocity_test_case()
v=initialize_velocity(parameters)
print("v[dW1] = ",v["dW1"])
print("v[db1] = ",v["db1"])
print("v[dW2] = ",v["dW2"])
print("v[db2] = ",v["db2"])

def update_parameters_with_momentum(parameters,grads,v,beta,learning_rate):
    """
    使用动量momentum更新参数
    :param parameters: 待更新的参数字典
    :param grads: 梯度字典
    :param v: 包含当前梯度速度的字典
    :param beta: 超参数 动量
    :param learning_rate: 学习率 超参数
    :return: parameters - 更新后的参数
                v - 包含了更新后的速度变量
    """
    L=len(parameters)//2

    for l in range(L):
        # 计算速度
        v["dW"+str(l+1)]=beta*v["dW"+str(l+1)]+(1-beta)*grads["dW"+str(l+1)]
        v["db"+str(l+1)]=beta*v["db"+str(l+1)]+(1-beta)*grads["db"+str(l+1)]

        #更新参数
        parameters["W"+str(l+1)]=parameters["W"+str(l+1)]-learning_rate*v["dW"+str(l+1)]
        parameters["b"+str(l+1)]=parameters["b"+str(l+1)]-learning_rate*v["db"+str(l+1)]

    return parameters,v

#测试update_parameters_with_momentum
print("-"*30+"测试update_parameters_with_momentum"+"-"*30)
parameters,grads,v=testCase.update_parameters_with_momentum_test_case()
parameters,v=update_parameters_with_momentum(parameters,grads,v,beta=0.9,learning_rate=0.01)

print("W1= ",parameters["W1"])
print("b1= ",parameters["b1"])
print("W2= ",parameters["W2"])
print("b2= ",parameters["b2"])
print("v[dW1]= ",v["dW1"])
print("v[db1]= ",v["db1"])
print("v[dW2]= ",v["dW2"])
print("v[db2]= ",v["db2"])

#Adam算法
#Momentum和RMSprop的结合体

def initialize_adam(parameters):
    """
    初始化v和s，它们都是字典变量，包含以下字段
        - keys: "dW1","db1",...,"dWL","dbL"
        - values: 与对应的梯度/参数相同维度的值为0的numpy矩阵
    :param parameters: 参数字典变量
    :return: v - 包含梯度的指数加权平均值 v["dW"+str(l)],v["db"+str(l)]
            s - 包含平方梯度的指数加权平均值 v["dW"+str(l)],v["db"+str(l)]
    """

    L=len(parameters)//2
    v={}
    s={}

    for l in range(L):
        v["dW"+str(l+1)]=np.zeros_like(parameters["W"+str(l+1)])
        v["db"+str(l+1)]=np.zeros_like(parameters["b"+str(l+1)])

        s["dW"+str(l+1)]=np.zeros_like(parameters["W"+str(l+1)])
        s["db"+str(l+1)]=np.zeros_like(parameters["b"+str(l+1)])

    return (v,s)

#测试initialize_adam
print("-"*30+"测试initialize_adam"+"-"*30)
parameters=testCase.initialize_adam_test_case()
v,s=initialize_adam(parameters)

print('v["dW1"] = ' + str(v["dW1"]))
print('v["db1"] = ' + str(v["db1"]))
print('v["dW2"] = ' + str(v["dW2"]))
print('v["db2"] = ' + str(v["db2"]))
print('s["dW1"] = ' + str(s["dW1"]))
print('s["db1"] = ' + str(s["db1"]))
print('s["dW2"] = ' + str(s["dW2"]))
print('s["db2"] = ' + str(s["db2"]))

def update_parameters_with_adam(parameters,grads,v,s,t,learning_rate=0.01,beta1=0.9,beta2=0.999,epsilon=1e-8):
    """
    使用Adam更新参数

    :param parameters:参数字典
    :param grads: 梯度字典
    :param v: 梯度的加权平均值 字典变量
    :param s: 平方梯度的加权平均值 字典变量
    :param t: 当前迭代的次数
    :param learning_rate:学习率
    :param beta1:动量超参数，用于第一阶段，使得曲线的Y值不从0开始
    :param beta2:RMSprop的一个超参数
    :param epsilon:防止分母除0操作
    :return:parameters - 更新后的参数字典
            v - 梯度的加权平均值
            s - 平方梯度的加权平均值
    """
    L=len(parameters)//2
    v_corrected={} #修正后的值
    s_corrected={} #修正后的值

    for l in range(L):
        #梯度的移动平均值
        v["dW"+str(l+1)]=beta1*v["dW"+str(l+1)]+(1-beta1)*grads["dW"+str(l+1)]
        v["db"+str(l+1)]=beta1*v["db"+str(l+1)]+(1-beta1)*grads["db"+str(l+1)]

        #计算v的偏差修正后的估计值
        v_corrected["dW"+str(l+1)]= v["dW"+str(l+1)]/(1-np.power(beta1,t))
        v_corrected["db"+str(l+1)]= v["db"+str(l+1)]/(1-np.power(beta1,t))

        #平方梯度的移动平均值
        s["dW"+str(l+1)]=beta2*s["dW"+str(l+1)]+(1-beta2)*np.square(grads["dW"+str(l+1)])
        s["db"+str(l+1)]=beta2*s["db"+str(l+1)]+(1-beta2)*np.square(grads["db"+str(l+1)])

        #计算s的偏差修正后的估计值
        s_corrected["dW"+str(l+1)]= s["dW"+str(l+1)]/(1-np.power(beta2,t))
        s_corrected["db"+str(l+1)]= s["db"+str(l+1)]/(1-np.power(beta2,t))

        #更新参数
        parameters["W"+str(l+1)]=parameters["W"+str(l+1)]-learning_rate*v_corrected["dW"+str(l+1)]/(np.sqrt(s_corrected["dW"+str(l+1)]+epsilon))
        parameters["b"+str(l+1)]=parameters["b"+str(l+1)]-learning_rate*v_corrected["db"+str(l+1)]/(np.sqrt(s_corrected["db"+str(l+1)]+epsilon))

    return (parameters,v,s)

#测试update_with_parameters_with_adam
print("-------------测试update_with_parameters_with_adam-------------")
parameters , grads , v , s = testCase.update_parameters_with_adam_test_case()
parameters,v,s=update_parameters_with_adam(parameters,grads,v,s,t=2)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
print('v["dW1"] = ' + str(v["dW1"]))
print('v["db1"] = ' + str(v["db1"]))
print('v["dW2"] = ' + str(v["dW2"]))
print('v["db2"] = ' + str(v["db2"]))
print('s["dW1"] = ' + str(s["dW1"]))
print('s["db1"] = ' + str(s["db1"]))
print('s["dW2"] = ' + str(s["dW2"]))
print('s["db2"] = ' + str(s["db2"]))

#三种优化器测试

#加载数据集
train_X,train_Y=opt_utils.load_dataset(is_plot=True)


#定义模型（之前定义的三层神经网络）
def model(X, Y, layers_dims, optimizer, learning_rate=0.0007,
          mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999,
          epsilon=1e-8, num_epochs=10000, print_cost=True, is_plot=True):
    """
    可以运行在不同优化器模式下的3层神经网络模型。

    参数：
        X - 输入数据，维度为（2，输入的数据集里面样本数量）
        Y - 与X对应的标签
        layers_dims - 包含层数和节点数量的列表
        optimizer - 字符串类型的参数，用于选择优化类型，【 "gd" | "momentum" | "adam" 】
        learning_rate - 学习率
        mini_batch_size - 每个小批量数据集的大小
        beta - 用于动量优化的一个超参数
        beta1 - 用于计算梯度后的指数衰减的估计的超参数
        beta1 - 用于计算平方梯度后的指数衰减的估计的超参数
        epsilon - 用于在Adam中避免除零操作的超参数，一般不更改
        num_epochs - 整个训练集的遍历次数，（视频2.9学习率衰减，1分55秒处，视频中称作“代”）,相当于之前的num_iteration
        print_cost - 是否打印误差值，每遍历1000次数据集打印一次，但是每100次记录一个误差值，又称每1000代打印一次
        is_plot - 是否绘制出曲线图

    返回：
        parameters - 包含了学习后的参数

    """
    L = len(layers_dims)
    costs = []
    t = 0  # 每学习完一个minibatch就增加1
    seed = 10  # 随机种子

    # 初始化参数
    parameters = opt_utils.initialize_parameters(layers_dims)

    # 选择优化器
    if optimizer == "gd":
        pass  # 不使用任何优化器，直接使用梯度下降法
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)  # 使用动量
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)  # 使用Adam优化
    else:
        print("optimizer参数错误，程序退出。")
        exit(1)

    # 开始学习
    for i in range(num_epochs):
        # 定义随机 minibatches,我们在每次遍历数据集之后增加种子以重新排列数据集，使每次数据的顺序都不同
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:
            # 选择一个minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # 前向传播
            A3, cache = opt_utils.forward_propagation(minibatch_X, parameters)

            # 计算误差
            cost = opt_utils.compute_cost(A3, minibatch_Y)

            # 反向传播
            grads = opt_utils.backward_propagation(minibatch_X, minibatch_Y, cache)

            # 更新参数
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2,
                                                               epsilon)
        # 记录误差值
        if i % 100 == 0:
            costs.append(cost)
            # 是否打印误差值
            if print_cost and i % 1000 == 0:
                print("第" + str(i) + "次遍历整个数据集，当前误差值：" + str(cost))
    # 是否绘制曲线图
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs (per 100)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

    return parameters

#1 梯度下降测试
#使用批量梯度下降(普通梯度下降)
layer_dims=[train_X.shape[0],5,2,1]
parameters=model(train_X,train_Y,layer_dims,optimizer="gd",is_plot=True)

#预测
predictions=opt_utils.predict(train_X,train_Y,parameters)

#绘制分类情况
plt.title("Model with Gradient Descent optimization")
axes=plt.gca() #进行坐标轴的移动
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,2.5])
opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)

#2 Momentum梯度下降测试
layer_dims=[train_X.shape[0],5,2,1]
parameters=model(train_X,train_Y,layer_dims,optimizer="momentum",beta=0.9,is_plot=True)

#预测
predictions=opt_utils.predict(train_X,train_Y,parameters)

#绘制分类情况
plt.title("Model with Momentum Gradient Descent optimization")
axes=plt.gca() #进行坐标轴的移动
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,2.5])
opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)

#3 Adam测试
layer_dims=[train_X.shape[0],5,2,1]
parameters=model(train_X,train_Y,layer_dims,optimizer="adam",beta=0.9,is_plot=True)

#预测
predictions=opt_utils.predict(train_X,train_Y,parameters)

#绘制分类情况
plt.title("Model with Adam optimization")
axes=plt.gca() #进行坐标轴的移动
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,2.5])
opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)



"""
Conclusion:
具有动量的梯度下降通常可以有很好的效果，但由于小的学习速率和简单的数据集所以它的影响几乎是轻微的。
另一方面，Adam明显优于小批量梯度下降和具有动量的梯度下降，如果在这个简单的模型上运行更多时间的数据集，这三种方法都会产生非常好的结果，
然而，我们已经看到Adam收敛得更快。
Adam的一些优点包括相对较低的内存要求（虽然比梯度下降和动量下降更高）和通常运作良好，
即使对参数进行微调（除了学习率α）
"""