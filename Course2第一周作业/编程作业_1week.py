# -*- coding: utf-8 -*-
# @Time : 2022/8/30 15:39
# @Author : zhuyu
# @File : 编程作业_1week.py
# @Project : Python菜鸟教程

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils
import reg_utils
import gc_utils

"""
1. 初始化参数：
	1.1：使用0来初始化参数。
	1.2：使用随机数来初始化参数。
	1.3：使用抑梯度异常初始化参数（参见视频中的梯度消失和梯度爆炸）。
2. 正则化模型：
	2.1：使用二范数对二分类模型正则化，尝试避免过拟合。
	2.2：使用随机删除节点的方法精简模型，同样是为了尝试避免过拟合。
3. 梯度校验  ：对模型使用梯度校验，检测它是否在梯度下降的过程中出现误差过大的情况。
"""


# #初始化参数
#
# #读取并绘制数据
# train_X,train_Y,test_X,test_Y=init_utils.load_dataset(is_plot=True)
# print(train_X.shape)
# print(train_Y.shape)
#
# def print_test_information(function_name):
#     print("-"*30+str(function_name)+"【Test】"+"-"*30)
#
# def initialize_paramters_zeros(layers_dims):
#     """
#     初始化模型参数全部为0
#     :param layers_dims: 各层的节点数
#     :return: parameters
#     """
#     parameters={}
#     layers_num=len(layers_dims)
#     for i in range(1,layers_num):
#         parameters["W"+str(i)]=np.zeros(shape=(layers_dims[i],layers_dims[i-1]))
#         parameters["b"+str(i)]=np.zeros(shape=(layers_dims[i],1))
#
#     return parameters
#
# # #test - initialize_paramters_zeros
# # print_test_information("initialize_paramters_zeros")
# # layers_dims=[2,10,5,1]
# # parameters=initialize_paramters_zeros(layers_dims)
# # print(parameters["W1"].shape) #(10,2)
#
# def initialize_paramters_random(layers_dims):
#     """
#     随机初始化模型参数
#     :param layers_dims: 各层的节点数
#     :return: parameters
#     """
#     np.random.seed(3) #指定随机数种子
#     parameters={}
#     layers_num=len(layers_dims)
#     for i in range(1,layers_num):
#         parameters["W"+str(i)]=np.random.randn(layers_dims[i],layers_dims[i-1])*10 #使用10倍缩放
#         parameters["b"+str(i)]=np.zeros(shape=(layers_dims[i],1))
#
#     return parameters
#
# # #test - initialize_paramters_random
# # print_test_information("initialize_paramters_random")
# # layers_dims=[2,10,5,1]
# # parameters=initialize_paramters_random(layers_dims)
# # print(parameters,parameters["W1"].shape)
#
# def initialize_paramters_he(layers_dims):
#     """
#     使用根号下（2/上一层维度）这个公式来进行初始化，目的是为了抑制梯度消失贺梯度爆炸
#     :param layer_dims: 各层的节点数
#     :return: parameters
#     """
#     np.random.seed(3)
#     parameters = {}
#     layers_num = len(layers_dims)
#     for i in range(1, layers_num):
#         parameters["W" + str(i)] = np.random.randn(layers_dims[i],layers_dims[i-1])*np.sqrt(2/layers_dims[i-1])
#         parameters["b" + str(i)] = np.zeros(shape=(layers_dims[i], 1))
#
#     return parameters
#
# # print_test_information("initialize_paramters_he")
# # parameters = initialize_paramters_he([2, 4, 1])
# # print("W1 = " + str(parameters["W1"]))
# # print("b1 = " + str(parameters["b1"]))
# # print("W2 = " + str(parameters["W2"]))
# # print("b2 = " + str(parameters["b2"]))
#
# def model(X,Y,learning_rate=0.01,num_iteration=15000,print_cost=True,initialization="he",is_plot=True):
#     """
#     实现一个三层神经网络 Linear-relu-Linear-relu-Linear-sigmoid
#     :param X: 输入数据 维度(2,样本数) (2,300)
#     :param Y: 标签向量 维度(1,样本数) (1,300)
#     :param learning_rate: 学习率
#     :param num_iteration: 迭代数
#     :param print_cost: 是否打印成本值，每迭代1000次打印一次
#     :param initialization: 选择初始化方式 默认"he" 初始化类型【”zeros“||"random"||"he"】
#     :param is_plot: 是否绘制梯度下降曲线
#     :return: parameters - 学习后的参数
#     """
#
#     grads=[]
#     costs=[]
#     m=X.shape[1]
#     layers_dims=[X.shape[0],10,5,1] #各层节点数
#
#     #选择初始化模型参数的类型
#     if initialization=="zeros":
#         parameters=initialize_paramters_zeros(layers_dims)
#     elif initialization=="random":
#         parameters=initialize_paramters_random(layers_dims)
#     elif initialization=="he" :
#         parameters=initialize_paramters_he(layers_dims)
#     else  :
#         print("模型参数初始化错误！程序退出")
#         exit()
#
#     #开始学习
#     for i in range(num_iteration):
#         #前向传播
#         a3,cache=init_utils.forward_propagation(X,parameters)
#
#         #计算成本
#         cost=init_utils.compute_loss(a3,Y)
#
#         #反向传播
#         grads=init_utils.backward_propagation(X,Y,cache)
#
#         #更新参数
#         parameters=init_utils.update_parameters(parameters,grads,learning_rate=learning_rate)
#
#         #记录成本
#         if i%1000==0:
#             costs.append(cost)
#             if print_cost:
#                 print("第"+str(i)+"次迭代，cost = ",cost)
#
#     #学习完毕，绘制梯度下降cost曲线
#     if is_plot:
#         plt.plot(costs)
#         plt.ylabel("cost")
#         plt.xlabel("iteration (per hundreds)")
#         plt.title("Learning rate = "+str(learning_rate))
#         plt.show()
#
#     return parameters
#
# #比较三种不同初始化方式来训练模型
# initialization_choose=["zeros","random","he"]
# for test_choose in initialization_choose:
#     parameters=model(train_X,train_Y,learning_rate=0.01,num_iteration=15000,print_cost=True,initialization=test_choose,is_plot=True)
#     #查看预测结果
#     print("训练集")
#     predictions_train=init_utils.predict(train_X,train_Y,parameters)
#     print("测试集")
#     predictions_test=init_utils.predict(test_X,test_Y,parameters)
#
#     print("predictions_train = " + str(predictions_train))
#     print("predictions_test = " + str(predictions_test))
#
#     plt.title("Model with {0} initialization".format(test_choose))
#     axes = plt.gca()
#     axes.set_xlim([-1.5, 1.5])
#     axes.set_ylim([-1.5, 1.5])
#     init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)
#     plt.show()


#正则化模型 - 防止过拟合
"""
1 不使用正则化
2 使用正则化
    2.1 使用L2正则化
    2.2 使用随机节点删除
"""

train_X,train_Y,test_X,test_Y=reg_utils.load_2D_dataset()
print(train_X.shape)
print(train_Y.shape)

def compute_cost_with_regularization(A3,Y,parameters,lambd):
    """
    实现L2的正则化计算cost
    :param a3: 模型正向传播的输出结果 维度(最后一层节点数量，样本数)
    :param Y: 模型标签向量 维度(输出节点数量，样本数)
    :param lambd: L2正则化的惩罚项λ
    :return: cost 正则化后的cost
    """
    m=Y.shape[1]
    W1=parameters["W1"]
    W2=parameters["W2"]
    W3=parameters["W3"]

    cross_entropy_cost=reg_utils.compute_cost(A3,Y)  #不含正则项的cost
    L2_regularization_cost=lambd*(np.sum(np.square(W1))+np.sum(np.square(W2))+np.sum(np.square(W3)))/(2*m)
    cost=cross_entropy_cost+L2_regularization_cost

    return cost

#L2正则化改变了cost，因此也需要改变反向传播函数

def backward_propagation_with_regulariztion(X,Y,cache,lambd):
    """
    实现添加了L2正则化的模型的反向传播
    :param X: 输入数据
    :param Y: 标签向量
    :param cache: 正向传播cache缓存输出
    :param lambd: 正则项
    :return: gradients - 包含所有参数、激活值和预激活值变量梯度的字典
    """
    m= X.shape[1]

    (Z1,A1,W1,b1,Z2,A2,W2,b2,Z3,A3,W3,b3)=cache

    dZ3=A3-Y

    dW3=(1/m)*np.dot(dZ3,A2.T)+(lambd*W3/m)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = (1 / m) * np.dot(dZ2, A1.T) + ((lambd * W2) / m)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = (1 / m) * np.dot(dZ1, X.T) + ((lambd * W1) / m)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

#Dropout - 随机删除节点
#key： A -> A*D (D矩阵为01矩阵，控制节点的激活与失活)
def forward_propagation_with_dropout(X,parameters,keep_prob):
    """
    实现具有Dropout的三层神经网络的前向传播
    :param X: 输入数据集
    :param parameter: 各层参数W1 b1 W2 b2 W3 b3的字典
    :param keep_prob: 随机删除的概率 实数
    :return: A3 最后输出值
            cache 缓存值
    """
    np.random.seed(1)

    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]
    W3=parameters["W3"]
    b3=parameters["b3"]

    Z1=np.dot(W1,X)+b1
    A1=reg_utils.relu(Z1)

    D1=np.random.rand(A1.shape[0],A1.shape[1]) #步骤1：初始化矩阵D1
    D1=D1<keep_prob                            #步骤2：将D1值转为01矩阵
    A1=D1*A1                                   #步骤3：舍弃A1部分节点
    A1=A1/keep_prob                            #步骤4：缩放未舍弃的节点（D不为0的值）

    """
    #不理解的同学运行一下下面代码就知道了。
    import numpy as np

    np.random.seed(1)
    A1 = np.random.randn(1,3)
    print(A1)
    
    D1 = np.random.rand(A1.shape[0],A1.shape[1])
    print(D1)
    keep_prob=0.5
    D1 = D1 < keep_prob
    print(D1)
    
    A1 = A1 * D1
    print(A1)
    A1 = A1 / keep_prob
    print(A1)
    """
    #以下各层重复上述四个步骤
    #注意：这里选用的统一keep_prob 实际上可以传入一个keep_prob列表，针对不同层采用不同的概率值
    Z2=np.dot(W2,A1)+b2
    A2=reg_utils.relu(Z2)

    D2=np.random.rand(A2.shape[0],A2.shape[1])
    D2=D2<keep_prob
    A2=A2*D2
    A2=A2/keep_prob

    Z3=np.dot(W3,A2)+b3
    A3=reg_utils.sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    return A3,cache


def backward_propagation_with_dropout(X,Y,cache,keep_prob):
    """
    实现包含Dropout的前向传播的反向传播算法
    :param X:  输入数据集
    :param Y: 标签向量
    :param cache: 缓存参数字典
    :param keep_prob: 随机删除的概率
    :return: gradients - 包含各参数的字典
    """
    m=X.shape[1]

    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)=cache

    dZ3 = A3 - Y
    dW3 = (1 / m) * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)

    dA2=dA2*D2          #步骤1：使用正向传播时关闭的节点在反向传播时也要对应关闭
    dA2=dA2/keep_prob   #步骤2：缩放未舍弃的节点的值

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)

    dA1 = dA1 * D1  # 步骤1：使用正向传播时关闭的节点在反向传播时也要对应关闭
    dA1 = dA1 / keep_prob  # 步骤2：缩放未舍弃的节点的值

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


def model(X,Y,learning_rate=0.3,num_iterations=30000,print_cost=True,is_plot=True,lambd=0,keep_prob=1):
    """
    实现一个三层神经网络 Linear - relu - Linear - relu - Linear - Sigmoid
    :param X: 输入数据(2,样本数) (2,211)
    :param Y: 标签向量(1,样本数) (1,211)
    :param learning_rate: 学习率
    :param num_iterations: 迭代数
    :param print_cost: 是否打印cost
    :param is_plot: 是否绘制梯度下降学习曲线
    :param lambd: L2正则项超参数
    :param keep_prob: Dropout正则项超参数
    :return: parameters
    """

    grads=[]
    costs=[]
    m=X.shape[1]
    layers_dims=[X.shape[0],20,3,1]

    #初始化参数
    parameters=reg_utils.initialize_parameters(layers_dims)

    #开始学习训练
    for i in range(num_iterations):
        #前向传播
        #是否随机删除节点
        if keep_prob == 1: #不删除节点
            a3,cache=reg_utils.forward_propagation(X,parameters)
        elif keep_prob < 1: #随机删除节点
            a3,cache=forward_propagation_with_dropout(X,parameters,keep_prob)
        else :
            print("keep_prob参数错误！程序退出")
            exit()

        #计算成本
        # 是否使用L2范数
        if lambd == 0: #不使用L2范数
            cost=reg_utils.compute_cost(a3,Y)
        else: #使用L2范数
            cost=compute_cost_with_regularization(a3,Y,parameters,lambd)

        #反向传播
        ##可以同时使用L2范数贺Dropout，但本次实验不使用
        assert (lambd == 0 or keep_prob == 1)

        #两个参数的使用情况
        if (lambd == 0 and keep_prob == 1): #都不使用
            grads=reg_utils.backward_propagation(X,Y,cache)
        elif lambd != 0 : #使用L2正则项
            grads=backward_propagation_with_regulariztion(X,Y,cache,lambd)
        elif keep_prob <1: #使用Dropout
            grads=backward_propagation_with_dropout(X,Y,cache,keep_prob)

        #更新参数
        parameters=reg_utils.update_parameters(parameters,grads,learning_rate=learning_rate)

        #记录打印成本
        if i % 1000 == 0:
            costs.append(cost)
            if print_cost:
                print("第{0}次迭代，cost = {1}".format(i,cost))

    #绘制成本曲线图
    if is_plot:
        plt.plot(costs)
        plt.ylabel("cost")
        plt.xlabel("Iterations (x1,000)")
        plt.title("Learning_rate = "+str(learning_rate))
        plt.show()

    return parameters

#训练模型 并查看预测结果
# #不使用正则化
# parameters = model(train_X, train_Y,is_plot=True)
# print("训练集:")
# predictions_train = reg_utils.predict(train_X, train_Y, parameters)
# print("测试集:")
# predictions_test = reg_utils.predict(test_X, test_Y, parameters)
#
# #绘制分割线
# plt.title("Model without regularization")
# axes = plt.gca()
# axes.set_xlim([-0.75,0.40])
# axes.set_ylim([-0.75,0.65])
# reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)

# #使用L2范数
# parameters=model(train_X,train_Y,lambd=0.7,is_plot=True)
# print("训练集:")
# predictions_train = reg_utils.predict(train_X, train_Y, parameters)
# print("测试集:")
# predictions_test = reg_utils.predict(test_X, test_Y, parameters)
#
# #绘制分割线
# plt.title("Model with L2 regularization")
# axes = plt.gca()
# axes.set_xlim([-0.75,0.40])
# axes.set_ylim([-0.75,0.65])
# reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)

# #调试比较不同的lambd 查看L2正则项超参数对模型偏差和方差的影响
# lambds=np.arange(0.1,1,step=0.3) #[0.1 0.4 0.7]
# for lambd in lambds:
#     parameters = model(train_X, train_Y, lambd=lambd, is_plot=True)
#     print("训练集{0}:".format(lambd))
#     predictions_train = reg_utils.predict(train_X, train_Y, parameters)
#     print("测试集{0}:".format(lambd))
#     predictions_test = reg_utils.predict(test_X, test_Y, parameters)
#     # 绘制分割线
#     plt.title("Model with L2 regularization λ = "+str(lambd))
#     axes = plt.gca()
#     axes.set_xlim([-0.75, 0.40])
#     axes.set_ylim([-0.75, 0.65])
#     reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)

# #测试含有Dropout的模型
# #测试keep_prob=0.86 即模型1 2层的各个节点都有14%的概率失活
# parameters = model(train_X, train_Y, keep_prob=0.86, is_plot=True)
# print("训练集:")
# predictions_train = reg_utils.predict(train_X, train_Y, parameters)
# print("测试集:")
# predictions_test = reg_utils.predict(test_X, test_Y, parameters)
# # 绘制分割线
# plt.title("Model with Dropout regularization (keep_prob = 0.86)")
# axes = plt.gca()
# axes.set_xlim([-0.75, 0.40])
# axes.set_ylim([-0.75, 0.65])
# reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)

# #调试比较不同的lambd 查看L2正则项超参数对模型偏差和方差的影响
# keep_drops=np.arange(0.1,1,step=0.3) #[0.1 0.4 0.7]
# for keep_drop in keep_drops:
#     parameters = model(train_X, train_Y, keep_prob=keep_drop, is_plot=True)
#     print("训练集{0}:".format(keep_drop))
#     predictions_train = reg_utils.predict(train_X, train_Y, parameters)
#     print("测试集{0}:".format(keep_drop))
#     predictions_test = reg_utils.predict(test_X, test_Y, parameters)
#     # 绘制分割线
#     plt.title("Model with L2 regularization λ = "+str(keep_drop))
#     axes = plt.gca()
#     axes.set_xlim([-0.75, 0.40])
#     axes.set_ylim([-0.75, 0.65])
#     reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)

#梯度检验 - 验证反向传播梯度的正确性

# #一维
# def forward_propagation(x, theta):
#     """
#
#     实现图中呈现的线性前向传播（计算J）（J（theta）= theta * x）
#
#     参数：
#     x  - 一个实值输入
#     theta  - 参数，也是一个实数
#
#     返回：
#     J  - 函数J的值，用公式J（theta）= theta * x计算
#     """
#     J = np.dot(theta, x)
#
#     return J
#
#
# def backward_propagation(x, theta):
#     """
#     计算J相对于θ的导数。
#
#     参数：
#         x  - 一个实值输入
#         theta  - 参数，也是一个实数
#
#     返回：
#         dtheta  - 相对于θ的成本梯度
#     """
#     dtheta = x
#
#     return dtheta
#
#
# def gradient_check(x, theta, epsilon=1e-7):
#     """
#
#     实现图中的反向传播。
#
#     参数：
#         x  - 一个实值输入
#         theta  - 参数，也是一个实数
#         epsilon  - 使用公式（3）计算输入的微小偏移以计算近似梯度
#
#     返回：
#         近似梯度和后向传播梯度之间的差异
#     """
#
#     # 使用公式（3）的左侧计算gradapprox。
#     thetaplus = theta + epsilon  # Step 1
#     thetaminus = theta - epsilon  # Step 2
#     J_plus = forward_propagation(x, thetaplus)  # Step 3
#     J_minus = forward_propagation(x, thetaminus)  # Step 4
#     gradapprox = (J_plus - J_minus) / (2 * epsilon)  # Step 5
#
#     # 检查gradapprox是否足够接近backward_propagation（）的输出
#     grad = backward_propagation(x, theta)
#
#     numerator = np.linalg.norm(grad - gradapprox)  # Step 1'
#     denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)  # Step 2'
#     difference = numerator / denominator  # Step 3'
#
#     if difference < 1e-7:
#         print("梯度检查：梯度正常!")
#     else:
#         print("梯度检查：梯度超出阈值!")
#
#     return difference
#
# #测试gradient_check
# print("-----------------测试gradient_check-----------------")
# x, theta = 2, 4
# difference = gradient_check(x, theta)
# print("difference = " + str(difference))
#
#
# #高维
# def forward_propagation_n(X, Y, parameters):
#     """
#     实现图中的前向传播（并计算成本）。
#
#     参数：
#         X - 训练集为m个例子
#         Y -  m个示例的标签
#         parameters - 包含参数“W1”，“b1”，“W2”，“b2”，“W3”，“b3”的python字典：
#             W1  - 权重矩阵，维度为（5,4）
#             b1  - 偏向量，维度为（5,1）
#             W2  - 权重矩阵，维度为（3,5）
#             b2  - 偏向量，维度为（3,1）
#             W3  - 权重矩阵，维度为（1,3）
#             b3  - 偏向量，维度为（1,1）
#
#     返回：
#         cost - 成本函数（logistic）
#     """
#     m = X.shape[1]
#     W1 = parameters["W1"]
#     b1 = parameters["b1"]
#     W2 = parameters["W2"]
#     b2 = parameters["b2"]
#     W3 = parameters["W3"]
#     b3 = parameters["b3"]
#
#     # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
#     Z1 = np.dot(W1, X) + b1
#     A1 = gc_utils.relu(Z1)
#
#     Z2 = np.dot(W2, A1) + b2
#     A2 = gc_utils.relu(Z2)
#
#     Z3 = np.dot(W3, A2) + b3
#     A3 = gc_utils.sigmoid(Z3)
#
#     # 计算成本
#     logprobs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
#     cost = (1 / m) * np.sum(logprobs)
#
#     cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
#
#     return cost, cache
#
#
# def backward_propagation_n(X, Y, cache):
#     """
#     实现图中所示的反向传播。
#
#     参数：
#         X - 输入数据点（输入节点数量，1）
#         Y - 标签
#         cache - 来自forward_propagation_n（）的cache输出
#
#     返回：
#         gradients - 一个字典，其中包含与每个参数、激活和激活前变量相关的成本梯度。
#     """
#     m = X.shape[1]
#     (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
#
#     dZ3 = A3 - Y
#     dW3 = (1. / m) * np.dot(dZ3, A2.T)
#     dW3 = 1. / m * np.dot(dZ3, A2.T)
#     db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
#
#     dA2 = np.dot(W3.T, dZ3)
#     dZ2 = np.multiply(dA2, np.int64(A2 > 0))
#     # dW2 = 1. / m * np.dot(dZ2, A1.T) * 2  # Should not multiply by 2
#     dW2 = 1. / m * np.dot(dZ2, A1.T)
#     db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)
#
#     dA1 = np.dot(W2.T, dZ2)
#     dZ1 = np.multiply(dA1, np.int64(A1 > 0))
#     dW1 = 1. / m * np.dot(dZ1, X.T)
#     # db1 = 4. / m * np.sum(dZ1, axis=1, keepdims=True) # Should not multiply by 4
#     db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)
#
#     gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
#                  "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
#                  "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
#
#     return gradients
#
#
# def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7):
#     """
#     检查backward_propagation_n是否正确计算forward_propagation_n输出的成本梯度
#
#     参数：
#         parameters - 包含参数“W1”，“b1”，“W2”，“b2”，“W3”，“b3”的python字典：
#         grad_output_propagation_n的输出包含与参数相关的成本梯度。
#         x  - 输入数据点，维度为（输入节点数量，1）
#         y  - 标签
#         epsilon  - 计算输入的微小偏移以计算近似梯度
#
#     返回：
#         difference - 近似梯度和后向传播梯度之间的差异
#     """
#     # 初始化参数
#     parameters_values, keys = gc_utils.dictionary_to_vector(parameters)  # keys用不到
#     grad = gc_utils.gradients_to_vector(gradients)
#     num_parameters = parameters_values.shape[0]
#     J_plus = np.zeros((num_parameters, 1))
#     J_minus = np.zeros((num_parameters, 1))
#     gradapprox = np.zeros((num_parameters, 1))
#
#     # 计算gradapprox
#     for i in range(num_parameters):
#         # 计算J_plus [i]。输入：“parameters_values，epsilon”。输出=“J_plus [i]”
#         thetaplus = np.copy(parameters_values)  # Step 1
#         thetaplus[i][0] = thetaplus[i][0] + epsilon  # Step 2
#         J_plus[i], cache = forward_propagation_n(X, Y, gc_utils.vector_to_dictionary(thetaplus))  # Step 3 ，cache用不到
#
#         # 计算J_minus [i]。输入：“parameters_values，epsilon”。输出=“J_minus [i]”。
#         thetaminus = np.copy(parameters_values)  # Step 1
#         thetaminus[i][0] = thetaminus[i][0] - epsilon  # Step 2
#         J_minus[i], cache = forward_propagation_n(X, Y, gc_utils.vector_to_dictionary(thetaminus))  # Step 3 ，cache用不到
#
#         # 计算gradapprox[i]
#         gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
#
#     # 通过计算差异比较gradapprox和后向传播梯度。
#     numerator = np.linalg.norm(grad - gradapprox)  # Step 1'
#     denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)  # Step 2'
#     difference = numerator / denominator  # Step 3'
#
#     if difference < 1e-7:
#         print("梯度检查：梯度正常!")
#     else:
#         print("梯度检查：梯度超出阈值!")
#
#     return difference


