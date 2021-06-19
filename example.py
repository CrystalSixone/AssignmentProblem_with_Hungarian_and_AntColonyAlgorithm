# -*- coding: utf-8 -*-
'''
@Date:   2021/06/1
@Author: w61
@Brief:  Ant Colony Algorithm
'''

import numpy as np
from numpy.random import rand
from antColony import AntColony
from matplotlib import pyplot as plt
import math

def randomMatrix(dim_x,dim_y,save_data=True,file_num=1):
    """random generate of two dimentional matrix
    :param dim_x:[x,y]
    :param dim_y:[x,y]
    """
    matrix = np.random.randint(1,100,size=(dim_x,dim_y))
    if save_data:
        np.save('matrix_{}_dimx_{}_dimy_{}.npy'.format(file_num,dim_x,dim_y),matrix)
        print('random matrix has been saved.')
    return matrix

def loadMatrix(dim_x,dim_y,file_num):
    matrix = np.load('matrix_{}_dimx_{}_dimy_{}.npy'.format(file_num,dim_x,dim_y))
    print('data has been loaded. the matrix is \n',matrix)
    return matrix

def generateDistances(matrix):
    """generate the distance matrix according to ant the cost matrix
    :param matrix: the input cost matrix
    :return new_matrix: the distance matrix used for ant colony
    """
    matrix = np.array(matrix)
    m,n = matrix.shape
    new_matrix = np.ones((m+n,m+n))
    for i in range(m+n):
        for j in range(m+n):
            if i < m and j < m:
                new_matrix[i][j] = max_num
            elif i < m and j >= m:
                new_matrix[i][j] = matrix[i][j-m]
            elif i >= m and j < m:
                new_matrix[i][j] = min_num
            elif i >= m and j >= m:
                new_matrix[i][j] = max_num
    # print('transformed matrix:\n',new_matrix)
    return new_matrix

def printAllocation(path,cost_matrix,uav_num,uav_x=None,uav_y=None,target_x=None,target_y=None,draw_map=True):
    """print allocation according to the best path.
    :param path: the best path calculated by the ant colony
    :param cost_matrix: the cost matrix
    :param uav_num: the number of the uav
    """
    cost = 0
    cost_matrix = np.array(cost_matrix)
    path = path[0]
    path.sort()
    if draw_map:
        plt.title('Result')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([1,110])
        plt.ylim([1,110])
    for item in path:
        i,j = item[0],item[1]
        if i<j:
            temp_i = i+1
            temp_j = j+1-uav_num
            print('第{}个人完成第{}个任务'.format(temp_i,temp_j))
            cost += cost_matrix[i][j-uav_num]
            if draw_map:
                x1,y1,x2,y2 = uav_x[i],uav_y[i],target_x[j-uav_num],target_y[j-uav_num]
                plt.scatter(x1,y1,marker='o',color='red',s=100)
                plt.text(x1+1,y1+1,'uav'+str(temp_i),fontsize=10)
                plt.scatter(x2,y2,marker='v',color='blue',s=100)
                plt.plot([x1,x2],[y1,y2],label='task_{}'.format(i+1))
                plt.text(x2+1,y2+1,'target'+str(temp_j),fontsize=10)
                # plt.arrow(x1,y1,x2-x1,y2-y1,width=1)
    print('总代价',cost)
    # plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
    plt.savefig('result_{}_ant.png'.format(len(cost_matrix)),dpi=450,bbox_inches='tight')
    # plt.show()

def getDistance(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)

def generateMap(dim_x,dim_y,draw_map=True,random=True):
    if random:
        uav_x = np.random.randint(1,100,dim_x)
        uav_y = np.random.randint(1,100,dim_y)
        target_x = np.random.randint(1,100,dim_x)
        target_y = np.random.randint(1,100,dim_y)

        # np.save('map_uav_x_{}.npy'.format(dim_x),uav_x)
        # np.save('map_uav_y_{}.npy'.format(dim_y),uav_y)
        # np.save('map_target_x_{}.npy'.format(dim_x),target_x)
        # np.save('map_target_y_{}.npy'.format(dim_y),target_y)
    else:
        uav_x = np.load('map_uav_x_{}.npy'.format(dim_x))
        uav_y = np.load('map_uav_y_{}.npy'.format(dim_y))
        target_x = np.load('map_target_x_{}.npy'.format(dim_x))
        target_y = np.load('map_target_y_{}.npy'.format(dim_y))

    if draw_map:
        plt.title('initial')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([1,110])
        plt.ylim([1,110])
        plt.scatter(uav_x,uav_y,marker='o',label='uav',color='red',s=100)
        plt.scatter(target_x,target_y,marker='v',label='target',color='blue',s=100)
        for i in range(dim_x):
            plt.text(uav_x[i]+1,uav_y[i]+1,'uav'+str(i),fontsize=10)
            plt.text(target_x[i]+1,target_y[i]+1,'target'+str(i),fontsize=10)
        plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
        plt.savefig('result_origin_{}_ant.png'.format(dim_x),dpi=450,bbox_inches='tight')
        plt.show()
    
    matrix = np.ones((dim_x,dim_y))
    for i in range(dim_x):
        for j in range(dim_y):
            matrix[i][j] = getDistance(uav_x[i],uav_y[i],target_x[j],target_y[j])
    return uav_x,uav_y,target_x,target_y,matrix

if __name__ == '__main__':
    max_num = 9999
    min_num = 0.001

    test = [
        [2,15,13,4],
        [10,4,14,15],
        [9,14,16,13],
        [7,8,11,9]
    ]

    test2 = [
            [12,7,9,7,9],
            [8,9,6,6,6],
            [7,17,12,14,9],
            [15,14,6,6,10],
            [4,10,7,10,9]
        ]

    test3 = [
            [12,7,9,7,9],
            [8,9,6,6,6],
            [7,17,12,14,9],
            [15,14,6,6,10],
        ]
    
    test4 = [
        [2,15,13,4],
        [10,4,14,15],
        [9,14,16,13],
    ]

    test_unbalanced_ap1 = [
        [30,25,18,32,27,19,22],
        [29,31,19,18,21,20,30],
        [28,29,30,19,19,22,23],
        [29,30,19,24,25,19,18],
        [21,20,18,17,16,14,16]
    ]

    test_unbalanced_ap2 = [
        [300,250,180,320,270,190,220,260],
        [290,310,190,180,210,200,300,190],
        [280,290,300,190,190,220,230,260],
        [290,300,190,240,250,190,180,210],
        [210,200,180,170,160,140,160,180]
    ]

    test_unbalanced_ap3 = [
        [10,2,14,9,6,7,21,32,18,11],
        [7,12,9,3,5,6,9,16,54,12],
        [4,8,6,12,21,9,21,14,45,13],
        [21,9,12,9,32,10,19,25,16,10],
        [10,12,30,15,12,17,30,12,12,9],
        [15,7,34,17,7,16,14,17,9,5]
    ]

    test_unbalanced_ap4 = [
        [21,11,16,9,15,10,12,32,26,16],
        [14,15,20,10,16,3,6,9,21,14],
        [9,17,11,31,21,16,7,9,10,11],
        [16,23,8,15,10,3,6,3,20,23],
        [12,40,14,36,9,21,14,19,4,13],
        [8,18,9,42,8,11,19,9,32,20],
        [21,9,12,9,32,10,19,25,16,10]
    ]

    test_unbalanced_5 = [
        [10,5,10,15],
        [3,9,18,3],
        [10,7,3,2]
    ]

    test_unblanced_6= [
        [120,70,90,100,70,90],
        [80,90,60,80,60,60],
        [70,170,120,110,140,90],
        [150,140,60,70,60,100],
        [40,100,70,60,100,90]
    ]
    # test4 = randomMatrix(40,40)
    # print(test4)
    
    n_ants = 50
    n_best = 45
    n_iterations = 300
    decay = 0.9
    alpha = 1
    beta = 4
    
    """定matrix，不做图
    """
    matrix = test_unbalanced_ap2
    uav_num = len(matrix)
    filename = 'uav_{}_ants_{}_ite_{}_decay_{}_alpha_{}_beta_{}'.format(uav_num,n_ants,n_iterations,\
        decay,alpha,beta)
    print(filename)

    distances = generateDistances(matrix)
    ant_colony = AntColony(distances, n_ants, n_best, n_iterations, decay, uav_num, filename,alpha, beta)
    total_shortest_path = 9999
    for i in range(5):
        shortest_path = ant_colony.run()
        if shortest_path[1] < total_shortest_path:
            total_shortest_path = shortest_path[1]
        print ("shorted_path: {}".format(shortest_path))
        printAllocation(shortest_path,matrix,uav_num,draw_map=False)
    print('last result:',total_shortest_path)
    """生成地图，对结果连线
    """
    # uav_x,uav_y,target_x,target_y,matrix = generateMap(20,20)
    # # np.save('compare_matrix.npy',matrix)
    # # np.save('compare_uav_x.npy',uav_x)
    # # np.save('compare_uav_y.npy',uav_y)
    # # np.save('compare_target_x.npy',target_x)
    # # np.save('compare_target_y.npy',target_y)
    # print(matrix)
    # uav_num = len(matrix)
    # filename = 'uav_{}_ants_{}_ite_{}_decay_{}_alpha_{}_beta_{}'.format(uav_num,n_ants,n_iterations,\
    #     decay,alpha,beta)
    # print(filename)

    # distances = generateDistances(matrix)
    # ant_colony = AntColony(distances, n_ants, n_best, n_iterations, decay, uav_num, filename,alpha, beta)
    # shortest_path = ant_colony.run()
    # print ("shorted_path: {}".format(shortest_path))
    # printAllocation(shortest_path,matrix,uav_num,uav_x,uav_y,target_x,target_y)

