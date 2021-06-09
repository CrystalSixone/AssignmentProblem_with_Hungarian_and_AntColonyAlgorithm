# -*- coding: utf-8 -*-
'''
@Date:   2021/06/04
@Author: w61
@Brief:  Generate the cost matrix and draw them on the map.
         Include generate the matrix randomly, generate the 2D map randomly and calculate the distances \
         between the points, draw the initial points on the map, and draw the assignment result on the map.
'''

import math
import numpy as np
from matplotlib import pyplot as plt

def getDistance(x1,y1,x2,y2):
    """
    Get distances between (x1,y1) and (x2,y2)
    
    :param x1
    :param y1
    :param x2
    :param ye
    :return distance
    """
    return math.sqrt((x1-x2)**2+(y1-y2)**2)

def randomMatrix(dim_x,dim_y,save_data=True,file_num=1):
    """
    Random generate 2D matrix
    
    :param dim_x: the dimention of the x
    :param dim_y: the dimention of the y
    :param save_data: whether save the generated matrix. default is 'True'
    :param file_num: the number/index of this file. used for distribuiting the matrixs with the same dimention.
    :return matrix
    """
    matrix = np.random.randint(1,100,size=(dim_x,dim_y))
    if save_data:
        np.save('matrix_{}_dimx_{}_dimy_{}.npy'.format(file_num,dim_x,dim_y),matrix)
        print('random matrix has been saved.')
    return matrix

def loadMatrix(dim_x,dim_y,file_num):
    """
    load the saved matrix.
    :param dim_x
    :param dim_y
    :param file_num
    """
    matrix = np.load('matrix_{}_dimx_{}_dimy_{}.npy'.format(file_num,dim_x,dim_y))
    print('data has been loaded. the matrix is\n',matrix)
    return matrix

def randomMap(dim_x,dim_y,draw_map=False,save_np=False,save_fig=False):
    """
    Generate the points of the uavs and targets in a 2D map, and calculate the cost(distance) matrix.
    
    :param dim_x
    :param dim_y
    :param draw_map: whether draw the map
    :param save_np: whether save the data
    :param save_fig: whether save the figure
    :return matrix: the cost/distance matrix of the points
    """
    uav_x = np.random.randint(1,100,dim_x)
    uav_y = np.random.randint(1,100,dim_y)
    target_x = np.random.randint(1,100,dim_x)
    target_y = np.random.randint(1,100,dim_y)
    matrix = np.ones((dim_x,dim_y))
    for i in range(dim_x):
        for j in range(dim_y):
            matrix[i][j] = getDistance(uav_x[i],uav_y[i],target_x[i],target_y[i])
    if save_np:
        np.save('map_uav_x_{}.npy'.format(dim_x),uav_x)
        np.save('map_uav_y_{}.npy'.format(dim_y),uav_y)
        np.save('map_target_x_{}.npy'.format(dim_x),target_x)
        np.save('map_target_y_{}.npy'.format(dim_y),target_y)
        np.save('map_matrix_{}x{}.npy'.format(dim_x,dim_y),matrix)

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
        if save_fig:
            plt.savefig('map_origin_{}x{}.png'.format(dim_x,dim_y),dpi=450,bbox_inches='tight')
        plt.show()

    return uav_x,uav_y,target_x,target_y,matrix

def loadMap(dim_x,dim_y,draw_map=False,save_fig=False):
    """
    load the saved map.
    :param dim_x
    :param dim_y
    :param draw_map: whether draw the points on the map
    :param save_fig: whether save the figure
    :return uav_x,uav_y,target_x,target_y,matrix
    """
    uav_x = np.load('map_uav_x_{}.npy'.format(dim_x))
    uav_y = np.load('map_uav_y_{}.npy'.format(dim_y))
    target_x = np.load('map_target_x_{}.npy'.format(dim_x))
    target_y = np.load('map_target_y_{}.npy'.format(dim_y))
    matrix = np.load('map_matrix_{}x{}.npy'.format(dim_x,dim_y))

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
        if save_fig:
            plt.savefig('map_origin_{}x{}.png'.format(dim_x,dim_y),dpi=450,bbox_inches='tight')
        plt.show()
    
    return uav_x,uav_y,target_x,target_y,matrix
    
    
