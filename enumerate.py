# -*- coding: utf-8 -*-
'''
@Date:  2021/05/31
@Author:w61
@Brief: Enumerate Method
'''
import math
import numpy as np
import itertools
import time

def loadMatrix(dim_x,dim_y,file_num):
    matrix = np.load('matrix_{}_dimx_{}_dimy_{}.npy'.format(file_num,dim_x,dim_y))
    print('data has been loaded. the matrix is\n',matrix)
    return matrix

def randomMatrix(dim_x,dim_y):
    """random generate of two dimentional matrix
    :param dim_x:[x,y]
    :param dim_y:[x,y]
    """
    return np.random.randint(1,100,size=(dim_x,dim_y))

class Enumerate():
    def __init__(self,matrix):
        """ use Eenumerate method to complete the best path
        :param matrix: the input matrix
        """
        self.input_matrix = np.array(matrix)
        self.dim = self.input_matrix.shape[0]
    
    def enumerate(self):
        start_time = time.time()
        min_cost = 9999
        min_list = [] # save task assignment
        self.temp_arange = np.arange(self.dim)
        permutation = list(itertools.permutations(self.temp_arange)) # generate full permutation
        for item in permutation:
            cost = 0
            for i,per_uav in enumerate(item):
                a = self.input_matrix[per_uav][i]
                cost += self.input_matrix[per_uav][i]
            if cost < min_cost:
                min_cost = cost
                min_list = item

        total_time = time.time() - start_time
        print('====finish====')
        print('total_time:{}'.format(total_time))
        print('total_cost:{}'.format(min_cost))
        print('task assignment:')
        for i,item in enumerate(min_list):
            print('第{}个人做第{}个任务'.format(item+1,i+1))
    
    def enumerate_2(self):
        min_cost = 9999
        min_list = []


if __name__ == '__main__':
    # matrix = [
    #     [2,15,13,4],
    #     [10,4,14,15],
    #     [9,14,16,13],
    #     [7,8,11,9]
    # ]
    dim = 10
    print('阶层：{}'.format(math.factorial(dim)))
    # matrix = randomMatrix(dim,dim)
    matrix = loadMatrix(8,8,1)
    print(str(list(matrix)).replace('array','').replace('(','').replace(')',''))
    sol = Enumerate(matrix)
    sol.enumerate()


