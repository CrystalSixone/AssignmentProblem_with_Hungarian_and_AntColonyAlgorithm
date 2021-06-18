# -*- coding: utf-8 -*-
'''
@Date:  2021/05/30
@Author:w61
@Brief: Using imporved Hungarian algorithm to solve balanced and unbalanced assignment problems.
'''
import math
import numpy as np
import copy
import time

from matplotlib import pyplot as plt
from numpy.core.fromnumeric import sort
from costMatrix import getDistance,randomMatrix,loadMatrix,randomMap,loadMap

class Hungarian():
    def __init__(self,matrix):
        """
        Impletement or Hungarian Algorithm using python
        
        :param matrix: the dimention of the matrix can be different. m x n. (m machines, n targets, and m <= n)
        """
        self.input_matrix = np.array(matrix) # Origin input matrix
        self.cal_matrix = copy.deepcopy(self.input_matrix) # matrix used in calculation
        self.dim_row, self.dim_col = self.input_matrix.shape
    
    def convertZeroMatrix(self):
        """ 
        Let at least one zero element appear in each row and column of the matrix
        
        :return convert_matrix
        """
        for i in range(self.dim_row):
            if 0 in self.cal_matrix[i,:]:
                continue
            self.cal_matrix[i] = self.cal_matrix[i] - min(self.cal_matrix[i,:])
        
        for i in range(self.dim_col):
            if 0 in self.cal_matrix[:,i]:
                continue
            self.cal_matrix[:,i] = self.cal_matrix[:,i] - min(self.cal_matrix[:,i])
        return self.cal_matrix
    
    def circleAndCross(self):
        """
        Circle and cross zero on the zero element matrix
        
        :return circle_hash, cross_hash
        """
        circle_num = 0   # the number of zeros in the cal_matrix
        circle_hash = []
        cross_hash = []
        for i in range(self.dim_row): # calculate the number of zeros in the matrix
            for j in range(self.dim_col):
                if self.cal_matrix[i][j] == 0:
                    circle_num += 1

        while len(circle_hash) + len(cross_hash) < circle_num: 
            zero_row_num = [] # the number of zeros in each row
            min_val = self.dim_row
            min_index = 0
            for i in range(self.dim_row): # ith row
                zero_num = 0
                for j in range(self.dim_col):
                    if self.cal_matrix[i][j] == 0 and [i,j] not in cross_hash and [i,j] not in circle_hash:
                        zero_num += 1
                zero_row_num.append([i,zero_num]) # record the number of zero in each row. [ith row, zero numbers]

            for item in zero_row_num:   # find the row with the least zero
                num, val = item[0],item[1]
                if val < min_val and val != 0:
                    min_val = val
                    min_index = num

            min_col_zero = self.dim_col
            min_col_index = 0
            if min_val == 1:  # if there is only one zero in the row
                for i in range(self.dim_col):
                    if self.cal_matrix[min_index][i] == 0 and [min_index,i] not in circle_hash and [min_index,i] not in cross_hash:
                        min_col_index = i
                        break
            else:
                for i in range(self.dim_col): # find the column with the least zero
                    if self.cal_matrix[min_index][i] == 0 and [min_index,i] not in circle_hash and [min_index,i] not in cross_hash:
                        num_zero_col = 0
                        for j in range(self.dim_row):
                            if self.cal_matrix[j][i] == 0 and [j,i] not in cross_hash and [j,i] not in circle_hash:
                                num_zero_col += 1
                        if num_zero_col < min_col_zero:
                            min_col_zero = num_zero_col
                            min_col_index = i

            circle_hash.append([min_index,min_col_index]) # circle the zero
            for row_index in range(self.dim_row): # cross the zero
                if row_index != min_index and self.cal_matrix[row_index][min_col_index]==0 and [row_index,min_col_index] not in cross_hash \
                    and [row_index,min_col_index] not in circle_hash:
                    cross_hash.append([row_index,min_col_index])
            for col_index in range(self.dim_col):
                if col_index != min_col_index and self.cal_matrix[min_index][col_index]==0 and [min_index,col_index] not in cross_hash \
                    and [min_index,col_index] not in circle_hash:
                    cross_hash.append([min_index,col_index])

        return circle_hash,cross_hash
    
    def circleAndCrossUnbalanced(self):
        """
        Circle and cross zero on the zero element matrix when the dimention is not square
        
        :return circle_hash, cross_hash
        """
        circle_num = 0   # the number of zeros in the cal_matrix
        circle_hash = []
        cross_hash = []
        for i in range(self.dim_row): # calculate the number of zeros in the matrix
            for j in range(self.dim_col):
                if self.cal_matrix[i][j] == 0:
                    circle_num += 1

        while len(circle_hash) + len(cross_hash) < circle_num: 
            zero_row_num = [] # the number of zeros in each row
            zero_distribution = [] # the list to record the locations of the zero. [i,j]
            min_val = self.dim_row
            min_index = 0
            for i in range(self.dim_row): # ith row
                zero_num = 0
                for j in range(self.dim_col):
                    if self.cal_matrix[i][j] == 0 and [i,j] not in cross_hash and [i,j] not in circle_hash:
                        zero_distribution.append([i,j])
                        zero_num += 1
                if zero_num != 0:
                    zero_row_num.append([i,zero_num]) # record the number of zero in each row. [ith row, zero numbers]

            zero_row_num.sort(key = lambda x:x[0]) # find the rows with the least zero.
            last_zero_num = -1
            least_zero_list = [] # the list to record the rows with the least number of zero
            for item in zero_row_num:
                if last_zero_num == -1:
                    last_zero_num = item[1] # record the first number after sorted
                    least_zero_list.append(item[0])
                    continue
                if item[1] == last_zero_num:
                    least_zero_list.append(item[0])
                else:
                    break
            
            min_val = 9999
            min_index,min_col_index = 0,0
            for row in least_zero_list: # find the smallest cost in these zero
                for item in zero_distribution:
                    if item[0] == row:
                        if self.input_matrix[row][item[1]] < min_val:
                            min_val = self.input_matrix[row][item[1]]
                            min_index,min_col_index = item[0],item[1]

            circle_hash.append([min_index,min_col_index]) # circle the zero
            for row_index in range(self.dim_row): # cross the zero. since the matrix is not square, don't cross the zero in the same row.
                if row_index != min_index and self.cal_matrix[row_index][min_col_index]==0 and [row_index,min_col_index] not in cross_hash \
                    and [row_index,min_col_index] not in circle_hash:
                    cross_hash.append([row_index,min_col_index])

        return circle_hash,cross_hash
    
    def tickAndLine(self,circle_hash,cross_hash):
        """ 
        When the length of circle_hash is less than self.dim_col(the number of the targets), we tick and line on the matrix.
        
        :return: cal_matrix
        """
        tick_row = []
        tick_col = []
        temp_row = np.arange(self.dim_row)
        temp_circle_list = []
        for item in circle_hash: # find which row has no circle 
            i,j = item[0],item[1]
            temp_circle_list.append(i)
        temp_row = np.delete(temp_row,temp_circle_list)
        for i in temp_row:
            tick_row.append(i)  # add the row into the row tick list

        while True:
            break_flag = True 
            for i in tick_row:
                for j in range(self.dim_col):
                    if [i,j] in cross_hash and j not in tick_col:
                        tick_col.append(j)  # add the column into the col tick list
                        break_flag = False
            
            for j in tick_col:
                for i in range(self.dim_row):
                    if [i,j] in circle_hash and i not in tick_row:
                        tick_row.append(i)  # add the row into the row tick list
                        break_flag = False
            
            if break_flag: # break if there is no more ticks
                break
            
        min_val = 9999
        for i in tick_row:  # find the smallest value in the cal matrix without lined
            for j in range(self.dim_col):
                if j not in tick_col:
                    if self.cal_matrix[i][j] < min_val:
                        min_val = self.cal_matrix[i][j]
        for i in range(self.dim_row):  # substract and sum the smallest value
            for j in range(self.dim_col):
                if i in tick_row and j not in tick_col:
                    self.cal_matrix[i][j] -= min_val
                if i not in tick_row and j in tick_col:
                    self.cal_matrix[i][j] += min_val
        
        return self.cal_matrix
    
    def paddingZero(self,matrix):
        """
        padding the zero column or row when the dimention is not equare.
        :param matrix:the input matrix
        :return matrix: the square matrix with padding zero.
        """
        m,n = matrix.shape
        if m == n:
            print('the input matrix is saqure.')
        elif m < n:
            temp = np.zeros([n-m,n])
            matrix = np.vstack((matrix,temp))
        elif m > n:
            temp = np.zeros([m,m-n])
            matrix = np.hstack((matrix,temp))
        return matrix
    
    def methodYadaiah(self,matrix):
        """ The split method proposed by Yadaiah
        :param matrix: the input matrix
        :return matrixs
        """
        matrix = np.array(matrix).transpose()
        sum_row = matrix.sum(axis=1)
        sum_col = matrix.sum(axis=0)
        sort_row = sorted(enumerate(sum_row),key=lambda x:x[1])
        sort_col = sorted(enumerate(sum_col),key=lambda x:x[1])
        sub_matrix = []
        temp_list = []
        for i in range(matrix.shape[1]):
            temp_list.append(sort_row[i][0])
        temp_list.sort()
        for item in temp_list:
            sub_matrix.append(matrix[item])
        
        sub_matrix = np.array(sub_matrix)

        subsub_matrix = []
        temp_list_col = []
        for i in range(matrix.shape[0]-matrix.shape[1]):
            temp_list_col.append(sort_col[i][0])
        temp_list_col.sort()

        for i in range(matrix.shape[0]):
            row_list = []
            if i in temp_list:
                continue
            for j in range(matrix.shape[1]):
                if j in temp_list_col:
                    row_list.append(matrix[i][j])
            subsub_matrix.append(row_list)
        subsub_matrix = np.array(subsub_matrix)

        print('temp_list',temp_list)
        print('temp_list_col',temp_list_col)
        np.save('sub_matrix.npy',sub_matrix)
        np.save('subsub_matrix.npy',subsub_matrix)
        print(sub_matrix,'\n',subsub_matrix)
    
    def methodBetts(self,matrix):
        """ The copy and zero padding method proposed by Betts
        :param matrix: the input matrix
        :return matrixs
        """
        matrix = np.array(matrix).transpose()
        matrix = np.column_stack((matrix,matrix))
        m,n = matrix.shape
        temp_zero = np.zeros([n-m,n])
        matrix = np.row_stack((matrix,temp_zero))
        np.save('betts_matrix',matrix)
        print(matrix)

    
    def drawMap(self,uav_x,uav_y,target_x,target_y):
        """
        draw the calculated result between the uavs and targets on the map.
        :param uav_x: coordinate X for uavs.
        :param uav_y: coordinate Y for uavs.
        :param target_x: coordinate X for targets.
        :param target_y: coordinate Y for targets.
        """
        plt.title('result')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([1,110])
        plt.ylim([1,110])
        for item in self.circle_hash:
            i = item[0]
            j = item[1]
            x1,y1,x2,y2 = uav_x[i],uav_y[i],target_x[j],target_y[j]
            plt.scatter(x1,y1,marker='o',color='red',s=100)
            plt.text(x1+1,y1+1,'uav'+str(i+1),fontsize=10)
            plt.scatter(x2,y2,marker='v',color='blue',s=100)
            plt.plot([x1,x2],[y1,y2],label='task_{}'.format(i+1))
            plt.text(x2+1,y2+1,'target'+str(j+1),fontsize=10)
        plt.show()
    
    def runQuazzafi(self):
        """ 
        Main function for modified hungarian method written by Quazzafi.
        """
        print('==============start==============')
        start_time = time.time()
        self.convertZeroMatrix()
        circle_hash,cross_hash = self.circleAndCross()
        while len(circle_hash) != self.dim_col:
            self.tickAndLine(circle_hash,cross_hash)
            if self.dim_row == self.dim_col:
                circle_hash,cross_hash = self.circleAndCross()
            else:
                circle_hash,cross_hash = self.circleAndCrossUnbalanced()
        total_time = time.time() - start_time
        circle_hash.sort()
        self.circle_hash = circle_hash
        total_cost = 0
        last_uav_index = 1
        print('circle_hash',circle_hash)
        print('assignment result is:')
        print('\tUAV\tTASK')
        for i,item in enumerate(circle_hash):
            if item[0]+1 != last_uav_index:
                print('-----------------------')
            print('\t{}\t{}'.format(item[0]+1,item[1]+1))
            last_uav_index = item[0]+1
            total_cost += self.input_matrix[item[0]][item[1]]
        print('total cost: {}'.format(total_cost))
        print('total time:{}'.format(total_time))
        print('==============finish==============')
    
    def runPaddingZero(self):
        """ 
        Main function for hungarian with zero paddings.
        """
        print('==============start==============')
        start_time = time.time()
        self.cal_matrix = self.paddingZero(self.cal_matrix)
        self.dim_row, self.dim_col = self.cal_matrix.shape
        self.convertZeroMatrix()
        circle_hash,cross_hash = self.circleAndCross()
        while len(circle_hash) != self.dim_col:
            self.tickAndLine(circle_hash,cross_hash)
            if self.dim_row == self.dim_col:
                circle_hash,cross_hash = self.circleAndCross()
            else:
                circle_hash,cross_hash = self.circleAndCrossUnbalanced()
        total_time = time.time() - start_time
        circle_hash.sort()
        self.circle_hash = circle_hash
        total_cost = 0
        last_uav_index = 1
        print('circle_hash',circle_hash)
        print('assignment result is:')
        print('\tUAV\tTASK')
        for i,item in enumerate(circle_hash):
            if item[0]+1 != last_uav_index:
                print('-----------------------')
            print('\t{}\t{}'.format(item[0]+1,item[1]+1))
            last_uav_index = item[0]+1
            if item[0] < self.input_matrix.shape[0] and item[1] < self.input_matrix.shape[1]:
                total_cost += self.input_matrix[item[0]][item[1]]
        print('total cost: {}'.format(total_cost))
        print('total time:{}'.format(total_time))
        print('==============finish==============')
    
    def runYadaiah(self):
        self.methodYadaiah(self.input_matrix)
    
    def runBetts(self):
        self.methodBetts(self.input_matrix)

if __name__ == '__main__':
    test = [
        [2,15,13,4],
        [10,4,14,15],
        [9,14,16,13],
        [7,8,11,9]
    ]

    test3 = [
        [2,15,13,4],
        [10,4,14,15],
        [9,14,16,13],
    ]

    test2 = [
        [12,7,9,7,9],
        [8,9,6,6,6],
        [7,17,12,14,9],
        [15,14,6,6,10],
        [4,10,7,10,9]
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

    # sub_matrix = np.load('sub_matrix.npy')
    # subsub_matrix = np.load('subsub_matrix.npy')
    betts_matrix = np.load('betts_matrix.npy')
    # matrix = loadMatrix(40,40,1)
    # matrix = generateMap(25,25)
    # uav_x,uav_y,target_x,target_y,matrix = loadMap()
    # matrix = test_unbalanced_ap3
    matrix = betts_matrix
    # sol_1 = Hungarian(matrix)
    # sol_1.runPaddingZero()

    sol_2 = Hungarian(matrix)
    sol_2.runQuazzafi()

    # sol_3 = Hungarian(matrix)
    # sol_3.runYadaiah()

    # sol_4 = Hungarian(matrix)
    # sol_4.runBetts()
    
    # sol.drawMap(uav_x,uav_y,target_x,target_y)

    # test = [
    #     [0,3],
    #     [1,2],
    #     [4,1]
    # ]
    # test.sort(key = lambda x:x[1])
    # print(test)