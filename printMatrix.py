import numpy as np

def printMatrix(matrix):
    matrix = np.array(matrix)
    for row in matrix:
        for i,item in enumerate(row):
            if i < len(row)-1:
                print('%d' %item,end='&')
            else:
                print('%d' %item,end=r'\\')
        print('')



test_unbalanced_ap3 = [
    [10,2,14,9,6,7,21,32,18,11],
    [7,12,9,3,5,6,9,16,54,12],
    [4,8,6,12,21,9,21,14,45,13],
    [21,9,12,9,32,10,19,25,16,10],
    [10,12,30,15,12,17,30,12,12,9],
    [15,7,34,17,7,16,14,17,9,5]
]
printMatrix(test_unbalanced_ap3)