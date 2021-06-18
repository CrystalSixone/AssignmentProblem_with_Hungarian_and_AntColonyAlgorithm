# AssignmentProblem_with_Hungarian_and_AntColonyAlgorithm
Using Hungarian Algorithm and Ant Colony Algorithm to deal with balanced (m==n) and unbalanced (m&lt;n) assignment problems.

# 复现Yadaiah的步骤：
1) 先注释掉 `sub_matrix = np.load('sub_matrix.npy')
    subsub_matrix = np.load('subsub_matrix.npy')`
    这两句话，然后用以下函数求得两个子矩阵：
    ```
    sol_3 = Hungarian(matrix)
    sol_3.runYadaiah()
    ```

2) 两个子矩阵会自动保存成.npy文件，这时候就可以打开np.load来分别对两个nxn的子矩阵使用匈牙利算法。注意，这里得到的结果，需要用输出的temp_list和temp_list_col来与原来的矩阵对应。

3)对应方法：temp_list中的数代表第一个子矩阵(sub_matrix)所选的行，
            temp_list_col中的数代表第二个子矩阵(subsub_matrix)所选的列，行是第一次没有选择的剩下的行。

4) 总cost是两次子矩阵的cost之和。