import random as rn
import numpy as np
from numpy.random import choice as np_choice
from matplotlib import pyplot as plt
import time

SAVE_DATA = True

class AntColony(object):
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, uav_num, filename='result', alpha=1, beta=1,max_time=0):
        """
        Args:
            distances (2D numpy.array): Square matrix of distances. Diagonal is assumed to be np.inf.
            n_ants (int): Number of ants running per iteration
            n_best (int): Number of best ants who deposit pheromone
            n_iteration (int): Number of iterations
            decay (float): Rate it which pheromone decays. The pheromone value is multiplied by decay, so 0.95 will lead to decay, 0.5 to much faster decay.
            alpha (int or float): exponenet on pheromone, higher alpha gives pheromone more weight. Default=1
            beta (int or float): exponent on distance, higher beta give distance more weight. Default=1
            uav_num(int): the number of the uav
            filename(str): save the data into csv file
            max_time(int): per uav execute the maxmium number of the tasks
        Example:
            ant_colony = AntColony(german_distances, 100, 20, 2000, 0.95, alpha=1, beta=2)          
        """
        self.distances  = distances
        self.max_num = distances[0][0]
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.uav_num = uav_num
        if max_time == 0 or max_time > self.uav_num:
            self.uav_visited_max_time = self.uav_num
        else:
            self.uav_visited_max_time = max_time    
        print(self.uav_visited_max_time)       
        # drawing
        self.draw_x = np.arange(n_iterations)
        self.draw_y = np.ones(n_iterations)
        # save data
        self.filename = filename

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        start_time = time.time()
        count_stable = 0
        for i in range(self.n_iterations):
            # reduce the number of the ants
            # if i % (self.n_iterations/4) == 0 and i!=0: 
            #     self.n_ants = int(self.n_ants/2)
            #     self.n_best = int(self.n_best/2)

            all_paths = self.gen_all_paths()
            self.spread_pheronome(all_paths, self.n_best, shortest_path=shortest_path)
            shortest_path = min(all_paths, key=lambda x: x[1])
            # print (shortest_path)
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
                count_stable = 0            
            # self.pheromone = self.pheromone * self.decay   
            # self.draw_y[i] = all_time_shortest_path[1] - self.uav_num
            self.draw_y[i] = all_time_shortest_path[1]

            count_stable += 1
            if count_stable >= 100: # break if the shortest path keep stable more than 100 times
                self.draw_y[i:] = self.draw_y[i]
                break

        print('total time:',time.time()-start_time)
        # if SAVE_DATA:
            # self.saveData() 
        self.drawing()     
        return all_time_shortest_path

    def spread_pheronome(self, all_paths, n_best, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                # self.pheromone[move] += 1.0 / self.distances[move]
                self.pheromone[move] = self.decay * self.pheromone[move] + 1.0/self.distances[move]

    def gen_path_dist(self, path):
        total_dist = 0
        for ele in path:
            total_dist += self.distances[ele]
        return total_dist

    def gen_all_paths(self):
        """generate paths for all ants
        """
        all_paths = []
        for i in range(self.n_ants):
            start = np.random.randint(self.uav_num)
            path = self.gen_path(0)
            # path = self.gen_path(start) # randomly generate the start point
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self, start):
        """generate one path
        """
        path = []
        visited = set()
        visited_task = set()
        self.uav_visited_time = np.zeros([1,self.uav_num])
        visited.add(start)
        self.uav_visited_time[0][start] += 1
        prev = start
        # for i in range(len(self.distances) - 1):
        while len(visited) < len(self.distances):  # all the nodes should be visited at least once
        # while True:
            # print('visited',visited)
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            if move not in visited:
                visited.add(move)
            if move < self.uav_num:
                self.uav_visited_time[0][move] += 1

            # if move not in visited_task and move >= self.uav_num: # Chinese version
            #     visited_task.add(move)
            # if len(visited_task) == len(self.distances)-self.uav_num:
            #     break

        path.append((prev, start)) # going back to where we startqed    
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        for i in visited:
            if i >= self.uav_num: # node i is the target
                pheromone[i] = 0  # node i cannot be visited again
            elif i < self.uav_num:# node i is the uav
                if self.uav_visited_time[0][i] >= self.uav_visited_max_time:
                    pheromone[i] = 0
        # pheromone[list(visited)] = 0
        row = pheromone ** self.alpha * (( 1.0 / dist) ** self.beta)
        norm_row = (row / row.sum())
        # print('norm_row',norm_row)
        move = np_choice(self.all_inds, 1, p=norm_row)[0]
        return move

    def drawing(self):
        plt.title("Result")
        plt.xlabel('iteration')
        plt.ylabel('cost')
        plt.plot(self.draw_x,self.draw_y)
        if SAVE_DATA:
            plt.savefig('{}_fig.png'.format(self.filename),dpi=450)
        # plt.show()
    
    def saveData(self):
        np.save('{}_x.npy'.format(self.filename),self.draw_x)
        np.save('{}_y.npy'.format(self.filename),self.draw_y)
        print('data has been saved')
    
    def printAllocation(self,path,cost_matrix,uav_num,uav_x=None,uav_y=None,target_x=None,target_y=None,draw_map=True):
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
            plt.title('result')
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
        plt.legend(loc='center left',bbox_to_anchor=(1,0.5))
        # plt.savefig('result_{}_ant.png'.format(len(cost_matrix)),dpi=450,bbox_inches='tight')
        # plt.show()

