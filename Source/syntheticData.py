"""
    Synthetic data
    X ~ N(0, 1)
    T ~ Confounding with X, and from discrete value from [0, k]
    Y ~ Linear or non-linear function between X, T and Y
"""

import numpy as np
import math
from tqdm import tqdm


class Synthetic_data:

    def __init__(self, size, x_dimension, t_d, seed, std_strength=1.0, vary_group=5):
        self.size = size
        self.x_dimension = x_dimension
        self.t_d = t_d
        self.X = np.zeros(shape=(size, x_dimension))
        self.T = np.zeros(shape=(size, 1))
        self.Y = np.zeros(shape=(size, 1))
        self.std_strength = std_strength
        self.vary_group = vary_group
        self.seed = seed

        # Initialisation
        self.rng = np.random.RandomState(self.seed)
        self.generate_weights()
        self.make_context()
        self.initialise_contribution_weight()
        np.random.seed(seed)


    def generate_weights(self):
        """
            Sparsity: sparsity of the weight martix

            Generate weights for synthetic data
            1. Treatment generation
                - Confounder weights
                - Treatment weights
            2. Reward generation
                - Contribution weights
        """
        
        # --------- Weight for treatment assignment --------------
        self.confounder_weight = self.rng.binomial(0, 0.5, size=(self.t_d, self.x_dimension))

        # --------- Weight for reward --------------
        # self.interaction_weight = self.rng.normal(0.5, 1, size=(self.t_d, self.x_dimension - 1))
        idx = self.rng.choice(10, 7, replace=False)
        self.x_contribution_weight = self.rng.normal(0, 1, size=(self.x_dimension, 1))

        self.x_interaction = self.rng.normal(0, 1, size=(self.x_dimension, 1))
        self.x_contribution_weight[idx] = 0
        idx = self.rng.choice(10, 3, replace=False)
        self.x_interaction[idx] = 0

    def get_data(self):
        """
        :return: A dataset (X, T) -> R^(N * (D + K))
        """
        dict = {}
        dict['context'] = self.X
        dict['action'] = self.T.squeeze().astype(int)
        dict['data'] = np.concatenate((self.X, self.T), axis=1)
        dict['reward'] = self.Y.squeeze()
        return dict


    def get_group_data(self):
        dict = {}
        dict['context'] = self.X
        dict['action'] = self.T
        dict['data'] = np.concatenate((self.X, self.T), axis=1)
        dict['reward'] = self.Y
        return dict

    def make_context(self):
        """
            :param size: Size of the data
            :param x_dimension: The dimension of x of each sample
            :return: Return the synthetic dataset for X (size x dimension)
        """
        mean = 0
        std = 1
        self.X = self.rng.normal(mean, std, size=[self.size, self.x_dimension])
        # self.X = self.rng.uniform(0, 1, size=[self.size, self.x_dimension])
        return self.X
    
    def initialise_contribution_weight(self):

        self.GROUP = self.vary_group
        self.K = 2
        self.sigma = 0.1
        self.num_group = int(self.t_d / self.GROUP)
        A =  - ( self.t_d / self.GROUP - 1) / self.K / 2
        self.interaction_weight = np.zeros(shape=(self.t_d, 1))
        
        # self.interaction_params = self.rng.normal(0, 1, size=(self.x_dimension - 1, self.x_dimension))
        idx = self.rng.choice(10, 5, replace=False)
        # self.interaction_params[:,idx] = 0
        self.v = self.rng.uniform(0, 1, size=(self.x_dimension, 3))
        print(self.v)
        for i in range(3):
            self.v[:, i] /= np.linalg.norm(self.v[:, i], ord=2)

        idx = [i for i in range(self.t_d)]
        # self.rng.shuffle(idx)
        self.action_to_bin = {i:int(idx[i]//self.GROUP) for i in range(self.t_d)}
        for i in range(self.t_d):
            # self.interaction_weight[i] = self.rng.normal( int(i//self.GROUP), self.sigma, size=(1))
            self.interaction_weight[i] = self.rng.normal( int(idx[i]//self.GROUP) * (3/self.num_group), 1/self.t_d * self.std_strength, size=(1))
            # self.interaction_weight[i] = self.rng.normal( 0, 1, size=(1))
            # self.interaction_weight[i] = 0
        # interaction_copy = self.interaction_weight.copy()
        # self.rng.shuffle(self.interaction_weight)

        print(self.interaction_weight)
        print(self.action_to_bin)

    def get_action_to_bin(self):
        return self.action_to_bin

    def generate_behavior_policy(self, beta, treatmentEffect):
        """
            Here we generate the behavior policy based on the reward of given action and treatment
            which is \delta(x, a), and it follows the value of beta 
        """

        self.behavior_policy = np.zeros(shape=(self.size, self.t_d))
        self.rewards = np.zeros(shape=(self.size, self.t_d))
        self.beta_rewards = np.zeros(shape=(self.size, self.t_d))
        
        self.epislon = self.rng.normal(0, 1, size=[self.size, 1])
        self.treatmentEffect = treatmentEffect
        print("------Generate behavior policy-------")
        alpha = 5
        dist = {i:0 for i in range(-50,500)}
        for i in tqdm(range(self.size)):
            temp = 0
            # print(np.matmul(self.X[i], self.v[:, 0]))
            temp =  np.exp( np.matmul(self.X[i], self.v[:, 0]) ) * - np.matmul(self.X[i], self.v[:, 1])
            temp += self.epislon[i]
            # temp = np.matmul(self.X[i], self.v[:, 0])
            self.rewards[i] = temp.repeat(self.t_d, 0)
            for t in range(self.t_d ):

                # r = np.matmul(self.X[i], self.x_contribution_weight) 
                self.rewards[i][t] += abs(self.interaction_weight[t] + np.matmul(self.X[i], self.v[:, 2])) ** treatmentEffect
                # self.rewards[i][t] += 2 * np.matmul(self.X[i], self.v[:, 1]) * abs(self.interaction_weight[t])
                # self.rewards[i][t] -=  (abs(self.interaction_weight[t]) ** 2)
                
            self.beta_rewards[i] = self.rewards[i]
            dist[int(np.matmul(self.X[i], self.v[:, 2]))] += 1
            self.beta_rewards[i] /= abs(self.beta_rewards[i]).mean()
            
            self.beta_rewards[i] *= beta
            # self.beta_rewards[i] /= np.linalg.norm(self.beta_rewards[i], ord=2)
            self.behavior_policy[i] = np.exp(self.beta_rewards[i]) / sum(np.exp(self.beta_rewards[i]))

        print(self.rewards[0:10])
        # print(self.behavior_policy[0:2])
        return self.behavior_policy


    def generate_target_policy(self, epislon):
        """
            The target policy is generated based on the reward of each action
            Epislon controls the quality of target policy, 0 stands for perfect policy.
        """

        self.target_policy = np.zeros(shape=(self.size, self.t_d))
        ground_truth = 0
        print("-------Generate target policy-------")
        beta = 1
        for i in tqdm(range(self.size)):
            optimal_t = self.rewards[i].argmax()
            # self.target_reward = self.rewards[i] / abs(self.rewards[i]).mean()
            # self.target_policy[i] = np.exp(beta * self.target_reward) / sum(np.exp(beta * self.target_reward))
            for t in range(self.t_d):
                # ground_truth += self.target_policy[i][t] * self.rewards[i][t]
                if t == optimal_t:
                    self.target_policy[i][t] = (1 - epislon) 
                    ground_truth += self.target_policy[i][t] * self.rewards[i][t]
                else:
                    self.target_policy[i][t] =  epislon / (self.t_d - 1)
                    ground_truth += self.target_policy[i][t] * self.rewards[i][t]

        return self.target_policy, ground_truth / self.size

    def make_treatment(self, policy):
        """
            Assign the treatment according to the policy and user feature x. 
        """

        for i in range(self.size):

            self.T[i] = np.random.choice(self.t_d, p=self.behavior_policy[i])
            # print(self.behavior_policy[i])
            self.Y[i] = self.rewards[i][int(self.T[i])]
        
        values, counts = np.unique(self.T, return_counts=True)
        print("Treatment values:{} and counts:{}".format(values,counts))

        print("Average rewards:", self.Y.mean())
        return self.T

    def get_reward(self, X, t_d, cluster, treatment_effect):
        """
            Support the cluster visulisation

            Compute the Y vector based on given user feature

            Return the reward matrix of size [X, t_d]
        """


        rewards = np.zeros(shape=( X.shape[0], t_d))


        for t in range(t_d):
            # Fixed part of the reward only depends on the user feature

            for i in range(X.shape[0]):
                # Varying part of the reward based on the given treatment
                rewards[i][t] = abs(self.interaction_weight[t] + np.matmul(X[i], self.v[:, 2])) ** treatment_effect
                rewards[i][t] +=   np.exp( np.matmul(X[i], self.v[:, 0]) ) * - np.matmul(X[i], self.v[:, 1])

                # rewards[i][t] = np.exp( np.matmul(self.X[i], self.v[:, 0]) ) * - np.matmul(self.X[i], self.v[:, 1])
                # rewards[i][t] += 2 * np.matmul(self.X[i], self.v[:, 2]) *  abs(self.interaction_weight[t] ) ** 0.8

        return rewards

        