import gymnasium as gym
import cvxpy as cp
import numpy as np
import torch
from torch.nn.functional import softmax
from torch.distributions import Categorical
from gymnasium.core import ActType, ObsType
from features import Feature_Map, Rbf_Feature_Map
from utils import Replay_Buffer

class SoftmaxPolicy:

    def __init__(self, feature_map: Feature_Map, n_actions=5, eta=0.01, embed_dim=10):
        self.eta = eta
        self.dim = embed_dim
        self.param = np.zeros(embed_dim)
        self.n_actions = n_actions
        self.feature_map = feature_map

    def get_action(self, state: ObsType):
        Phi_s = self.feature_map.get_all_action_matrix(state)
        X = torch.tensor(np.matmul(Phi_s, self.param))
        action_probs = softmax(X, dim=0)
        m = Categorical(action_probs)
        action = m.sample()
        return action.item()
    
    def step(self, grad: np.ndarray):
        self.param = self.param + self.eta*grad

    def get_action_probs(self, state):
        Phi_s = self.feature_map.get_all_action_matrix(state)
        print(Phi_s.shape)
        print(self.param.shape)
        X = torch.tensor(np.matmul(Phi_s, self.param))
        action_probs = softmax(X, dim=0)
        return action_probs.numpy()
    
    def reset(self):
        self.param = np.zeros(self.embed_dim)




class Linear_Agent:

    def __init__(self, 
                 data: Replay_Buffer,
                 env: gym.Env, 
                 feature_map: Feature_Map, 
                 n_actions=5, 
                 eta=0.01,
                 embed_dim=10,
                 reg_param=1
                 ):
        self.env = env
        self.data = data
        self.feature_map = feature_map
        self.policy = SoftmaxPolicy(feature_map, n_actions=n_actions, eta=eta, embed_dim=embed_dim)
        self.eta = eta
        self.n_actions = n_actions
        self.reg_param = reg_param
        self.embed_dim = embed_dim
        self.rewards = []
        self.name = None
    
    def act(self, state):
        action = self.policy.get_action(state)
        return action
    
    def get_cov(self, normalized=False):
        cov = self.reg_param*np.eye(self.embed_dim)
        for t in self.data:
            phi = self.feature_map(t['state'], t['action'])
            cov = cov + np.outer(phi, phi)
        if normalized:
            n = len(self.buffer)
            cov = (1/n)*cov
        return cov


    def step(self):
        raise NotImplementedError
    
    def eval(self, episode_length: int=100):
        '''
        Evaluate the current policy. Interact with the environment
        for 'episide_length' steps and compute average reward.

        Args:
            episode_length: number of steps to take
        '''

        observation, _ = self.env.reset()
        rewards = []
        for t in range(episode_length):
            action = self.policy.get_action(observation)
            observation, reward, _, _, _ = self.env.step(action)
            rewards.append(reward)

        avg_reward = rewards.mean()
        self.rewards.append(avg_reward)

        return avg_reward
    
    def train(self, steps=100, eval_every=10, eval_length=100, verbose=True):
        for i in range(steps):
            if i % eval_every == 0:
                avg_reward = self.eval(episode_length=eval_length)
                if verbose:
                    print(f"Agent: {self.name}, Episode: {i+1}, Reward: {avg_reward}")
            self.step()


    def reset(self):
        self.rewards = []
        self.policy.reset()




class OARAC_Agent(Linear_Agent):

    def __init__(self, 
                 data: Replay_Buffer,
                 env: gym.Env, 
                 feature_map: Feature_Map, 
                 n_actions=5, 
                 eta=0.01,
                 embed_dim=10,
                 reg_param=1,
                 beta=10,
                 bw=10
                 ):
        
        super().__init__(data,
                         env,
                         feature_map,
                         n_actions=n_actions,
                         eta=eta,
                         embed_dim=embed_dim,
                         reg_param=reg_param)
        self.beta = beta
        self.bw = bw
        self.name = 'OARAC Agent'

    def get_problem_params(self):
        """
        Get the matrices for the critic optimization problem:
            min v v^t e_1
            st Av = R
            v^tB_1v <= 1
            v^t B_2v <= B_w
            v^t B_3v <= beta

        Returns:
            tuple: matrices A in R^{d x 2d+1}
            B_1 in R^{2d+1 x 2d+1}
            B_2 in R^{2d+1 x 2d+1}
            B_3 in R^{2d+1 x 2d+1}
            R in R^{d}
        """
        d = self.embed_dim
        A1 = np.zeros((d, d))
        A2 = np.eye(d, d)
        A3 = np.zeros((1, d))
        R = np.zeros((1, d))

        for t in self.data:
            state = t['state']
            action = t['action']
            next_state = t['next_state']
            reward = t['reward']

            curr_feature = self.feature_map(state, action)
            Phi_prime = self.feature_map.get_all_action_matrix(next_state)
            action_probs = self.policy.get_action_probs(next_state)
            next_feature = Phi_prime.T @ action_probs

            A1 = A1 + curr_feature.reshape(-1, 1)
            A3 = A3 + np.outer(curr_feature, next_feature)
            R = R + reward*curr_feature.reshape(-1, 1)

        Lambda = self.get_cov()
        Lambda_inv = np.linalg.inv(Lambda)
        A1 = Lambda_inv@A1
        A3 = Lambda_inv@A3
        A3 = np.eye(d, d) - A3
        R = Lambda_inv@R

        A = np.hstack((A1, A2, A3))

        B_1 = np.zeros((2*d + 1, 2*d + 1))
        B_1[0, 0] = 1

        Z_d = np.zeros((d, d))

        B_2 = np.block([
            [np.zeros((1, d)), Z_d, Z_d],
            [np.zeros((1, d)), Lambda, Z_d],
            [np.zeros((1, d)), Z_d, Z_d]
        ])

        B_3 = np.block([
            [np.zeros((1, d)), Z_d, Z_d],
            [np.zeros((1, d)), Z_d, Z_d],
            [np.zeros((1, d)), Z_d, np.eye(d, d)]
        ])

        return A, B_1, B_2, B_3, R



    def step(self):
        '''
        Perform a single algorithm step. 
        '''

        # critic update
        d = self.embed_dim

        v = cp.Variable(2*d  +1)
        A, B1, B2, B3, R = self.get_problem_params()
        constraints = [A @ v == R,
                       v.T @ B1 @ v <= 1,
                       v.T @ B2 @ v <= self.beta**2,
                       v.T @ B3 @ v <= self.bw**2
        ]
        e1 = np.zeros_like(2*d + 1)
        e1[0] = 1
        objective = cp.Minimize(e1.T @ v)
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # actor update
        w = v.value[d+1:]
        self.policy.step(w)


        






def test():
    centers = np.linspace(0, 1, 10)
    state = 5
    feature_map = Rbf_Feature_Map(centers, n_actions=4)
    embed_dim = 4*(len(centers) + 1)
    policy = SoftmaxPolicy(feature_map, n_actions=4, eta=1, embed_dim=embed_dim)
    action = policy.get_action(state)
    print(action)
    grad = np.random.randn(embed_dim)
    policy.step(grad)
    action = policy.get_action(state)
    print(action)
    action_probs = policy.get_action_probs(state)
    print(action_probs.reshape(-1, 1))
    features = feature_map.get_all_action_matrix(state)
    print(features)
    print(features.T @ action_probs)


if __name__ == '__main__':
    test()






