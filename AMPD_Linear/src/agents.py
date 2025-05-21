import gymnasium as gym
#import cvxpy as cp
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
        return action_probs




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
    
    def act(self, state):
        action = self.policy.get_action(state)
        return action
    
    def get_cov(self, normalized=False):
        cov = self.reg_param*np.eye(self.embed_dim)
        for t in self.data:
            phi = self.feature_map.compute_feature_vector(t['state'], t['action'])
            cov = cov + np.outer(phi, phi)
        if normalized:
            n = len(self.buffer)
            cov = (1/n)*cov
        return cov


    def step():
        raise NotImplementedError
    
    def train():
        raise NotImplementedError



class OARAC_Agent(Linear_Agent):

    def __init__(self, eta=0.01, beta=5, bw=20):
        pass




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
    print(policy.get_action_probs(state))


if __name__ == '__main__':
    test()






