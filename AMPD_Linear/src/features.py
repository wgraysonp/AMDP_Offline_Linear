import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from gymnasium.core import ActType, ObsType

class Feature_Map:

    def __init__():
        pass
    
    def compute_feature_vector(self, state: ObsType, action: ActType):
        raise NotImplementedError
    
    def get_all_action_matrix(self, state: ObsType):
        Phi = np.zeros((self.n_actions, self.dim*self.n_actions))
        for a in range(self.n_actions):
            Phi[a, :] = self.compute_feature_vector(state, a)
        return Phi
    
    __call__ = compute_feature_vector
    


class Rbf_Feature_Map(Feature_Map):
    def __init__(self, centers: np.ndarray, n_actions, gamma=0.25):
        self.centers = centers
        self.n_actions = n_actions
        self.dim = len(centers) + 1
        self.gamma = gamma

    def compute_feature_vector(self, state: ObsType, action: ActType):
        a = int(action)
        phi = np.zeros(self.dim*self.n_actions)
        x = np.array(state).reshape(-1, 1)
        y = self.centers.reshape(-1, 1)
        v = rbf_kernel(x, y, gamma=0.25).squeeze()
        v = np.concatenate((np.array([1]), v))
        phi[self.dim*a:self.dim*(a+1)] = v
        return phi
    

def test():
    centers = np.linspace(0, 1, 10)
    state = 5
    feature_map = Rbf_Feature_Map(centers, n_actions=4)
    action = np.array([1])
    print(feature_map(state, action))
    print(feature_map.get_all_action_matrix(state))

if __name__ == '__main__':
    test()





