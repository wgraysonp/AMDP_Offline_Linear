import numpy as np
import random
import torch


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

class Replay_Buffer:
    def __init__(self, max_len):
        self.max_len = max_len
        self.curr_len = 0
        self.curr_idx = 0
        self.data = []

    def _is_full(self):
        return self.curr_len == self.max_len

    def push(self, state=None, action=None, next_state=None, reward=None):
        if not self._is_full():
            transition = dict(state=state, action=action, next_state=next_state, reward=reward)
            self.data.append(transition)
            self.curr_len += 1
            return True
        return False
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.curr_idx == self.max_len:
            self.curr_idx = 0
            raise StopIteration
        else:
            transition = self.data[self.curr_idx]
            self.curr_idx += 1
            return transition
        
    def __len__(self):
        return self.curr_len


def test():

    buffer = Replay_Buffer(max_len=10)
    state = 1
    action = 1
    next_state = 2
    reward = 0
    while buffer.push(state=state, action=action, next_state=next_state, reward=reward):
        state += 1
    for t in buffer:
        print(t['state'])
    for t in buffer:
        print(t)


if __name__ == "__main__":
    test()


    


        