from typing import override, SupportsFloat, Any
import gymnasium as gym
from gymnasium.core import ActType, ObsType, RenderFrame, WrapperObsType
from gymnasium.wrappers import Autoreset

class Terminal_Reset_Wrapper(Autoreset):
    '''Autoreset wrapper with custom reward when terminal state is reached
       Modified from "https://gymnasium.farama.org/_modules/gymnasium/wrappers/common/#Autoreset"

       Args:
            env (gym.Env): Environment to apply wrapper
            reward (float): reward to receive when episode terminates    
    '''

    def __init__(self, env: gym.Env, reward=0):
        super().__init__(self, env)
        self.reward = reward
    
    @override
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment with action and resets the environment if a terminated or truncated signal is encountered.

        Args:
            action: The action to take

        Returns:
            The autoreset environment :meth:`step`
        """
        if self.autoreset:
            obs, info = self.env.reset()
            reward, terminated, truncated = reward, False, False
        else:
            obs, reward, terminated, truncated, info = self.env.step(action)

        self.autoreset = terminated or truncated
        return obs, reward, terminated, truncated, info
