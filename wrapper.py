import cv2 
import numpy as np 
import collections
import gym
import gym.spaces
from numpy.lib.utils import info 

class FireRestEnv(gym.Wrapper):
    def __init__(self, env=None):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings())>=3

    def step(self, action):
        return self.env.step(action)
    
    def reset(self, **kwargs):
        self.env.reset()
        obs,_,done,_ = self.env.step(1)
        if done: 
            self.env.reset()
        obs,_,done,_  = self.env.step(2)
        if done:
            self.reset()
        return obs 

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None,skip = 4):
        super().__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip 
    def step(self,action):
        total_reward= 0.0
        done = None 
        for _ in range(self._skip):
            obs,reward,done,inf = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done: 
                break
        max_frame = np.max(np.stack(self._obs_buffer),axis=0)
        return max_frame,total_reward,done,info
    
    def reset(self):
        self._obs_buffer.clear() 
        obs = self.env.reset() 
        self._obs_buffer.append(obs)
        return obs 

class ProcessFrame84(gym.ObservationWrapper):
    """
    Convert input observation from the emulator which normally has 
    resolution 210*160 rgb to 84x84 grayscale 
    """
    def __init__(self, env=None):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,high=255,shape=(84,84,1),dtype=np.uint8)
        
    def observation(self, obs):
        return ProcessFrame84.process(obs)
    
    @staticmethod
    def process(frame):
        if frame.size == 210*160*3:
            img = np.reshape(frame,[210,160,3]).astype(np.float32)
        elif frame.size == 250*160*3:
            img = np.reshape(frame,[250,160,3]).astype(np.float32)
        else:
            assert False, "Unknow resolution."
        
        img = img[:,:,0]*0.299 + img[:,:,1]*0.5787+img[:,:,2]*0.114
        resized_screen = cv2.resize(
            img, (84,110),
            interpolation = cv2.INTER_AREA)
        x_t = resized_screen[18:102,:]
        #Note the format: W H C
        x_t = np.reshape(x_t,[84,84,1])
        return x_t.astype(np.uint8)

class BufferWrapper(gym.ObservationWrapper):
    """
    Create stack of sequence frames along the first dimension and returns them as 
    an observation, get network a ideal about the dynamics of the objects,such as 
    speed, which is impossible to obtian from a single image 
    """
    #TODO: Try frame difference later
    def __init__(self, env,n_steps,dtype = np.float32):
        super(BufferWrapper,self).__init__(env) 
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            old_space.low.repeat(n_steps,axis=0),
            old_space.high.repeat(n_steps,axis=0),dtype = dtype)
        
    def reset(self):
        self.buffer = np.zeros_like(
            self.observation_space.low,dtype=self.dtype)
        
        return self.observation(self.env.reset())
    
    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer
    
class ImageToPytorch(gym.ObservationWrapper):
    """
    Changes the shape of the observation from HWC
    to CHW (Channel, height, width)
    """
    def __init__(self, env):
        super(ImageToPytorch,self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1],old_shape[0],old_shape[1])
        #print(new_shape)
        self.observation_space = gym.spaces.Box(
            low=0.0,high=1.0,
            shape= new_shape,dtype=np.float32)
    
    def observation(self, observation):
        return np.moveaxis(observation,2,0)
    
class ScaledFloatFrame(gym.ObservationWrapper):
    """
    Normalize frame
    """
    def observation(self, obs):
        return np.array(obs).astype(np.float32)/255.0
    
def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireRestEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPytorch(env)
    env = BufferWrapper(env,4)
    return ScaledFloatFrame(env)
    
