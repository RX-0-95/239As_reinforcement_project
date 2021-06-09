from tarfile import is_tarfile
import cv2
from gym.core import RewardWrapper 
import numpy as np 
import collections
import gym
import gym.spaces
from numpy.lib.utils import info 

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs

class ClippedRewardsWrapper(gym.RewardWrapper):
    def reward(self, reward):
        """Change all the positive rewards to 1, negative to -1 and keep zero."""
        return np.sign(reward)

class FireRestEnv(gym.Wrapper):
    def __init__(self, env=None):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings())>=3

    def step(self, action):
        #self.env.step(1)
        #fire_ram = np.random.randint(1,30)
        #if fire_ram > 15:
        #    self.env.step(action)
        return self.env.step(action)
    
    def reset(self):
        self.env.reset()
        #print('------------Fire---------------')
        obs,_,done,_ = self.env.step(1)
        if done: 
            self.env.reset()
        obs,_,done,_  = self.env.step(2)
        if done:
            self.reset()
        return obs 

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True
        self.was_real_reset = False
        #self.is_fire_able = (env.unwrapped.get_action_meanings()[1] == 'FIRE')
        #print('Is the env fireable: {}'.format(self.is_fire_able))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        #print('Lives remian {}, done: {}, selflives: {}'.format(lives,done,self.lives))
        self.lives = lives
        return obs, reward, done, info

    def reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
            self.was_real_reset = True
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            #fire if the the env can fire 
            #if self.is_fire_able:
                #print('-----------Eposiod fire ----------')
            #    self.env.step(1)
            self.was_real_reset = False
        self.lives = self.env.unwrapped.ale.lives()
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
    

def make_env(env_name,stack_frames =4, episodic_life = True,reward_clipping=False,
            norm_frame = True,*args,**kwargs):
    env = gym.make(env_name,*args,**kwargs)
    if episodic_life:
        env = EpisodicLifeEnv(env)

    env = NoopResetEnv(env,noop_max=30)
    
    env = MaxAndSkipEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireRestEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPytorch(env)
    env = BufferWrapper(env,stack_frames)
    if norm_frame:
        env = ScaledFloatFrame(env)
    if reward_clipping:
        env = ClippedRewardsWrapper(env)
    return env

"""
def make_env(env_name,stack_frames =4, episodic_life = True,reward_clipping=True,
            norm_frame = False):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireRestEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPytorch(env)
    env = BufferWrapper(env,stack_frames)
    return ScaledFloatFrame(env)
"""
