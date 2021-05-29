import gym
import time 
import argparse 
import numpy as np 
import torch
from torch.functional import tensordot 
import wrapper 
import dqn_model
import collections
import imageio 
from skimage.transform import resize
DEFAULT_ENV_NAME = 'BreakoutNoFrameskip-v4'
DEFAULT_MODEL ='model_backup/Episode-life-BreakoutNoFrameskip-v4-best.dat'
FPS = 25 
GAME_LIFT = 5 
GIF_SAVE_PATH = 'GIF'

def generate_gif(env_name,frame_number,frames_for_gif,reward,path):
    """
    inputs:
    frame_number: integer
    frame_for_gif: A sequence of (210, 160, 3) frames of an Atari game in RGB
    reward: Integer, Total reward of the episode that es ouputted as a gif
    path: String, path where gif is saved
    """
    
    for idx, frame_idx in enumerate(frames_for_gif):
        frames_for_gif[idx] = resize(frame_idx, (420, 320, 3), 
                                     preserve_range=True, order=0).astype(np.uint8)
    #imageio.mimsave('{path}{"{}_frame_{0}_reward_{1}.gif".format(env_name,frame_number, reward)}', 
    #            frames_for_gif, duration=1/30)
    imageio.mimsave('{}_{}_reward{}.gif'.format(path,env_name,reward),frames_for_gif, duration=1/30)
if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('-m','--model',required= False, 
        help = 'Model file to load',default=DEFAULT_MODEL)
    parser.add_argument('-e','--env',default=DEFAULT_ENV_NAME,
                        help='Environment name to use, default='+DEFAULT_ENV_NAME)
    parser.add_argument('-r','--record',help='Directory for video',action='store_true')
    parser.add_argument('--no-vis',default=True, dest='vis',
                        help = 'Deisable visualization',action= 'store_false')
    args = parser.parse_args() 


    env = wrapper.make_env(args.env,episodic_life=True,reward_clipping=False) 
    
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    state = torch.load(args.model,map_location=lambda stg,_: stg)
    net.load_state_dict(state)
    state = env.reset() 
    total_reward = 0.0 
    c = collections.Counter() 
    done_counter  = 0
    frames_for_gif = [] 
    frame_counter = 0
    while True:
        start_ts = time.time() 
        if args.vis:
            env.render(mode='human')
        if args.record:
            frames_for_gif.append(env.render(mode='rgb_array'))
        state_v = torch.tensor(np.array([state],copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        c[action] +=1 
        
        state, reward,done, _ = env.step(action)
        total_reward += reward 
        print('done : {} reward:{} toatal reward:{}'.format(done,reward,total_reward))
        if done:
            done_counter += 1 
            if done_counter%GAME_LIFT == 0:
                break
            else:
                env.reset()
        if args.vis:
            delta = 1/FPS - (time.time()- start_ts)
            if delta > 0:
                time.sleep(delta)
    print('Total Reward : %.2f' % total_reward)
    print('Action counts:', c)
    if args.record:
        if total_reward > 100:
            generate_gif(args.env,len(frames_for_gif),frames_for_gif,total_reward,GIF_SAVE_PATH)
        env.env.close() 