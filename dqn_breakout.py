
from ast import parse
from typing import List

from torch.nn.modules import loss
#from typing_extensions import ParamSpec
import wrapper
import dqn_model

import argparse
import time 
import collections
import copy

import numpy as np 
import torch 
import torch.nn as nn
import torch.optim as optim 
from tensorboardX import SummaryWriter

DEFAULT_ENV_NAME = 'BreakoutNoFrameskip-v4'
MEAN_REWARD_BOUND = 250

GAMMA = 0.99
BATCH_SIZE = 32
# replays size larger than 20000 with cause cpu ram issuse 
#FIXME: use other high efficiency container other than collections
REPLAY_SIZE = 20000
REPLAY_START_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000

EPSILON_STAGES= [1000000,2000000]
EPSILON_LIST  = [1,0.1,0.01] 

GAME_LIFE = 5

Experience = collections.namedtuple('Experience',field_names=['state','action','reward','done','new_state'])

class ExperienceBuffer:
    def __init__(self,capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self,experience):
        self.buffer.append(experience)
    
    def sample(self,batch_size):
        indices = np.random.choice(len(self.buffer),batch_size,replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states),np.array(actions),np.array(rewards,dtype=np.float32),\
            np.array(dones, dtype=np.uint8),np.array(next_states) 
    
class Agent: 
    def __init__(self,env,exp_buffer):
        self.env = env 
        self.exp_buffer = exp_buffer
        self._reset() 
    
    def _reset(self):
        self.state = self.env.reset()
        
        self.total_reward = 0.0 
    
    @torch.no_grad()
    def play_step(self,net,epsilon = 0.0, device = 'cpu'):
        done_reward = None
        if np.random.random() < epsilon:
            action = self.env.action_space.sample() 
        else:#epsilion greedy acction 
            state_a = np.array([self.state],copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_val_v = net(state_v)
            _, act_v = torch.max(q_val_v,dim=1)
            action = int(act_v.item())
        
        # do step in the env 
        new_state, reward, is_done,_ = self.env.step(action)
        self.total_reward += reward 

        exp = Experience(self.state,action,reward,is_done,new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done: 
            done_reward = self.total_reward
            self._reset() 
        return done_reward

        
def calc_loss(batch,net,tgt_net,device = 'cpu',double=True,loss_fn = nn.MSELoss()):
    states, actions, rewards, dones, next_states = batch 
    
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    
    state_action_values = net(states_v).gather(1,actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        #add doubel deep q learning action 
        if double:
            next_state_acts  = net(next_states_v).max(1)[1]
            next_state_acts = next_state_acts.unsqueeze(-1)
            next_state_values = tgt_net(next_states_v).gather(1,next_state_acts).squeeze(-1)
        else:
            next_state_values = tgt_net(next_states_v).max(1)[0]
        
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach() 

    exptected_state_action_value = next_state_values * GAMMA + rewards_v

    return loss_fn(state_action_values,exptected_state_action_value)

def load_model(model,model_dir):
    state = torch.load(model_dir,map_location=lambda stg,_: stg)
    model.load_state_dict(state)

class EpsilonScheduler:
    def __init__(self,epsilons:List, epsilon_stages:List):
        """
        epsilons: list of epsilons, eg: [1 0.1 0.01]
        epsilon_stages: list of when epsilon changes, eg[1000,2000]
        """
        assert (len(epsilons)-1) == len(epsilon_stages), \
            'The length of eplison_stages is not compitable with eplisons'
        self.epsilons =  copy.deepcopy(epsilons)
        self.epsilon_stages = copy.deepcopy(epsilon_stages)
        #add 0 to the front of the list
        self.epsilon_stages.insert(0,0)
        self.reset()
        
    def reset(self):
        self.idx = 0 
        self.stage_level = 0 
        self.next_stage = self._to_next_stage()
        self.eps_stable =False 

    def _to_next_stage(self):
        if self.stage_level >= len(self.epsilon_stages)-1:
            self.eps_stable = True 
            next_stage = self.next_stage
        else:
            self.epsilon_threshold = self.epsilon_stages[self.stage_level]
            self.epsilon_next_threshold = self.epsilon_stages[self.stage_level+1]
            self.epsilon_s = self.epsilons[self.stage_level]
            self.epsilon_f = self.epsilons[self.stage_level+1]
            next_stage = self.epsilon_next_threshold
            print('s0: {}, s1:{}, next_stage{},idx{}'.format(self.epsilon_s,self.epsilon_f,next_stage,self.idx))

        self.stage_level += 1 
        return next_stage

    def update_eplison(self):
        if self.eps_stable:
            epsilon = self.epsilons[-1]
        else:
            if self.idx<self.next_stage:
                #Calulate epsilon        
                epsilon = self.epsilon_s - (self.epsilon_s-self.epsilon_f)*\
                    (self.idx-self.epsilon_threshold)/(self.epsilon_next_threshold-self.epsilon_threshold)
            else:
                #update stage and calculae epsilon
                self.next_stage = self._to_next_stage()
                epsilon = self.epsilon_s - (self.epsilon_s-self.epsilon_f)*\
                    (self.idx-self.epsilon_threshold)/(self.epsilon_next_threshold-self.epsilon_threshold)
            self.idx += 1
        return epsilon

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    parser.add_argument('--double',default=False,action="store_true")
    parser.add_argument('--eps',type=float, nargs='+',default=EPSILON_LIST)
    parser.add_argument('--eps_stage',type=int,nargs='+',default=EPSILON_STAGES)
    parser.add_argument('--huber_loss',default=False, action='store_true', help='enable the huber loss')
    parser.add_argument('--load_model',default=False, action='store_true')
    parser.add_argument('--model_dir',default=DEFAULT_ENV_NAME+'-best.dat', help = 'model directory')

    args = parser.parse_args()

    double = args.double
    print('Double Q learning mode: {}'.format('True' if double else 'False'))
    print('The target reward: {}'.format(args.reward))
    
    args.cuda = True
    device = torch.device("cuda" if args.cuda else "cpu")
    env = wrapper.make_env(args.env,norm_frame=True,episodic_life=True,reward_clipping=False)
    

    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)

    if args.load_model:
        load_model(net,args.model_dir)
        load_model(tgt_net,args.model_dir)


    writer = SummaryWriter(comment="-" + args.env)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)

    #epsilon = EPSILON_START
    eps =args.eps
    eps_stage = args.eps_stage
    print('Epsilons:{}, update at {}'.format(eps,eps_stage))
    epsilon_scheduler = EpsilonScheduler(eps,eps_stage)

    huber_loss = args.huber_loss 
    if huber_loss:
        loss_fn = nn.SmoothL1Loss()
    else:
        loss_fn = nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None
    #life_counter = 0 
    #life_mean_reward = 0 
    #life_reward = 0 
    while True:
        frame_idx += 1
        #epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        epsilon = epsilon_scheduler.update_eplison() 
        reward = agent.play_step(net, epsilon, device=device)
        
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            #life_mean_reward += mean_reward
            #total_reward_len = len(total_rewards)
            #if len(total_rewards) % GAME_LIFE == 0:
            print("%d: done %d games, mean reward %.3f, eps %.3f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), mean_reward, epsilon,
                speed
            ))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), args.env + "-best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > args.reward:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net,device=device,loss_fn=loss_fn)
        loss_t.backward()
        optimizer.step()
    writer.close()