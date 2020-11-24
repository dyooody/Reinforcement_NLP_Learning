import gym
import torch
import time
import os
import ray
import numpy as np

from tqdm import tqdm
from random import uniform, randint

import io
import base64
from IPython.display import HTML

from dqn_model import DQNModel
from dqn_model import _DQNModel
from memory import ReplayBuffer

import matplotlib.pyplot as plt
import matplotlib
#%matplotlib inline

FloatTensor = torch.FloatTensor


def plot_result(total_rewards ,learning_num, legend):
    print("\nLearning Performance:\n")
    episodes = []
    for i in range(len(total_rewards)):
        episodes.append(i * learning_num + 1)
    
    plt.figure(num = 1)
    fig, ax = plt.subplots()
    plt.plot(episodes, total_rewards)
    plt.title('performance')
    plt.legend(legend)
    plt.xlabel("Episodes")
    plt.ylabel("total rewards")
    #plt.show()
    plt.savefig('rewards.png')


hyperparams_CartPole = {
    'epsilon_decay_steps' : 100000, 
    'final_epsilon' : 0.1,
    'batch_size' : 32, 
    'update_steps' : 10, 
    'memory_size' : 2000, 
    'beta' : 0.99, 
    'model_replace_freq' : 2000,
    'learning_rate' : 0.0003,
    'use_target_model': True
}



ray.shutdown()
ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=500000000, object_store_memory=5000000000)


from memory_remote import ReplayBuffer_remote
from dqn_model import _DQNModel
import torch
from custom_cartpole import CartPoleEnv


# Set the Env name and action space for CartPole
ENV_NAME = 'CartPole_distributed'

# Set result saveing floder
result_floder = ENV_NAME + "_distributed"
result_file = ENV_NAME + "/results.txt"
if not os.path.isdir(result_floder):
    os.mkdir(result_floder)
torch.set_num_threads(12)


# Move left, Move right
ACTION_DICT = {
    "LEFT": 0,
    "RIGHT":1
}
# Register the environment
env = CartPoleEnv()

memory = ReplayBuffer_remote.remote(2000)

@ray.remote    
class DQN_server(object):
    def __init__(self, learning_rate, training_episodes, memory, env,
                 test_interval = 50, batch_size = 32,
                 action_space = len(ACTION_DICT), beta = 0.99):
        
        
        self.env = env
        #self.max_episode_steps = env._max_episode_steps
        
        self.batch_num = training_episodes // test_interval
        self.steps = 0
        
        self.collector_done = False
        self.evaluator_done = False
        self.training_episodes = training_episodes
        self.episode = 0
        #self.esults = []
        self.batch_size = batch_size
        self.privous_q_model = []
        self.results = [0] * (self.batch_num + 1)
        self.result_count = 0
        self.memory = memory
        self.use_target_model = True

        state = env.reset()
        input_len = len(state)
        output_len = action_space
        self.eval_model = DQNModel(input_len, output_len, learning_rate = 0.0003)
        self.target_model = DQNModel(input_len, output_len)

        
        
        self.batch_size = hyper_params['batch_size']
        self.update_steps = hyper_params['update_steps']
        self.model_replace_freq = hyper_params['model_replace_freq']
    
        
    def get_eval_model(self):
        print(self.episode)
        if self.episode >= self.training_episodes:
            self.collector_done = True
            
        return self.collector_done

    def add_episode(self):
        self.episode += 1
        return self.episode

    def update_batch(self):
        if self.collector_done:
            return
        if ray.get(self.memory.__len__.remote()) < self.batch_size or self.steps % self.update_steps != 0:
            return

        batch = ray.get(self.memory.sample.remote(self.batch_size))

        (states, actions, reward, next_states,
         is_terminal) = batch
        
        self.steps += self.update_steps
        states = states
        next_states = next_states
        terminal = FloatTensor([1 if t else 0 for t in is_terminal])
        reward = FloatTensor(reward)
        batch_index = torch.arange(self.batch_size,
                                   dtype=torch.long)
        
        # Current Q Values
        _, q_values = self.eval_model.predict_batch(states)
        q_values = q_values[batch_index, actions]
        
        # Calculate target
        if self.use_target_model:
            actions, q_next = self.target_model.predict_batch(next_states)
        else:
            actions, q_next = self.eval_model.predict_batch(next_states)
            
        #INSERT YOUR CODE HERE --- neet to compute 'q_targets' used below
        q_targets = [0] * self.batch_size
        for i in range(self.batch_size):
            if terminal[i] == 1:
                q_targets[i] = reward[i]
            else:
                max_value = torch.max(q_next, dim = 1).values[i].data
                q_targets[i] = reward[i] + beta * max_value
                
        q_target = FloatTensor(q_targets)

        # update model
        self.eval_model.fit(q_values, q_target)

        if self.episode // test_interval +1 > len(self.privous_q_model) :
            model_id = ray.put(self.eval_model)
            self.privous_q_model.append(model_id)
        return self.steps 
    
    # evalutor
    def add_result(self, result, num):
        #print(num)
        self.results[num] = result
    
    def get_results(self):
        return self.results
    
    def ask_evaluation(self):
        if len(self.privous_q_model) > self.result_count:
            num = self.result_count
            evluation_q_model = self.privous_q_model[num]
            self.result_count += 1
            return evluation_q_model, False, num
        else:
            if self.episode >= self.training_episodes:
                self.evaluator_done = True
            return [], self.evaluator_done, None

    def replace(self):
        self.target_model.replace(self.eval_model)

    def predict(self,state):
        return self.eval_model.predict(state)
        
@ray.remote
def collecting_worker(server, env, memory, batch_size = 32, beta = 0.99,
                    initial_epsilon = 1, final_epsilon = 0.1,
                    epsilon_decay_steps = 100000, test_interval = 50):

    def linear_decrease(initial_value, final_value, curr_steps, final_decay_steps):
        decay_rate = curr_steps / final_decay_steps
        if decay_rate > 1:
            decay_rate = 1
        return initial_value - (initial_value - final_value) * decay_rate

    def explore_or_exploit_policy(state):
        p = uniform(0, 1)
        # Get decreased epsilon
        epsilon = linear_decrease(initial_value = 1, 
                                  final_value = 0.1, 
                                  curr_steps = self_steps, 
                                  final_decay_steps = 100000)
        
        if p < epsilon:
            #return action
            return randint(0, action_space - 1)
        else:
            #return action
            return ray.get(server.predict.remote(state))

    max_episode_steps = env._max_episode_steps    
    episode = 0
    update_steps = 10
    model_replace_freq = 2000
    
    best_reward = 0
    learning = True
    use_target_model = True
    self_steps = 0
    
    while True:
        learn_done = ray.get(server.get_eval_model.remote())
        if learn_done:
            break
        for episode in tqdm(range(test_interval), desc="Training"):
            state = env.reset()
            done = False
            steps = 0
            

            while steps < max_episode_steps and not done:
                #INSERT YOUR CODE HERE
                # add experience from explore-exploit policy to memory
                action = explore_or_exploit_policy(state)
                next_state, reward, done, info = env.step(action)
                memory.add.remote(state, action, reward, next_state, done)
                state = next_state
                
                self_steps += 1
                steps += 1

                
                # update the model every 'update_steps' of experience
                if self_steps % update_steps == 0:
                    self_steps = ray.get(server.update_batch.remote())
                
                # update the target network (if the target network is being used) every 'model_replace_freq' of experiences
                #if self_steps % model_replace_freq == 0:    
                if use_target_model and (self_steps % model_replace_freq == 0):
                    server.replace.remote()
                               
            server.add_episode.remote()

        
@ray.remote
def evaluation_worker(server, env, trials = 30, action_space = len(ACTION_DICT), beta = 0.99): 
    max_episode_steps = env._max_episode_steps
    while True:
        model_id, done, num = ray.get(server.ask_evaluation.remote())
        eval_model = ray.get(model_id)

        if done:
            break
        if eval_model == []:
            continue
        total_reward = 0
        print("evaluate")
        for _ in tqdm(range(trials), desc="Evaluating"):
            state = env.reset()
            done = False
            steps = 0

            while steps < max_episode_steps and not done:
                steps += 1
                action = eval_model.predict(state)
                state, reward, done, info = env.step(action)
                total_reward += reward

        avg_reward = total_reward / trials
        print(avg_reward)
        f = open(result_file, "a+")
        f.write(str(avg_reward) + "\n")
        f.close()

        server.add_result.remote(avg_reward, num)

class distributed_DQN_agent():
    def __init__(self, learning_rate, training_episodes, 
                test_interval = 50, batch_size = 32, cw_num = 12, ew_num = 4,
                action_space = len(ACTION_DICT), beta = 0.99):
        
        self.server = DQN_server.remote(learning_rate, training_episodes, memory, env,
                                        test_interval = test_interval, batch_size = batch_size,
                                        action_space = action_space, beta = beta)
        self.workers_id = []
        self.batch_size = batch_size
        self.cw_num = cw_num
        self.ew_num = ew_num
        self.agent_name = "Distributed DQN"
        
    def learn_and_evaluate(self):
        workers_id = []
        
        #INSERT YOUR CODE HERE
        for _ in range(self.cw_num):
            cw_id = collecting_worker.remote(self.server, env, memory, batch_size = batch_size)
            workers_id.append(cw_id)
            
        for _ in range(self.ew_num):
            ew_id = evaluation_worker.remote(self.server, env, trials = 30, action_space = len(ACTION_DICT), beta = 0.99)
            workers_id.append(ew_id)

        ray.wait(workers_id, len(workers_id))
        return ray.get(self.server.get_results.remote())
    
    
run_time = {}
training_episodes, test_interval = 10000, 50
hyper_params = hyperparams_CartPole
learning_rate = 0.0003
action_space = len(ACTION_DICT)
beta = 0.99
batch_size = 32
#test_interval = 50
update_steps = 10
cw_num = 4
ew_num = 4


start_time = time.time()
distributed_DQN_agent = distributed_DQN_agent(learning_rate, training_episodes, 
                                              test_interval, batch_size,
                                              cw_num, ew_num,
                                              action_space = len(ACTION_DICT), beta = 0.99)

                
total_rewards = distributed_DQN_agent.learn_and_evaluate()
run_time['Distributed DQN agent'] = time.time() - start_time
print("Learning time:\n")
print(run_time['Distributed DQN agent'])
plot_result(total_rewards, test_interval, ["batch_update with target_model"])

