'''Instructions to run the code:
   To test the performance of trained agent, by loading saved weights file (.h5 file),
   uncomment line 213 --- agent.test_performance() and 
   comment line 210 ----- agent.run() 
   [Also, you can comment line 216 to 227 if you are only testing, because plotting about reward variation
   during training, while just testingdoesn't makes sense and might give some errors]
   To train a new agent, and observe variation of reward in a plot, 
   uncomment line 210 --- agent.run ()
   uncomment lines 216 to 227 
   comment line 213 ----- agent.test_performance()
'''

import gym
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#list to store reward values
Reward_list = []

def Neural_Network(input_shape, action_space):
    '''This function builds neural network to predict Q and choose actions(left(0) or right(1)) that minimizes Q

    Inputs: size of states, size of control or action

    Output: model
    '''
    X_input = Input(input_shape)

    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)

    # Hidden layer with 256 nodes
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)
    
    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    # Output Layer with # of actions: 2 nodes (left, right)
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs = X_input, outputs = X, name='CartPole DQN model')
    model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    model.summary()
    return model

class Deep_Q_Network_Agent:
    '''This class desribes Deep Q-Learning algorithm and provides helper functions to achieve that
    '''
    def __init__(self):
        '''Constructor for class
        '''
        self.env = gym.make('CartPole-v1')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.episode_length = 1000
        self.memory = deque(maxlen=2000)
        
        #Hyper-parameters assignment
        self.alpha = 0.95           # discount rate
        self.epsilon = 1.0          # initial exploration
        self.epsilon_min = 0.001    # minimum exploration
        self.epsilon_decay = 0.999  # rate at which exploration decreases
        self.batch_size = 64
        self.train_start = 1000

        # create main model
        self.model = Neural_Network(input_shape=(self.state_size,), action_space = self.action_size)

    def remember(self, state, action, reward, next_state, done):
        '''This function stores experience in memory (Experience Replay). In addition to that, it 
           reduces exploration at certain rate specified only affter few episodes and ensures that 
           exploration doesn't go below minimum value specified.

        Inputs: current state, current action, current reward, next state, 
                boolean variable to indicate episode end
        '''
        #Storing the experience
        self.memory.append((state, action, reward, next_state, done))
        #To reduce policy gradually, this works well for Deep-Q Network
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def epsilon_greedy_policy(self, state):
        '''This function implementes epsilon greedy policy

        Input: current state

        Output: control for next state
        '''
        if np.random.random() <= self.epsilon:
            #exploration
            return random.randrange(self.action_size)
        else:
            #exploitation
            return np.argmax(self.model.predict(state))

    def replay(self):
        '''This function randomly samples minibatch of experience from memory
        '''
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        #batch prediction
        target = self.model.predict(state)
        #for 2nd Q network
        target_next = self.model.predict(next_state)

        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # According to Deep Q-Network algorithm, choose actions that maximizes Q value
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + self.alpha * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)


    def load(self, name):
        '''This function loads the trained model of agent to test the performance.
        '''
        self.model = load_model(name)
    
    def save(self, name):
        '''This function saves the weights of trained agent.
        '''
        self.model.save(name)
            
    def run(self):
        '''This function traines the Deep Q-Network agent.
        '''
        for e in range(self.episode_length):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                #To display the learning
                self.env.render()
                action = self.epsilon_greedy_policy(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:                   
                    print("episode: {}/{}, score: {}, epsilon: {:.2}".format(e, self.episode_length, i, self.epsilon))
                    Reward_list.append(i)
                    if i == 500:
                        print("Saving trained model as dqn_weights.h5")
                        #saving weights so that performance can be tested
                        self.save("dqn_weights_2.h5")
                        return
                self.replay()
    
    #function to test the performance of trained agent
    def test_performance(self):
        '''This function tests the performance of trained Deep Q-Network agent
        '''
        #loading saved weights file
        self.load("dqn_weights_2.h5")
        for e in range(self.episode_length):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                #to display performance in each time step of each episode
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.episode_length, i))
                    break

if __name__ == "__main__":
    agent = Deep_Q_Network_Agent()

    #Training (Comment next line if you are only testing agent performance)
    #agent.run()

    #Running agent after training and saving weights in .h5 file, to test performace of agent
    agent.test_performance()
    
    #plotting rewards obtained in each episode (Uncomment only if you are training)
    '''x = []
    for k in range(len(Reward_list)):
        x.append(k)
    plt.plot(x,Reward_list, "--", label='scores per episode')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title("Reward variation (Deep Q-Network) ")
    z = np.polyfit(x, Reward_list, 1)
    p = np.poly1d(z)
    plt.plot(x,p(x),"--", label='trend')
    plt.legend()
    plt.show()'''

