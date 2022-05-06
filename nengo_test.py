from collections import deque
from re import S
import cv2
import gym
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Pool
import nengo
import nengo_dl
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm

EPISODES_TEST = 200

class DQNAgent:
    def __init__(self, env_name):
        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.env.seed(random.randint(0, 1000))  
       
        self.env._max_episode_steps = 1000
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.gamma = 0.95    # discount rate
        
        # EXPLORATION HYPERPARAMETERS for epsilon and epsilon greedy strategy
        self.epsilon = 1.0 
        self.epsilon_min = 0.01 
        self.epsilon_decay = 0.0005  

        self.batch_size = 32
        self.TAU = 0.1 # target network soft update hyperparameter
        
        self.Save_Path = 'Models'
        if not os.path.exists(self.Save_Path): 
            os.makedirs(self.Save_Path)

        self.scores, self.episodes, self.average = [], [], []
        self.Model_name = os.path.join(self.Save_Path, self.env_name+"_DQN_CNN.h5")

        self.ROWS = 160
        self.COLS = 240
        self.FRAME_STEP = 4

        self.image_memory = np.zeros((self.FRAME_STEP, self.ROWS, self.COLS))
        self.state_size = (self.FRAME_STEP, self.ROWS, self.COLS)


    def act(self, state, decay_step, explore):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= (1-self.epsilon_decay)
        explore_probability = self.epsilon
    
        if explore and explore_probability > np.random.rand():
            # Make a random action (exploration)
            return random.randrange(self.action_size), explore_probability
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            # Take the biggest Q value (= the best action)
            return np.argmax(self.model.predict(state)), explore_probability


    def load(self, name):
        self.model = load_model(name)


    def save(self, name):
        self.model.save(name)


    def GetImage(self):
        img = self.env.render(mode='rgb_array')
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_rgb_resized = cv2.resize(img_rgb, (self.COLS, self.ROWS), interpolation=cv2.INTER_CUBIC)
        img_rgb_resized[img_rgb_resized < 255] = 0
        img_rgb_resized = img_rgb_resized / 255

        self.image_memory = np.roll(self.image_memory, 1, axis = 0)
        self.image_memory[0,:,:] = img_rgb_resized
        return np.expand_dims(self.image_memory, axis=0)


    def reset(self):
        self.env.reset()
        for i in range(self.FRAME_STEP):
            state = self.GetImage()
        return state


    def step(self,action):
        next_state, reward, done, info = self.env.step(action)
        next_state = self.GetImage()
        return next_state, reward, done, info
    

    def test(self):
        self.load(self.Model_name)
        self.model.summary()
        scores = np.zeros(EPISODES_TEST)
        for e in range(EPISODES_TEST):
            state = self.reset()
            done = False
            i = 0
            while not done:
                action, explore_probability = self.act(state, 0, False)
                state, reward, done, _ = self.step(action)
                i += 1
                if done:
                    scores[e] = i
                    break
    
        return scores



    def test_ndl(self, num_steps=30, scale_fr=1, synapse=None):
        self.load(self.Model_name)

        nengo_converter = nengo_dl.Converter(
            self.model,
            swap_activations={tf.keras.activations.relu: nengo.SpikingRectifiedLinear()},
            scale_firing_rates=scale_fr,
            synapse=synapse,
        )
        # get input/output objects
        nengo_input = nengo_converter.inputs[self.model.layers[0]]
        nengo_output = nengo_converter.outputs[self.model.layers[8]]

        # build network, load in trained weights, run inference on test images
        with nengo_dl.Simulator(nengo_converter.net, minibatch_size=1, progress_bar=False) as nengo_sim:
            scores = np.zeros(EPISODES_TEST)
            for e in range(EPISODES_TEST):
                state = self.reset()
                done = False
                i = 0
                while not done:
                    # repeat inputs for some number of timesteps
                    tiled_test_images = np.tile(np.reshape(state, (1, 1, -1)), (1, num_steps, 1))

                    data = nengo_sim.predict({nengo_input: tiled_test_images})
                    action = np.argmax(data[nengo_output][:, -1], axis=-1)[0]
                    state, reward, done, _ = self.step(action)
                    i += 1
                    if done:
                        scores[e] = i
                        break
                self.env.close()
            return scores



def test_ann():
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True) 
    env_name = 'CartPole-v1'
    agent = DQNAgent(env_name)
    return ((None, None, None), agent.test())


def test_snn(hp):
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True) 
    env_name = 'CartPole-v1'
    agent = DQNAgent(env_name)
    return (hp, agent.test_ndl(*hp))


def show_results():
    data = np.load('results.npy', allow_pickle=True)
    print(data)

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=3)
    #print(test_ann())


    # Presentation time, scale firing rate, synaptic smoothing

    #hyperparameters = [(50, 1000, 0.01)]
    #[ 50.    1000.     0.01   116.52  70.958 200.   ]

    #first set of hyperparameters, basically nothing worked until sfr was 1000
    #hyperparameters = [( 5,   10,  0.01), (10, 10,  0.01), (50,  10, 0.01), (100,   10, 0.01),
    #                   (50,    1,  0.01), (50, 10,  0.01), (50, 100, 0.01), ( 50, 1000, 0.01),
    #                   (50,   10, 0.001), (50, 10, 0.005), (50,  10, 0.01), ( 50,   10, 0.05),]

    #second set of hyperparameters to evaluate other parameters when sfr=1000
    hyperparameters = [( 5,   1000,  0.01), (10, 1000,  0.01), (50,  1000, 0.01), (100,   1000, 0.01),
                       (50,   1000, 0.001), (50, 1000, 0.005), (50,  1000, 0.01), ( 50,   1000, 0.05)]

    #[  5.     10.      0.01    9.34    0.784 200.   ]
    #[ 10.     10.      0.01   10.939   3.832 198.   ]
    #[ 50.     10.      0.01   15.4     6.782 200.   ]
    #[100.     10.      0.01   10.25    2.582 200.   ]

    #[ 50.      1.      0.01    9.465   0.754 200.   ]
    #[ 50.     10.      0.01    9.63    1.369 200.   ]
    #[ 50.    100.      0.01   13.26    4.664 200.   ]
    #[ 50.    1000.     0.01   130.37  82.562 200.   ]

    #[ 50.     10.      0.001   9.355   0.761 200.   ]
    #[ 50.     10.      0.005   9.36    0.768 200.   ]
    #[ 50.     10.      0.01    9.51    0.995 200.   ]
    #[ 50.     10.      0.05    9.34    0.731 200.   ]
    

    #[   5.    1000.       0.01     9.4      0.748  200.   ]
    #[  10.    1000.       0.01     9.455    0.754  200.   ]
    #[  50.    1000.       0.01   118.305   78.494  200.   ]
    #[ 100.    1000.       0.01   111.815   81.367  200.   ]

    #[  50.    1000.       0.001   84.735   62.879  200.   ]
    #[  50.    1000.       0.005  118.205   88.828  200.   ]
    #[  50.    1000.       0.01   135.665   83.287  200.   ]
    #[  50.    1000.       0.05     9.38     0.745  200.   ]

    if True:
        num_threads = 2
        if EPISODES_TEST/num_threads - int(EPISODES_TEST/num_threads) > 1e-6:
            print("Warning: EPISODES_TEST not cleanly divisible by num_threads, trials may be lower than intended")
        EPISODES_TEST = int(EPISODES_TEST/num_threads)

        results = []
        for h in hyperparameters:
            h2 = [h for _ in range(num_threads)]
            with Pool(processes=num_threads) as pool:
                partials = np.array(pool.map(test_snn, h2), dtype=object)
                scores = np.concatenate(partials[:,1])
                h_results = np.concatenate((h,[scores.mean(), scores.std(), EPISODES_TEST*num_threads]))
                print(h_results)
                results.append(h_results)
        with open("results.npy", "wb") as f:
            np.save(f, results, allow_pickle=True)
        
    show_results()
