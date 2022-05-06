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
            #swap_activations={tf.keras.activations.relu: nengo.SpikingRectifiedLinear()},
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
                    #print(data)
                    #print(data[nengo_output][:, -1])
                    #print(np.argmax(data[nengo_output][:, -1], axis=-1))
                    #input()
                    action = np.argmax(data[nengo_output][:, -1], axis=-1)[0]
                    state, reward, done, _ = self.step(action)
                    i += 1
                    if done:
                        scores[e] = i
                        break
                self.env.close()
            return scores








# ToDo use this to show trend across hyperparameters

def plot_spikes(probe, test_data_idx=0, num_neurons=512, dt=0.001):
  """
  Plots the spikes of the layer corresponding to the `probe`.

  Args:
    probe <nengo.probe.Probe>: The probe object of the layer whose spikes are to
                               be plotted.
    test_data_idx <int>: Test image's index for which spikes were generated.
    num_neurons <int>: Number of random neurons for which spikes are to be plotted.
    dt <int>: The duration of each timestep. Nengo-DL's default duration is 0.001s.
  """
  lyr_name = probe.obj.ensemble.label
  spikes_matrix = ndl_mdl_spikes[test_data_idx][lyr_name] * sfr * dt
  neurons = np.random.choice(spikes_matrix.shape[1], num_neurons, replace=False)
  spikes_matrix = spikes_matrix[:, neurons]

  fig, ax = plt.subplots(figsize=(14, 12), facecolor="#00FFFF")
  color = matplotlib.cm.get_cmap('tab10')(0)
  timesteps = np.arange(n_steps)
  for i in range(num_neurons):
    for t in timesteps[np.where(spikes_matrix[:, i] != 0)]:
      ax.plot([t, t], [i+0.5, i+1.5], color=color)

  ax.set_ylim(0.5, num_neurons+0.5)
  ax.set_yticks(list(range(1, num_neurons+1, int(np.ceil(num_neurons/50)))))
  ax.set_xticks(list(range(1, n_steps+1, 10)))
  ax.set_ylabel("Neuron Index")
  ax.set_xlabel("Time in $ms$")
  ax.set_title("Layer: %s" % lyr_name)









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
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=3)
    print(data)

if __name__ == '__main__':
    #print(test_ann())


    # Presentation time, scale firing rate, synaptic smoothing

    hyperparameters = [(50, 1000, 0.01)]
    #hyperparameters = [( 5, 10,  0.01), (10, 10,  0.01), (50,  10, 0.01), (100,   10, 0.01),
    #                   (50,  1,  0.01), (50, 10,  0.01), (50, 100, 0.01), ( 50, 1000, 0.01),
    #                   (50, 10, 0.001), (50, 10, 0.005), (50,  10, 0.01), ( 50,   10, 0.05)]

    if True:
        num_threads = 3
        if EPISODES_TEST/num_threads - int(EPISODES_TEST/num_threads) > 1e-6:
            print("Warning: EPISODES_TEST not cleanly divisible by num_threads, trials may be lower than intended")
        EPISODES_TEST = int(EPISODES_TEST/num_threads)

        results = []
        for h in hyperparameters:
            h2 = [h for _ in range(num_threads)]
            with Pool(processes=num_threads) as pool:
                partials = np.array(pool.map(test_snn, h2), dtype=object)
                scores = np.concatenate(partials[:,1])
                results.append(np.concatenate((h,[scores.mean(), scores.std(), EPISODES_TEST*num_threads])))
                print(results)
        with open("results.npy", "wb") as f:
            np.save(f, results, allow_pickle=True)
        
    show_results()





# ToDo modify probe locations to suit new model architecture

# Get the probes for Input, first Conv, and the Output layers.

#plot_spikes(first_conv_probe)
#plot_spikes(penltmt_dense_probe)

