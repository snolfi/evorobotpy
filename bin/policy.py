#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/snolfi/evorobotpy
   and has been written by Stefano Nolfi and Paolo Pagliuca, stefano.nolfi@istc.cnr.it, paolo.pagliuca@istc.cnr.it

   policy.py takes care of the creation and evaluation of the policy
   Requires the net.so library that can be obtained by compiling with cython the following files contained in the ./lib directory:
   evonet.cpp, evonet.h, utilities.cpp, utilities.h, net.pxd, net.pyx and setupevonet.py
   with the commands: cd ./evorobotpy/lib; python3 setupevonet.py build_ext –inplace; cp net*.so ../bin
   Also requires renderWorld.py to display neurons and to display the behavior of Er environments

"""
import net
import numpy as np
import configparser
import time

class Policy(object):
    def __init__(self, env, ninputs, noutputs, low, high, ob, ac, filename, seed, nrobots, heterogeneous, test):
        # Copy environment
        self.env = env
        self.seed = seed
        self.rs = np.random.RandomState(seed)
        self.ninputs = ninputs 
        self.noutputs = noutputs
        self.test = test
        # Initialize parameters to default values
        self.ntrials = 1     # evaluation triala
        self.nttrials = 0    # post-evaluation trials
        self.maxsteps = 1000 # max number of steps (used from ERPolicy only)
        self.nhiddens = 50   # number of hiddens
        self.nhiddens2 = 0   # number of hiddens of the second layer 
        self.nlayers = 1     # number of hidden layers 
        self.bias = 0        # whether we have biases
        self.out_type = 2    # output type (1=logistic,2=tanh,3=linear,4=binary)
        self.architecture =0 # Feed-forward, recurrent, or full-recurrent network
        self.afunction = 2   # activation function
        self.nbins = 1       # number of bins 1=no-beans
        self.winit = 0       # weight initialization: Xavier, normc, uniform
        self.action_noise = 0# whether we apply noise to actions
        self.action_noise_range = 0.01 # action noise range
        self.normalize = 0   # Do not normalize observations
        self.clip = 0        # clip observation
        self.displayneurons=0# Gym policies can display or the robot or the neurons activations
        self.wrange = 1.0    # weight range, used in uniform initialization only
        # Read configuration file
        self.readConfig(filename)
        # Display info
        print("Evaluation: Episodes %d Test Episodes %d MaxSteps %d" % (self.ntrials, self.nttrials, self.maxsteps))
        # Initialize the neural network
        self.nn = net.PyEvonet(nrobots, heterogeneous, self.ninputs, (self.nhiddens * self.nlayers), self.noutputs, self.nlayers, self.nhiddens2, self.bias, self.architecture, self.afunction, self.out_type, self.winit, self.clip, self.normalize, self.action_noise, self.action_noise_range, self.wrange, self.nbins, low, high)
        # Initialize policy parameters
        self.nparams = self.nn.computeParameters()
        self.params = np.arange(self.nparams, dtype=np.float64)
        # Initialize normalization vector
        if (self.normalize == 1):
            self.normvector = np.arange(self.ninputs*2, dtype=np.float64)
        else:
            self.normvector = None
        # allocate neuron activation vector
        if (self.nbins == 1):
            self.nact = np.arange((self.ninputs + (self.nhiddens * self.nlayers) + self.noutputs) * nrobots, dtype=np.float64)
        else:
            self.nact = np.arange((self.ninputs + (self.nhiddens * self.nlayers) + (self.noutputs * self.nbins)) * nrobots, dtype=np.float64)            
        # Allocate space for observation and action
        self.ob = ob
        self.ac = ac
        # Copy pointers
        self.nn.copyGenotype(self.params)
        self.nn.copyInput(self.ob)
        self.nn.copyOutput(self.ac)
        self.nn.copyNeuronact(self.nact)
        if (self.normalize == 1):
            self.nn.copyNormalization(self.normvector)
        # Initialize weights
        self.nn.seed(self.seed)
        self.nn.initWeights()


    def reset(self):
        self.nn.seed(self.seed)
        self.nn.initWeights()
        if (self.normalize == 1):
            self.nn.resetNormalizationVectors()

    # virtual function, implemented in sub-classes
    def rollout(self, render=False, timestep_limit=None, seed=None):
                raise NotImplementedError

    def set_trainable_flat(self, x):
        self.params = np.copy(x)
        self.nn.copyGenotype(self.params)

    def get_trainable_flat(self):
        return self.params


    def readConfig(self, filename):
        # parse the [POLICY] section of the configuration file
        config = configparser.ConfigParser()
        config.read(filename)
        options = config.options("POLICY")
        for o in options:
          found = 0
          if (o == "ntrials"):
              self.ntrials = config.getint("POLICY","ntrials")
              found = 1
          if (o == "nttrials"):
              self.nttrials = config.getint("POLICY","nttrials")
              found = 1
          if (o == "maxsteps"):
              self.maxsteps = config.getint("POLICY","maxsteps")
              found = 1
          if (o == "nhiddens"):
              self.nhiddens = config.getint("POLICY","nhiddens")
              found = 1
          if (o == "nhiddens2"):
              self.nhiddens2 = config.getint("POLICY","nhiddens2")
              found = 1
          if (o == "nlayers"):
              self.nlayers = config.getint("POLICY","nlayers")
              found = 1
          if (o == "bias"):
              self.bias = config.getint("POLICY","bias")
              found = 1
          if (o == "out_type"):
              self.out_type = config.getint("POLICY","out_type")
              found = 1
          if (o == "nbins"):
              self.nbins = config.getint("POLICY","nbins")
              found = 1
          if (o == "afunction"):
              self.afunction = config.getint("POLICY","afunction")
              found = 1
          if (o == "architecture"):
              self.architecture = config.getint("POLICY","architecture")
              found = 1
          if (o == "winit"):
              self.winit = config.getint("POLICY","winit")
              found = 1
          if (o == "action_noise"):
              self.action_noise = config.getint("POLICY","action_noise")
              found = 1
          if (o == "action_noise_range"):
              self.action_noise_range = config.getfloat("POLICY","action_noise_range")
              found = 1
          if (o == "normalize"):
              self.normalize = config.getint("POLICY","normalize")
              found = 1
          if (o == "clip"):
              self.clip = config.getint("POLICY","clip")
              found = 1
          if (o == "wrange"):
              self.wrange = config.getint("POLICY","wrange")
              found = 1  
          if (found == 0):
              print("\033[1mOption %s in section [POLICY] of %s file is unknown\033[0m" % (o, filename))
              sys.exit()

    @property
    def get_seed(self):
        return self.seed

# Bullet use float values for observation and action vectors
# The policy communicate the pointer to the new vector each timestep since pyBullet create a new vector each time
# Use renderWorld to show neurons
class BulletPolicy(Policy):
    def __init__(self, env, ninputs, noutputs, low, high, ob, ac, filename, seed, nrobots, heterogeneous, test):
        Policy.__init__(self, env, ninputs, noutputs, low, high, ob, ac, filename, seed, nrobots, heterogeneous, test)                            
    
    # === Rollouts/training ===
    def rollout(self, ntrials, render=False, timestep_limit=None, seed=None):
        rews = 0.0
        steps = 0
        # initialize the render for showing the activation of the neurons
        if (self.test == 2):
            self.objs = np.arange(10, dtype=np.float64)   
            self.objs[0] = -1
            import renderWorld
        # To ensure replicability (we always pass a valid seed, even if fully-random evaluation is going to be run)
        if seed is not None:
            self.env.seed(seed)
            self.nn.seed(seed)
        # Loop over the number of trials
        for trial in range(ntrials):
            # if normalize=1, occasionally we store data for input normalization
            if self.normalize:
                if np.random.uniform(low=0.0, high=1.0) < 0.01:
                    normphase = 1
                    self.nn.normphase(1)
                else:
                    normphase = 0
            # Reset environment
            self.ob = self.env.reset()
            # Reset network
            self.nn.resetNet()
            # Reward for current trial
            crew = 0.0
            # Reset episode-reward and step counter for current trial
            rew = 0
            t = 0
            while t < self.maxsteps:
                # Copy the input in the network
                self.nn.copyInput(self.ob)
                # Activate network
                self.nn.updateNet()
                # Perform a step
                self.ob, r, done, _ = self.env.step(self.ac)
                rew += r
                t += 1
                if (self.test > 0):
                    if (self.test == 1):
                        self.env.render(mode="human")
                        time.sleep(0.05)
                    if (self.test == 2):
                        info = 'Trial %d Step %d Fit %.2f %.2f' % (trial, t, r, rew)
                        renderWorld.update(self.objs, info, self.ob, self.ac, self.nact)
                if done:
                    break
            if (self.test > 0):
                print("Trial %d Fit %.2f Steps %d " % (trial, rew, t))
            # if we normalize, we might need to stop store data for normalization
            if self.normalize and normphase > 0:
                self.nn.normphase(0)
            # Update steps
            steps += t
            rews += rew
        # Normalize reward by the number of trials
        rews /= ntrials
        if (self.test > 0 and ntrials > 1):
            print("Average Fit %.2f Steps %.2f " % (rews, steps/float(ntrials)))
        return rews, steps
    
# standard gym policy use double observation and action vectors and recreate the observation vector each step
# consequently we convert the observation vector in double and we communicate the pointer to evonet each step
# Use renderWorld to show neurons
class GymPolicy(Policy):
    def __init__(self, env, ninputs, noutputs, low, high, ob, ac, filename, seed, nrobots, heterogeneous, test):
        Policy.__init__(self, env, ninputs, noutputs, low, high, ob, ac, filename, seed, nrobots, heterogeneous, test)
    
    # === Rollouts/training ===
    def rollout(self, ntrials, render=False, timestep_limit=None, seed=None):
        rews = 0.0
        steps = 0
        # initialize the render for showing the activation of the neurons
        if (self.test == 2):
            import renderWorld
            self.objs = np.arange(10, dtype=np.float64)   
            self.objs[0] = -1 
        # To ensure replicability (we always pass a valid seed, even if fully-random evaluation is going to be run)
        if seed is not None:
            self.env.seed(seed)
            self.nn.seed(seed)
        # Loop over the number of trials
        for trial in range(ntrials):
            # if normalize=1, occasionally we store data for input normalization
            if self.normalize:
                if np.random.uniform(low=0.0, high=1.0) < 0.01:
                    normphase = 1
                    self.nn.normphase(1)
                else:
                    normphase = 0
            # Reset environment
            self.ob = self.env.reset()
            # Reset network
            self.nn.resetNet()
            # Reset episode-reward and step counter for current trial
            rew = 0.0
            t = 0
            while t < self.maxsteps:
                # Copy the input in the network
                self.nn.copyInput(np.float32(self.ob))
                # Activate network
                self.nn.updateNet()
                # Perform a step
                self.ob, r, done, _ = self.env.step(self.ac)
                # Append the reward
                rew += r
                t += 1
                if (self.test > 0):
                    if (self.test == 1):
                        self.env.render()
                        time.sleep(0.05)
                    if (self.test == 2):
                        info = 'Trial %d Step %d Fit %.2f %.2f' % (trial, t, r, rew)
                        renderWorld.update(self.objs, info, self.ob, self.ac, self.nact)
                if done:
                    break
            if (self.test > 0):
                print("Trial %d Fit %.2f Steps %d " % (trial, rew, t))
            # if we normalize, we might need to stop store data for normalization
            if self.normalize and normphase > 0:
                self.nn.normphase(0)
            # Update steps
            steps += t
            rews += rew
        # Normalize reward by the number of trials
        rews /= ntrials
        if (self.test > 0 and ntrials > 1):
            print("Average Fit %.2f Steps %.2f " % (rews, steps/float(ntrials)))
        return rews, steps

# standard gym policy use double observation and action vectors and recreate the observation vector each step
# consequently we convert the observation vector in double and we communicate the pointer to evonet each step
# in addition we convert the action vector into an integer since this policy is used for discrete output environment
# Use renderWorld to show neurons
class GymPolicyDiscr(Policy):
    def __init__(self, env, ninputs, noutputs, low, high, ob, ac, filename, seed, nrobots, heterogeneous, test):
        Policy.__init__(self, env, ninputs, noutputs, low, high, ob, ac, filename, seed, nrobots, heterogeneous, test)
    
    # === Rollouts/training ===
    def rollout(self, ntrials, render=False, timestep_limit=None, seed=None):
        rews = 0.0
        steps = 0
        # initialize the render for showing the activation of the neurons
        if (self.test == 2):
            import renderWorld
            self.objs = np.arange(10, dtype=np.float64)   
            self.objs[0] = -1 
        # To ensure replicability (we always pass a valid seed, even if fully-random evaluation is going to be run)
        if seed is not None:
            self.env.seed(seed)
            self.nn.seed(seed)
        # Loop over the number of trials
        for trial in range(ntrials):
            # if normalize=1, occasionally we store data for input normalization
            if self.normalize:
                if np.random.uniform(low=0.0, high=1.0) < 0.01:
                    normphase = 1
                    self.nn.normphase(1)
                else:
                    normphase = 0
            # Reset environment
            self.ob = self.env.reset()
            # Reset network
            self.nn.resetNet()
            # Reset episode-reward and step counter for current trial
            rew = 0.0
            t = 0
            while t < self.maxsteps:
                # Copy the input in the network
                self.nn.copyInput(np.float32(self.ob))
                # Activate network
                self.nn.updateNet()
                # Convert the action array into an integer
                action = np.argmax(self.ac)
                # Perform a step
                self.ob, r, done, _ = self.env.step(action)
                # Append the reward
                rew += r
                t += 1
                if render:
                    if (self.test == 1):
                        self.env.render()
                        time.sleep(0.05)
                    if (self.test == 2):
                        info = 'Trial %d Step %d Fit %.2f %.2f' % (trial, t, r, rew)
                        renderWorld.update(self.objs, info, self.ob, self.ac, self.nact)
                if done:
                    break
            if (self.test > 0):
                print("Trial %d Fit %.2f Steps %d " % (trial, rew, t))
            # if we normalize, we might need to stop store data for normalization
            if self.normalize and normphase > 0:
                self.nn.normphase(0)
            # Update steps
            steps += t
            rews += rew
        # Normalize reward by the number of trials
        rews /= ntrials
        if (self.test > 0 and ntrials > 1):
            print("Average Fit %.2f Steps %d " % (rews, steps/float(ntrials)))
        return rews, steps

# Evorobotpy policies use float observation and action vectors and do not re-allocate the observation vectors
# and use renderWorld to show robots and neurons
class ErPolicy(Policy):
    def __init__(self, env, ninputs, noutputs, low, high, ob, ac, done, filename, seed, nrobots, heterogeneous, test):
        Policy.__init__(self, env, ninputs, noutputs, low, high, ob, ac, filename, seed, nrobots, heterogeneous, test)
        self.done = done
            
    # === Rollouts/training ===
    def rollout(self, ntrials, render=False, timestep_limit=None, seed=None):
        rews = 0.0
        steps = 0
        # To ensure replicability (we always pass a valid seed, even if fully-random evaluation is going to be run)
        if seed is not None:
            self.env.seed(seed)
            self.nn.seed(seed)
        # initialize the render for showing the behavior of the robot/s and the activation of the neurons
        if (self.test > 0):
            self.objs = np.arange(1000, dtype=np.float64) # the environment can contain up to 100 objects to be displayed  
            self.objs[0] = -1
            self.env.copyDobj(self.objs)
            import renderWorld
        # Loop over the number of trials
        for trial in range(ntrials):
            # if normalize=1, occasionally we store data for input normalization
            if self.normalize:
                if np.random.uniform(low=0.0, high=1.0) < 0.01:
                    normphase = 1
                    self.nn.normphase(1)
                else:
                    normphase = 0
            # Reset environment
            self.env.reset()
            # Reset network
            self.nn.resetNet()
            # Reset episode-reward and step counter for current trial
            rew = 0.0
            t = 0
            while t < self.maxsteps:
                # Activate network
                self.nn.updateNet()
                # Perform a step
                rew += self.env.step()
                t += 1
                # Render
                if (self.test > 0):
                    self.env.render()
                    info = 'Trial %d Step %d Fit %.2f %.2f' % (trial, t, rew, rews)
                    renderWorld.update(self.objs, info, self.ob, self.ac, self.nact)
                if self.done:
                    break
            if (self.test > 0):
                print("Trial %d Fit %.2f Steps %d " % (trial, rew, t))
            # if we normalize, we might need to stop store data for normalization
            if self.normalize and normphase > 0:
                self.nn.normphase(0)
            # Update steps
            steps += t
            rews += rew
        # Normalize reward by the number of trials
        rews /= ntrials
        if (self.test > 0 and ntrials > 1):
            print("Average Fit %.2f Steps %.2f " % (rews, steps/float(ntrials)))
        return rews, steps
        
