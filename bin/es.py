#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/snolfi/evorobotpy
   and has been written by Stefano Nolfi and Paolo Pagliuca, stefano.nolfi@istc.cnr.it, paolo.pagliuca@istc.cnr.it

   es.py runs an evolutionary expriment or post-evaluate an evolved robot/s
   type python3 es.py for help

   Requires policy.py, evoalgo.py, and salimans.py
   Also requires the net.so library that can be obtained by compiling with cython the following files contained in the ./lib directory:
   evonet.cpp, evonet.h, utilities.cpp, utilities.h, net.pxd, net.pyx and setupevonet.py   
   with the commands: cd ./evorobotpy/lib; python3 setupevonet.py build_ext –inplace; cp net*.so ../bin 
"""


import numpy as np
import configparser
import sys
import os


# global variables
scriptdirname = os.path.dirname(os.path.realpath(__file__))  # Directory of the script .py
#sys.path.insert(0, scriptdirname) # add the diretcory to the path
#cwd = os.getcwd() # directoy from which the script has been lanched
#sys.path.insert(0, cwd) add the directory to the path
filedir = None                          # Directory used to save files
center = None                           # the solution center
sample = None                           # the solution samples
environment = None                      # the problem 
stepsize = 0.01                         # the learning stepsize
noiseStdDev = 0.02                      # the perturbation noise
sampleSize = 20                         # number of samples
wdecay = 0                              # wether we usse weight decay
sameenvcond = 0                         # whether population individuals experience the same conditions
maxsteps = 1000000                      # total max number of steps
evalCenter = 1                          # whether we evaluate the solution center
saveeach = 60                           # number of seconds after which we save data
saveeachg = 0                           # save pop data each n generations
fromgeneration = 0                      # start from generation   
nrobots = 1                             # number of robots
heterogeneous = 0                       # whether the parameters of robots are heterogeneous
algoname = "Salimans"                   # evolutionary algorithm

# Parse the [ADAPT] section of the configuration file
def parseConfigFile(filename):
    global maxsteps
    global envChangeEvery
    global environment
    global fullyRandom
    global stepsize
    global noiseStdDev
    global sampleSize
    global wdecay
    global sameenvcond
    global evalCenter
    global saveeach
    global nrobots
    global heterogeneous
    global algoname
    global saveeachg
    global fromgeneration

    if os.path.isfile(filename):

        config = configparser.ConfigParser()
        config.read(filename)

        # Section EVAL
        options = config.options("ADAPT")
        for o in options:
            found = 0
            if o == "nrobots":
                nrobots = config.getint("ADAPT","nrobots")
                found = 1
            if o == "heterogeneous":
                heterogeneous = config.getint("ADAPT","heterogeneous")
                found = 1
            if o == "maxmsteps":
                maxsteps = config.getint("ADAPT","maxmsteps") * 1000000
                found = 1
            if o == "environment":
                environment = config.get("ADAPT","environment")
                found = 1
            if o == "stepsize":
                stepsize = config.getfloat("ADAPT","stepsize")
                found = 1
            if o == "noisestddev":
                noiseStdDev = config.getfloat("ADAPT","noiseStdDev")
                found = 1
            if o == "samplesize":
                sampleSize = config.getint("ADAPT","sampleSize")
                found = 1
            if o == "wdecay":
                wdecay = config.getint("ADAPT","wdecay")
                found = 1
            if o == "sameenvcond":
                sameenvcond = config.getint("ADAPT","sameenvcond")
                found = 1
            if o == "evalcenter":
                evalCenter = config.getint("ADAPT","evalcenter")
                found = 1
            if o == "saveeach":
                saveeach = config.getint("ADAPT","saveeach")
                found = 1
            if o == "saveeachg":
                saveeachg = config.getint("ADAPT","saveeachg")
                found = 1
            if o == "fromgeneration":
                saveeachg = config.getint("ADAPT","fromgeneration")
                found = 1
            if o == "algo":
                algoname = config.get("ADAPT","algo")
                found = 1
              
            if found == 0:
                print("\033[1mOption %s in section [ADAPT] of %s file is unknown\033[0m" % (o, filename))
                sys.exit()
    else:
        print("\033[1mERROR: configuration file %s does not exist\033[0m" % (filename))
        sys.exit()

def helper():
    print("Main()")
    print("Program Arguments: ")
    print("-f [filename]             : the file containing the parameters shown below (mandatory)")
    print("-s [integer]              : the number used to initialize the seed")
    print("-n [integer]              : the number of replications to be run")
    print("-a [algorithm]            : the algorithm: CMAES, Salimans, xNES, sNES, or SSS (default Salimans)")
    print("-t [filename]             : the .npy file containing the policy to be tested")
    print("-T [filename]             : the .npy file containing the policy to be tested, display neurons")    
    print("-d [directory]            : the directory where all output files are stored (default current dir)")
    print("-tf                       : use tensorflow policy (valid only for gym and pybullet")
    print("")
    print("The .ini file contains the following [ADAPT] and [POLICY] parameters:")
    print("[ADAPT]")
    print("environment [string]      : environment (default 'CartPole-v0'")
    print("maxmsteps [integer]       : max number of (million) steps (default 1)")
    print("sampleSize [integer]      : number of samples (default 20)")
    print("stepsize [float]          : learning stepsize (default 0.01)")
    print("noiseStdDev [float]       : samples noise (default 0.02)")
    print("wdecay [0/2]              : weight decay (default 0), 1 = L1, 2 = L2")
    print("sameenvcond [0/1]         : samples experience the same environmental conditions")
    print("evalCenter [0/1]          : whether or not centroid is evaluated (default 1)")
    print("saveeach [integer]        : save data each n minutes (default 60)")
    print("saveeachg [integer]       : save pop data each n generation (default 0)")
    print("fromgeneration [integer]  : restart from generation n (default 0)")
    print("nrobots [integer]         : number of robots (default 1)")
    print("heterogeneous [integer]   : whether robots are heterogeneous (default 0)")
    print("algo [string]             : adaptive algorithm (default 'Salimans') CMAES, xNES, sNES, pepg, SSS, coevo2" )
    print("[POLICY]")
    print("ntrials [integer]         : number of evaluation episodes (default 1)")
    print("nttrials [integer]        : number of post-evaluation episodes (default 0)")
    print("maxsteps [integer]        : number of evaluation steps [for EREnvs only] (default 1000)")
    print("nhiddens [integer]        : number of hidden x layer (default 50)")
    print("nlayers [integer]         : number of hidden layers (default 1)")
    print("bias [0/1]                : whether we have biases (default 0)")
    print("out_type [integer]        : type of output: 1=logistic, 2=tanh, 3=linear, 4=binary (default 2)")
    print("nbins [integer]           : number of bins 1=no-beans (default 1)")
    print("architecture [0/1/2/3]    : network architecture 0=feedforward 1=recurrent 2=fullrecurrent 3=lstm recurrent (default 0)")
    print("afunction [1/2/3]         : the activation function of neurons 1=logistic 2=tanh 3=linear (default 2)")
    print("winit [0/1/2]             : weight initialization 0=xavier 1=norm incoming 2=uniform (default 0)")
    print("action_noise [0/1/2]      : action noise 0=none, 1=gaussian 2=gaussian-parametric (default 0)")
    print("action_noise_range        : action noise range (default 0.01)")   
    print("normalized [0/1]          : whether or not the input observations are normalized (default 1)")
    print("clip [0/1]                : whether we clip observation in [-5,5] (default 0)")
    print("")
    sys.exit()


def main(argv):
    global maxsteps
    global environment
    global filedir
    global saveeach
    global nrobots
    global algoname

    argc = len(argv)

    # if called without parameters display help information
    if (argc == 1):
        helper()
        sys.exit(-1)

    # Default parameters:
    filename = None         # configuration file
    cseed = 1               # seed
    nreplications = 1       # nreplications
    filedir = './'          # directory
    testfile = None         # file containing the policy to be tested
    test = 0                # whether we rewant to test a policy (1=show behavior, 2=show neurons)
    displayneurons = 0      # whether we want to display the activation state of the neurons
    useTf = False           # whether we want to use tensorflow to implement the policy
    
    i = 1
    while (i < argc):
        if (argv[i] == "-f"):
            i += 1
            if (i < argc):
                filename = argv[i]
                i += 1
        elif (argv[i] == "-s"):
            i += 1
            if (i < argc):
                cseed = int(argv[i])
                i += 1
        elif (argv[i] == "-n"):
            i += 1
            if (i < argc):
                nreplications = int(argv[i])
                i += 1
        elif (argv[i] == "-a"):
            i += 1
            if (i < argc):
                algorithm = argv[i]
                i += 1
        elif (argv[i] == "-t"):
            i += 1
            test = 1
            if (i < argc):
                testfile = argv[i]
                i += 1
        elif (argv[i] == "-T"):
            i += 1
            test = 2
            displayneurons = 1
            if (i < argc):
                testfile = argv[i]
                i += 1   
        elif (argv[i] == "-d"):
            i += 1
            if (i < argc):
                filedir = argv[i]
                i += 1
        elif (argv[i] == "-tf"):
            i += 1
            useTf = True
        else:
            # We simply ignore the argument
            print("\033[1mWARNING: unrecognized argument %s \033[0m" % argv[i])
            i += 1

    # load the .ini file
    if filename is not None:
        parseConfigFile(filename)
    else:
        print("\033[1mERROR: You need to specify an .ini file\033[0m" % filename)
        sys.exit(-1)
    # if a directory is not specified, we use the current directory
    if filedir is None:
        filedir = scriptdirname

    # check whether the user specified a valid algorithm
    availableAlgos = ('CMAES','Salimans','xNES', 'sNES','SSS', 'pepg', 'coevo2', 'coevo2r')
    if algoname not in availableAlgos:
        print("\033[1mAlgorithm %s is unknown\033[0m" % algoname)
        print("Please use one of the following algorithms:")
        for a in availableAlgos:
            print("%s" % a)
        sys.exit(-1)

    print("Environment %s nreplications %d maxmsteps %dm " % (environment, nreplications, maxsteps / 1000000))
    env = None
    policy = None
    
    # Evorobot Environments (we expect observation and action made of numpy array of float32)
    if "Er" in environment:
        ErProblem = __import__(environment)
        env = ErProblem.PyErProblem()
        # Create a new doublepole object
        #action_space = spaces.Box(-1., 1., shape=(env.noutputs,), dtype='float32')
        #observation_space = spaces.Box(-np.inf, np.inf, shape=(env.ninputs,), dtype='float32')
        ob = np.arange(env.ninputs * nrobots, dtype=np.float32)
        ac = np.arange(env.noutputs * nrobots, dtype=np.float32)
        done = np.arange(1, dtype=np.int32)
        env.copyObs(ob)
        env.copyAct(ac)
        env.copyDone(done)      
        from policy import ErPolicy
        policy = ErPolicy(env, env.ninputs, env.noutputs, env.low, env.high, ob, ac, done, filename, cseed, nrobots, heterogeneous, test)

    # Bullet environment (we expect observation and action made of numpy array of float32)
    if "Bullet" in environment:
        import gym
        from gym import spaces
        import pybullet
        import pybullet_envs
        env = gym.make(environment)
        # Define the objects required (they depend on the environment)
        ob = np.arange(env.observation_space.shape[0], dtype=np.float32)
        ac = np.arange(env.action_space.shape[0], dtype=np.float32)
        # Define the policy
        from policy import BulletPolicy
        policy = BulletPolicy(env, env.observation_space.shape[0], env.action_space.shape[0], env.action_space.low[0], env.action_space.high[0], ob, ac, filename, cseed, nrobots, heterogeneous, test)

   # Gym environment (we expect observation and action made of numpy array of float64 or discrete actions)
    if (not "Bullet" in environment) and (not "Er" in environment):
        import gym
        from gym import spaces
        env = gym.make(environment)
        # Define the objects required (they depend on the environment)
        ob = np.arange(env.observation_space.shape[0], dtype=np.float32)
        if (isinstance(env.action_space, gym.spaces.box.Box)):
            ac = np.arange(env.action_space.shape[0], dtype=np.float32)
        else:
            ac = np.arange(env.action_space.n, dtype=np.float32)            
        # Define the policy
        if (isinstance(env.action_space, gym.spaces.box.Box)):
            from policy import GymPolicy
            policy = GymPolicy(env, env.observation_space.shape[0], env.action_space.shape[0], env.action_space.low[0], env.action_space.high[0], ob, ac, filename, cseed, nrobots, heterogeneous, test)
        else:
            from policy import GymPolicyDiscr
            policy = GymPolicyDiscr(env, env.observation_space.shape[0], env.action_space.n, 0.0, 0.0, ob, ac, filename, cseed, nrobots, heterogeneous, test)           

    policy.environment = environment
    policy.saveeach = saveeach
    
    # Create the algorithm class
    if (algoname == 'CMAES'):
        from cmaes import CMAES
        algo = CMAES(env, policy, cseed, filedir)
    elif (algoname =='Salimans'):
        from salimans import Salimans
        algo = Salimans(env, policy, cseed, filedir)
    elif (algoname == 'xNES'):
        from xnes import xNES
        algo = xNES(env, policy, cseed, filedir)
    elif (algoname == 'sNES'):
        from snes import sNES
        algo = sNES(env, policy, cseed, filedir)
    elif (algoname == 'SSS'):
        from sss import SSS
        algo = SSS(env, policy, cseed, filedir)
    elif (algoname == 'coevo2'):
        from coevo2 import coevo2
        algo = coevo2(env, policy, cseed, filedir)
    elif (algoname == 'coevo2r'):
        from coevo2r import coevo2
        algo = coevo2(env, policy, cseed, filedir)
    elif (algoname == 'pepg'):
        from pepg import pepg
        algo = pepg(env, policy, cseed, filedir)
    # Set evolutionary variables
    algo.setEvoVars(sampleSize, stepsize, noiseStdDev, sameenvcond, wdecay, evalCenter, saveeachg, fromgeneration)

    if (test > 0):
        # test a policy
        print("Run Test: Environment %s testfile %s" % (environment, testfile))
        algo.test(testfile)
    else:
        # run evolution
        if (cseed != 0):
            print("Run Evolve: Environment %s Seed %d Nreplications %d" % (environment, cseed, nreplications))
            for r in range(nreplications):
                algo.run(maxsteps)
                algo.seed += 1
                policy.seed += 1
                algo.reset()
                policy.reset()
        else:
            print("\033[1mPlease indicate the seed to run evolution\033[0m")

if __name__ == "__main__":
    main(sys.argv)
