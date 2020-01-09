#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
   This file belong to https://github.com/snolfi/evorobotpy
   and has been written by Stefano Nolfi and Paolo Pagliuca, stefano.nolfi@istc.cnr.it, paolo.pagliuca@istc.cnr.it

   salimans.py include an implementation of the OpenAI-ES algorithm described in
   Salimans T., Ho J., Chen X., Sidor S & Sutskever I. (2017). Evolution strategies as a scalable alternative to reinforcement learning. arXiv:1703.03864v2

   requires es.py, policy.py, and evoalgo.py 

"""

import numpy as np
from numpy import zeros, dot, sqrt
import math
import time
from evoalgo import EvoAlgo
from utils import ascendent_sort

# Evolve with ES algorithm taken from Salimans et al. (2017)
class Salimans(EvoAlgo):
    def __init__(self, env, policy, seed, filedir):
        EvoAlgo.__init__(self, env, policy, seed, filedir)

    def run(self, maxsteps):

        start_time = time.time()

        # initialize the solution center
        center = self.policy.get_trainable_flat()
        
        # Extract the number of parameters
        nparams = self.policy.nparams
        # setting parameters
        batchSize = self.batchSize
        if batchSize == 0:
            # 4 + floor(3 * log(N))
            batchSize = int(4 + math.floor(3 * math.log(nparams)))
        # Symmetric weights in the range [-0.5,0.5]
        weights = zeros(batchSize)

        ceval = 0                    # current evaluation
        cgen = 0                # current generation
        # Parameters for Adam policy
        m = zeros(nparams)
        v = zeros(nparams)
        epsilon = 1e-08 # To avoid numerical issues with division by zero...
        beta1 = 0.9
        beta2 = 0.999
    
        # RandomState for perturbing the performed actions (used only for samples, not for centroid)
        rs = np.random.RandomState(self.seed)

        print("Salimans: seed %d maxmsteps %d batchSize %d stepsize %lf noiseStdDev %lf wdecay %d sameEnvCond %d nparams %d" % (self.seed, maxsteps / 1000000, batchSize, self.stepsize, self.noiseStdDev, self.wdecay, self.sameenvcond, nparams))

        if (self.fromgeneration > 0):
            cgen = self.fromgeneration
            filename = "S%dG%d.npy" % (self.seed, cgen)
            filedata = np.load(filename)
            filename = "S%dG%dm.npy" % (self.seed, cgen)
            m = np.load(filename)
            filename = "S%dG%dv.npy" % (self.seed, cgen)
            v = np.load(filename)
            fname = "statS%d.npy" % (self.seed)
            self.stat = np.load(fname)
            if (self.policy.normalize == 1):
                filename = "S%dG%dn.npy" % (self.seed, cgen)
                self.policy.normvector = np.load(fname)
                self.policy.nn.setNormalizationVectors()


        # main loop
        elapsed = 0
        while (ceval < maxsteps):
            cgen += 1


            # Extract half samples from Gaussian distribution with mean 0.0 and standard deviation 1.0
            samples = rs.randn(batchSize, nparams)
            # buffer vector for candidate
            candidate = np.arange(nparams, dtype=np.float64)
            # Evaluate offspring
            fitness = zeros(batchSize * 2)
            # If normalize=1 we update the normalization vectors
            if (self.policy.normalize == 1):
                self.policy.nn.updateNormalizationVectors()
            # Reset environmental seed every generation
            self.env.seed(self.policy.get_seed + cgen)
            self.policy.nn.seed(self.policy.get_seed + cgen)
            # Evaluate offspring
            for b in range(batchSize):
                for bb in range(2):
                    if (bb == 0):
                        candidate = center + samples[b,:] * self.noiseStdDev
                    else:
                        candidate = center - samples[b,:] * self.noiseStdDev                            
                    # Set policy parameters 
                    self.policy.set_trainable_flat(candidate) 
                    # Sample of the same generation experience the same environmental conditions
                    if (self.sameenvcond == 1):
                        self.env.seed(self.policy.get_seed + cgen)
                        self.policy.nn.seed(self.policy.get_seed + cgen)
                    # Evaluate the offspring
                    eval_rews, eval_length = self.policy.rollout(self.policy.ntrials, timestep_limit=1000)
                    # Get the fitness
                    fitness[b*2+bb] = eval_rews
                    # Update the number of evaluations
                    ceval += eval_length
                    # Update data if the current offspring is better than current best
                    self.updateBest(fitness[b*2+bb], candidate) 

            # Sort by fitness and compute weighted mean into center
            fitness, index = ascendent_sort(fitness)
            # Now me must compute the symmetric weights in the range [-0.5,0.5]
            utilities = zeros(batchSize * 2)
            for i in range(batchSize * 2):
                utilities[index[i]] = i
            utilities /= (batchSize * 2 - 1)
            utilities -= 0.5
            # Now we assign the weights to the samples
            for i in range(batchSize):
                idx = 2 * i
                weights[i] = (utilities[idx] - utilities[idx + 1]) # pos - neg

            # Evaluate the centroid
            if (self.sameenvcond == 1):
                self.env.seed(self.policy.get_seed + cgen)
                self.policy.nn.seed(self.policy.get_seed + cgen)
            self.policy.set_trainable_flat(center)
            eval_rews, eval_length = self.policy.rollout(self.policy.ntrials, timestep_limit=1000)
            centroidfit = eval_rews
            ceval += eval_length
            # Update data if the centroid is better than current best
            self.updateBest(centroidfit, center)

            # Evaluate generalization
            if (self.policy.nttrials > 0):
                if centroidfit > fitness[batchSize * 2 - 1]:
                    # the centroid is tested for generalization
                    candidate = np.copy(center)
                else:
                    # the best sample is tested for generalization
                    bestsamid = index[batchSize * 2 - 1]
                    if ((bestsamid % 2) == 0):
                        bestid = int(bestsamid / 2)
                        candidate = center + samples[bestid] * self.noiseStdDev
                    else:
                        bestid = int(bestsamid / 2)
                        candidate = center - samples[bestid] * self.noiseStdDev
                self.env.seed(self.policy.get_seed + 100000)
                self.policy.nn.seed(self.policy.get_seed + 100000)
                self.policy.set_trainable_flat(candidate) 
                eval_rews, eval_length = self.policy.rollout(self.policy.nttrials, timestep_limit=1000)
                gfit = eval_rews
                ceval += eval_length
                # eveltually store the new best generalization individual
                self.updateBestg(gfit, candidate)

            # Compute the gradient
            g = 0.0
            i = 0
            while i < batchSize:
                gsize = -1
                if batchSize - i < 500:
                    gsize = batchSize - i
                else:
                    gsize = 500
                g += dot(weights[i:i + gsize], samples[i:i + gsize,:]) # weights * samples
                i += gsize
            # Normalization over the number of samples
            g /= (batchSize * 2)
            # Weight decay
            if (self.wdecay == 1):
                globalg = -g + 0.005 * center
            else:
                globalg = -g
            # ADAM policy
            # Compute how much the center moves
            a = self.stepsize * sqrt(1.0 - beta2 ** cgen) / (1.0 - beta1 ** cgen)
            m = beta1 * m + (1.0 - beta1) * globalg
            v = beta2 * v + (1.0 - beta2) * (globalg * globalg)
            dCenter = -a * m / (sqrt(v) + epsilon)
            # update center
            center += dCenter

            # Compute the elapsed time (i.e., how much time the generation lasted)
            elapsed = (time.time() - start_time)

            # Update information
            self.updateInfo(cgen, ceval, fitness, center, centroidfit, fitness[batchSize * 2 - 1], elapsed, maxsteps)

            # Save centroid and associated vectors
            if (self.saveeachg > 0 and cgen > 0):
                if ((cgen % self.saveeachg) == 0):
                    filename = "S%dG%d.npy" % (self.seed, cgen)
                    np.save(filename, center)
                    filename = "S%dG%dm.npy" % (self.seed, cgen)
                    np.save(filename, m)
                    filename = "S%dG%dv.npy" % (self.seed, cgen)
                    np.save(filename, v)
                    if (self.policy.normalize == 1):
                        filename = "S%dG%dn.npy" % (self.seed, cgen)
                        np.save(filename, self.policy.normvector)  

        # save data
        self.save(cgen, ceval, centroidfit, center, fitness[batchSize * 2 - 1], (time.time() - start_time))

        # print simulation time
        end_time = time.time()
        print('Simulation time: %dm%ds ' % (divmod(end_time - start_time, 60)))

