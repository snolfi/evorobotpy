/*
 * This file belong to https://github.com/snolfi/evorobotpy
 * Author: Stefano Nolfi, stefano.nolfi@istc.cnr.it

 * evonet.cpp, include an implementation of a neural network policy

 * This file is part of the python module net.so that include the following files:
 * evonet.cpp, evonet.h, utilities.cpp, utilities.h, net.pxd, net.pyx and setupevonet.py
 * And can be compile with cython with the commands: cd ./evorobotpy/lib; python3 setupevonet.py build_ext –inplace; cp net*.so ../bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <string.h>
#include "evonet.h"
#include "utilities.h"

#define MAX_BLOCKS 20
#define MAXN 10000
#define CLIP_VALUE 5.0

// Local random number generator
RandomGenerator* netRng;

// Pointers to genotype, observation, action, neuron activation, normalization vectors
double *cgenotype = NULL;
float *cobservation = NULL;
float *caction = NULL;
double *neuronact = NULL;
double *cnormalization = NULL;

/*
 * logistic activation functiom
 */
double logistic(double f)
{
	return ((double) (1.0 / (1.0 + exp(-f))));
}

/*
 * hyperbolic tangent activation function
 */
double tanh(double f)
{
	if (f > 10.0)
		return 1.0;
	  else if (f < - 10.0)
		 return -1.0;
	    else
         return ((double) ((1.0 - exp(-2.0 * f)) / (1.0 + exp(-2.0 * f))));
	
}

/*
 * linear activation function
 */
double linear(double f)
{
        return (f);
}


// constructor
Evonet::Evonet()
{
	m_ninputs = 0;
	m_nhiddens = 0;
	m_noutputs = 0;
	m_nneurons = (m_ninputs + m_nhiddens + m_noutputs);
	m_nlayers = 1;
	m_bias = 0;
	m_netType = 0;
	m_actFunct = 2;
	m_outType = 0;
	m_wInit = 0;
	m_clip = 0;
	m_normalize = 0;
	m_randAct = 0;
    m_wrange = 1.0;
    m_nbins = 1;
    m_low = -1.0;
    m_high = 1.0;
	
    
	netRng = new RandomGenerator(time(NULL));
	//m_act = new double[MAXN];
	m_netinput = new double[MAXN];
	m_netblock = new int[MAX_BLOCKS * 5];
	m_nblocks = 0;
	m_neurontype = new int[MAXN];

}

/*
 * set the seed
 */
void Evonet::seed(int s)
{
    netRng->setSeed(s);
}

/*
 * initialize the network
 */
Evonet::Evonet(int nnetworks, int heterogeneous, int ninputs, int nhiddens, int noutputs, int nlayers, int nhiddens2, int bias, int netType, int actFunct, int outType, int wInit, int clip, int normalize, int randAct, double randActR, double wrange, int nbins, double low, double high)
{
    
    m_nnetworks = nnetworks;
    m_heterogeneous = heterogeneous;
	m_nbins = nbins;
	if (m_nbins < 1 || m_nbins > 20) // ensures that m_bins is in an appropriate rate
        m_nbins = 1;
    // set variables
	m_ninputs = ninputs;
	m_nhiddens = nhiddens;
	m_noutputs = noutputs;
	if (m_nbins > 1)
	{
	  m_noutputs = noutputs * m_nbins; // we several outputs for each actuator
	  m_noutputsbins = noutputs;        // we store the number of actuators
	}
	m_nneurons = (m_ninputs + m_nhiddens + m_noutputs);
	m_nlayers = nlayers;
	m_bias = bias;
	m_netType = netType;
	m_actFunct = actFunct;
	m_outType = outType;
	m_wInit = wInit;
	m_clip = clip;
	m_normalize = normalize;
	m_randAct = randAct;
	m_randActR = randActR;
    m_low = low;
    m_high = high;
    m_wrange = wrange;
    if (m_netType > 0 && m_nlayers > 1)
       {
        printf("WARNING: NUMBER OF LAYERS FORCED TO 1 SINCE ONLY FEED_FORWARD NETWORKS CAN HAVE MULTIPLE LAYERS");
        m_nlayers = 1;
       }
    // LSTM networks should use fake linear functions for internal neurons
    if (m_netType == 3)
     {
        m_actFunct = 3;
     }
    
	netRng = new RandomGenerator(time(NULL));

    // display info and check parameters are in range
    printf("Network %d->", m_ninputs);
    int l;
    for(l=0; l < nlayers; l++)
        printf("%d->", m_nhiddens / m_nlayers);
    printf("%d ", m_noutputs);
    if (m_netType == 0)
        printf("feedforward ");
    else if (m_netType == 1)
        printf("recurrent ");
    else if (m_netType == 2)
        printf("fully recurrent ");
    else if (m_netType == 3)
        printf("LSTM recurrent ");
    if (m_bias)
        printf("with bias ");
    if (m_actFunct == 1)
        printf("logistic ");
    else if (m_actFunct == 2)
        printf("tanh ");
    else if (m_actFunct == 3)
        printf("linear ");
    else if (m_actFunct == 3)
        printf("binary ");
    switch (m_outType)
    {
        case 1:
            printf("output:logistic ");
            break;
        case 2:
            printf("output:tanh ");
            break;
        case 3:
            printf("output:linear ");
            break;
        case 4:
            printf("output:binary ");
            break;
    }
	if (m_nbins > 1)
        printf("bins: %d", m_nbins);
    if (m_wInit < 0 || m_wInit > 2) // ensure it is in the proper range
        m_wInit = 0;
    if (m_wInit == 0)
        printf("init:xavier ");
    else if (m_wInit == 1)
        printf("init:norm-incoming ");
    else if (m_wInit == 2)
        printf("init:uniform ");
    if (m_normalize < 0 || m_normalize > 1)
        m_normalize = 0;
    if (m_normalize == 1)
        printf("input-normalization ");
    if (m_clip < 0 || m_clip > 1)
        m_clip = 0;
    if (m_clip == 1)
        printf("clip ");
    if (m_randAct < 0 || m_randAct > 2)
        m_randAct = 0;
    if (m_randAct == 1)
        printf("output-noise %.2f ", m_randActR);
    if (m_randAct == 2)
        printf("output-diagonal-gaussian ");  // associated parameters are initialized to 0.0
    printf("\n");
    
	// allocate variables
    m_nblocks = 0;
	//m_act = new double[m_nneurons];
	m_netinput = new double[m_nneurons];
    if (m_netType == 3)
    {
      m_nstate = new double[m_nneurons * m_nnetworks];
      m_pnstate = new double[m_nneurons * m_nnetworks];
    }
	m_netblock = new int[MAX_BLOCKS * 5];
	m_neurontype = new int[m_nneurons];
	// Initialize network architecture
	initNetArchitecture();
	
	// allocate vector for input normalization
	if (normalize == 1)
	  {
	  	m_mean = new double[m_ninputs];  // mean used for normalize
	    m_std = new double[m_ninputs];   // standard deviation used for normalize
	    m_sum = new double[m_ninputs];   // sum of input data used for calculating normalization vectors
	    m_sumsq = new double[m_ninputs]; // squared sum of input data used for calculating normalization vectors
	    m_count = 0.01;                  // to avoid division by 0
	    int i;
		for (i = 0; i < m_ninputs; i++)
	      {
		   m_sum[i] = 0.0;
		   m_sumsq[i] = 0.01;           // eps
		   m_mean[i] = 0.0;
		   m_std[i] = 1.0;
	      }
	  }
}


/*
 * destructor, yet to be implemented
 */
Evonet::~Evonet()
{
}


void Evonet::initNet(char* filename)
{
    
}

// reset the network
void Evonet::resetNet()
{
	int i;
    int n;
    double *neura;
    
    neura = neuronact;
    for (n = 0; n < m_nnetworks; n++)
     {
	  for (i = 0; i < m_nneurons; i++, neura++)
	  {
		*neura = 0.0;
	  }
     }
    if (m_netType == 3)
     {
       double *neurs;
       neurs = m_nstate;
         for (n = 0; n < m_nnetworks; n++)
         {
             for (i = 0; i < m_nneurons; i++, neurs++)
             {
                 *neurs = 0.0;
             }
         }
     }
}

// store pointer to weights vector
void Evonet::copyGenotype(double* genotype)
{
	cgenotype = genotype;
}

// store pointer to observation vector
void Evonet::copyInput(float* input)
{
	cobservation = input;
}

// store pointer to update vector
void Evonet::copyOutput(float* output)
{
	caction = output;
}

// store pointer to neuron activation vector
void Evonet::copyNeuronact(double* na)
{
    neuronact = na;
}

// store pointer to neuron activation vector
void Evonet::copyNormalization(double* no)
{
    cnormalization = no;
}

// update net
void Evonet::updateNet()
{
    double* p;         // free parameters
    double* neurona;   // the activation vector of the current network
    float* cobserv;    // the observation vector of the current network
    float* cact;       // the action vector of the current network
    double* nstate;    // the lstm-state vector of the current network
    double* pnstate;  // the previous lstm-state of the current network
    double* a;
    double* ni;
    int i;
    int t;
    int b;
    int* nbl;
    int* nt;
    int j;
    int n;
    int g;
    double lstm[4]; // gates: forget, input, output, gate-gate
	
	
    if (m_heterogeneous == 1)
       p = cgenotype;
    
    for(n=0, neurona = neuronact, cobserv = cobservation, cact = caction, nstate = m_nstate, pnstate = m_pnstate; n < m_nnetworks; n++, neurona = (neurona + m_nneurons), cobserv = (cobserv + m_ninputs), cact = (cact + m_noutputs))
    {
        // in case of homogeneous networks we use the same parameters for multiple networks
        if (m_heterogeneous == 0)
            p = cgenotype;
        
        // Collect the input for updatig the normalization
        // We do that before eventually clipping (not clear whether this is the best choice)
        if (m_normalize  == 1 && m_normphase == 1)
            collectNormalizationData();
        
        // Normalize input
        if (m_normalize == 1)
        {
            for (j = 0; j < m_ninputs; j++)
                cobserv[j] = (cobserv[j] - m_mean[j]) / m_std[j];
        }
        
        // Clip input values
        if (m_clip == 1)
        {
            for (j = 0; j < m_ninputs; j++)
            {
                if (cobserv[j] < -CLIP_VALUE)
                    cobserv[j] = -CLIP_VALUE;
                if (cobserv[j] > CLIP_VALUE)
                    cobserv[j] = CLIP_VALUE;
                //printf("%.1f ", cobservation[j]);
            }
            //printf("\n");
        }
        
        
        // compute biases
        if (m_bias == 1)
        {
            // Only non-input neurons have bias
            for(i = 0, ni = m_netinput; i < m_nneurons; i++, ni++)
            {
                if (i >= m_ninputs)
                {
                    *ni = *p;
                    p++;
                }
                else
                {
                    *ni = 0.0;
                }
            }
        }
        
        // blocks
        for (b = 0, nbl = m_netblock; b < m_nblocks; b++)
        {
            // connection block
            if (*nbl == 0)
            {
                for(t = 0, ni = (m_netinput + *(nbl + 1)); t < *(nbl + 2); t++, ni++)
                {
                    for(i = 0, a = (neurona + *(nbl + 3)); i < *(nbl + 4);i++, a++)
                    {
                        *ni += *a * *p;
                        p++;
                    }
                }
            }
            
            // LSTM connection block
            if (*nbl == 2)
            {
                int tt;
                for(t = 0, tt = *(nbl + 1); t < *(nbl + 2); t++, tt++)
                    pnstate[tt] = nstate[tt];
                for(t = 0, ni = (m_netinput + *(nbl + 1)), tt = *(nbl + 1); t < *(nbl + 2); t++, ni++, tt++)
                {
//printf("%d) ",tt);
                    for (int g = 0; g < 4; g++)
                    {
                      lstm[g] = 0;
                      for(i = 0, a = (neurona + *(nbl + 3)); i < *(nbl + 4);i++, a++)
                      {
                        lstm[g] += *a * *p;
//printf("%.2f ", *p);
                        p++;
                      }
//printf("%.2f ",lstm[g]);
                      // forget state can be sigmoid(netinput+1.0) where 1.0 is forget_bias
                      switch(g)
                       {
                         case 0: // forget gate
                           //lstm[0] = logistic(lstm[g]);
						   lstm[0] = logistic(lstm[g]+1.0); // forget bias = 1
                           break;
                         case 1: // input gate
                           lstm[1] = logistic(lstm[g]);
                           break;
                         case 2: // output gate
                           lstm[2] = logistic(lstm[g]);
                           break;
                         case 3: // act
                           lstm[3] = tanh(lstm[g]);
                           break;
					   }
//printf("%.2f ",lstm[g]);
                    }
//printf("\n");
                    // state = previous_state * forget_gate + gate_act * input_gate
                    nstate[tt] = pnstate[tt] * lstm[0] + lstm[3] * lstm[1];
                    // output = state * output_gate
                    // we store it in the netinput for simplicity and we use a linear activation function
                    *ni = tanh(nstate[tt]) * lstm[2];
//printf(" -> %.2f %.2f \n", m_nstate[tt], *ni);
                }

            }
            
            // update block
            if (*nbl == 1)
            {
                for(t = *(nbl + 1), a = (neurona + *(nbl + 1)), ni = (m_netinput + *(nbl + 1)), nt = (m_neurontype + *(nbl + 1)); t < (*(nbl + 1) + *(nbl + 2)); t++, a++, ni++, nt++)
                {
                    switch (*nt)
                    {
                        case 0:
                            // input neurons are simple rely units
                            *a = *(cobserv + t);
                            break;
                        case 1:
                            // Logistic
                            *a = logistic(*ni);
                            break;
                        case 2:
                            // Tanh
                            *a = tanh(*ni);
                            break;
                        case 3:
                            // linear
                            *a = linear(*ni);
                            break;
                        case 4:
                            // Binary
                            if (*ni >= 0.5)
                                *a = 1.0;
                            else
                                *a = -1.0;
                            break;
                    }
                }
            }
            nbl = (nbl + 5);
        }
        // store actions
        if (m_nbins == 1)
        {
            int i;
            for (i = 0; i < m_noutputs; i++)
            {
               switch(m_randAct)
               {
                  // standard no-noise
                  case 0:
                    cact[i] = neurona[m_ninputs + m_nhiddens + i];
                    break;
				  // gaussian noise with fixed range
				  case 1:
                    cact[i] = neurona[m_ninputs + m_nhiddens + i] + (netRng->getGaussian(1.0, 0.0) * m_randActR);
                    break;
				  // gaussian noise with parametric range (diagonal-gaussian)
				  case 2:
                    cact[i] = neurona[m_ninputs + m_nhiddens + i] + (netRng->getGaussian(1.0, 0.0) * exp(*p));
                    p++;
                    break;
				}
            }
        }
        else // with bins
        {
            int i = 0;
            int j = 0;
            double ccact;
            int ccidx;
            // For each output, we look for the bin with the highest activation
            for (i = 0; i < m_noutputsbins; i++)
            {
                // Current highest activation
                ccact = -9999.0;
                // Index of the current highest activation
                ccidx = -1;
                for (j = 0; j < m_nbins; j++)
                {
                    if (neurona[m_ninputs + m_nhiddens + ((i * m_nbins) + j)] > ccact)
                    {
                        ccact = neurona[m_ninputs + m_nhiddens + ((i * m_nbins) + j)];
                        ccidx = j;
                    }
                }
                cact[i] = 1.0 / ((double)m_nbins - 1.0) * (double)ccidx * (m_high - m_low) + m_low;
                if (m_randAct == 1)
                    cact[i] += (netRng->getGaussian(1.0, 0.0) * 0.01);
            }
        }
		
	   // we update the neuron-state pointers in case of lstm-architectures
       if (m_netType == 3)
         {
		   nstate = (nstate + m_nneurons);
		   pnstate = (pnstate + m_nneurons);
		 }
		
    }
    
}



// copy the output and eventually add noise
void Evonet::getOutput(float* output)
{
	
    // standard without bins
    if (m_nbins == 1)
    {
     int i;
	 for (i = 0; i < m_noutputs; i++)
	  {
		if (m_randAct == 1)
          output[i] = neuronact[m_ninputs + m_nhiddens + i] + (netRng->getGaussian(1.0, 0.0) * 0.01);
        else
          output[i] = neuronact[m_ninputs + m_nhiddens + i];
	  }
    }
    else // with bins
    {
     int i = 0;
     int j = 0;
     double cact;
     int cidx;
	 // For each output, we look for the bin with the highest activation
     for (i = 0; i < m_noutputsbins; i++)
     {
        // Current highest activation
        cact = -9999.0;
        // Index of the current highest activation
        cidx = -1;
        for (j = 0; j < m_nbins; j++)
        {
            if (m_act[m_ninputs + m_nhiddens + ((i * m_nbins) + j)] > cact)
            {
                cact = m_act[m_ninputs + m_nhiddens + ((i * m_nbins) + j)];
                cidx = j;
            }
        }
        output[i] = 1.0 / ((double)m_nbins - 1.0) * (double)cidx * (m_high - m_low) + m_low;
		if (m_randAct == 1)
		     output[i] += (netRng->getGaussian(1.0, 0.0) * 0.01);
     }
     }

}


// compute the number of required parameters
int Evonet::computeParameters()
{
	int nparams;
	int* nbl;
    int b;

	nparams = 0;
	
	// biases
	if (m_bias)
		nparams += (m_nhiddens + m_noutputs);
    
    // blocks
    for (b = 0, nbl = m_netblock; b < m_nblocks; b++)
    	{
		  // connection block
		  if (*nbl == 0)
              nparams += *(nbl + 2) * *(nbl + 4);
          // LSTM connection block
          if (*nbl == 2)
              nparams += *(nbl + 2) * *(nbl + 4) * 4;
		  nbl = (nbl + 5);
	}
	
	// gaussian output parameters
	if (m_randAct == 2)
	  nparams += m_noutputs;

    if (m_heterogeneous == 1)
        nparams *= m_nnetworks;
    
	m_nparams = nparams;

	return nparams;
}

// initialize the architecture description
void Evonet::initNetArchitecture()
{
	int* nbl;
	int* nt;
	int n;

	m_nblocks = 0;
	nbl = m_netblock;

	// neurons' type
	for (n = 0, nt = m_neurontype; n < m_nneurons; n++, nt++)
      	{
          	if (n < m_ninputs)
                *nt = 0; // Inputs correspond to neuron type 0
          	else
              {
			    if (n < (m_ninputs + m_nhiddens))
                   *nt = m_actFunct; // activation function
			     else
			      *nt = m_outType;  // output activation function
			  }
      	}
    
	// input update block
	*nbl = 1; nbl++;
	*nbl = 0; nbl++;
	*nbl = m_ninputs; nbl++;
	*nbl = 0; nbl++;
	*nbl = 0; nbl++;
	m_nblocks++;
	
    // Fully-recurrent network
	if (m_netType == 2)
	{
	  	// hiddens and outputs receive connections from input, hiddens and outputs
	  	*nbl = 0; nbl++;
	  	*nbl = m_ninputs; nbl++;
	  	*nbl = m_nhiddens + m_noutputs; nbl++;
	  	*nbl = 0; nbl++;
	  	*nbl = m_ninputs + m_nhiddens + m_noutputs; nbl++;
	  	m_nblocks++;
		
	  	// hidden-output update block
	  	*nbl = 1; nbl++;
	  	*nbl = m_ninputs; nbl++;
	  	*nbl = m_nhiddens + m_noutputs; nbl++;
	  	*nbl = 0; nbl++;
	  	*nbl = 0; nbl++;
	  	m_nblocks++;
	}
    // recurrent network with 1 layer
	if (m_netType == 1)
	{
        // input-hidden connections
        *nbl = 0; nbl++;
		*nbl = m_ninputs; nbl++;
		*nbl = m_nhiddens; nbl++;
		*nbl = 0; nbl++;
		*nbl = m_ninputs; nbl++;
		m_nblocks++;
    
        // hidden-hidden connections
		*nbl = 0; nbl++;
        *nbl = m_ninputs; nbl++;
        *nbl = m_nhiddens; nbl++;
        *nbl = m_ninputs; nbl++;
        *nbl = m_nhiddens; nbl++;
        m_nblocks++;
    
		// hidden update block
        *nbl = 1; nbl++;
        *nbl = m_ninputs; nbl++;
        *nbl = m_nhiddens; nbl++;
        *nbl = 0; nbl++;
        *nbl = 0; nbl++;
        m_nblocks++;

		// hidden-output connections
		*nbl = 0; nbl++;
        *nbl = m_ninputs + m_nhiddens; nbl++;
        *nbl = m_noutputs; nbl++;
        *nbl = m_ninputs; nbl++;
        *nbl = m_nhiddens; nbl++;
        m_nblocks++;
      
		// output-output connections
        *nbl = 0; nbl++;
        *nbl = m_ninputs + m_nhiddens; nbl++;
        *nbl = m_noutputs; nbl++;
        *nbl = m_ninputs + m_nhiddens; nbl++;
        *nbl = m_noutputs; nbl++;
        m_nblocks++;
    
        // output update block
        *nbl = 1; nbl++;
        *nbl = m_ninputs + m_nhiddens; nbl++;
        *nbl = m_noutputs; nbl++;
        *nbl = 0; nbl++;
        *nbl = 0; nbl++;
        m_nblocks++;
	}
    // feedforward
    if (m_netType == 0)
	{
		// Feed-forward network
		if (m_nhiddens == 0)
		{
			// Sensory-motor network
			*nbl = 0; nbl++;
		  	*nbl = m_ninputs; nbl++;
		  	*nbl = m_noutputs; nbl++;
		  	*nbl = 0; nbl++;
		  	*nbl = m_ninputs; nbl++;
		  	m_nblocks++;

			// output update block
            *nbl = 1; nbl++;
            *nbl = m_ninputs; nbl++;
		  	*nbl = m_noutputs; nbl++;
            *nbl = 0; nbl++;
            *nbl = 0; nbl++;
            m_nblocks++;
		}
		else
		{
			// input-hidden connections
			if (m_nlayers == 1)
			{
                *nbl = 0; nbl++;
				*nbl = m_ninputs; nbl++;
				*nbl = m_nhiddens; nbl++;
				*nbl = 0; nbl++;
				*nbl = m_ninputs; nbl++;
				m_nblocks++;
				
				// hidden update block
                *nbl = 1; nbl++;
			  	*nbl = m_ninputs; nbl++;
			  	*nbl = m_nhiddens; nbl++;
			  	*nbl = 0; nbl++;
			  	*nbl = 0; nbl++;
			  	m_nblocks++;

				// hidden-output connections
				*nbl = 0; nbl++;
			  	*nbl = m_ninputs + m_nhiddens; nbl++;
			  	*nbl = m_noutputs; nbl++;
			  	*nbl = m_ninputs; nbl++;
			  	*nbl = m_nhiddens; nbl++;
			  	m_nblocks++;

				// output update block
                *nbl = 1; nbl++;
                *nbl = m_ninputs + m_nhiddens; nbl++;
			  	*nbl = m_noutputs; nbl++;
                *nbl = 0; nbl++;
                *nbl = 0; nbl++;
                m_nblocks++;
			}

		}
	}
    // recurrent LSTM, 1 layer
    if (m_netType == 3)
    {
        
        // hiddens receive connections from input and hiddens
        *nbl = 2; nbl++;  // special connection block, type 2
        *nbl = m_ninputs; nbl++;
        *nbl = m_nhiddens; nbl++;
        *nbl = 0; nbl++;
        *nbl = m_ninputs + m_nhiddens; nbl++;
        m_nblocks++;
        
        // hidden update block
        *nbl = 1; nbl++;
        *nbl = m_ninputs; nbl++;
        *nbl = m_nhiddens; nbl++;
        *nbl = 0; nbl++;
        *nbl = 0; nbl++;
        m_nblocks++;
        
        // hidden-output connections
        *nbl = 0; nbl++;
        *nbl = m_ninputs + m_nhiddens; nbl++;
        *nbl = m_noutputs; nbl++;
        *nbl = m_ninputs; nbl++;
        *nbl = m_nhiddens; nbl++;
        m_nblocks++;
        
        // output update block
        *nbl = 1; nbl++;
        *nbl = m_ninputs + m_nhiddens; nbl++;
        *nbl = m_noutputs; nbl++;
        *nbl = 0; nbl++;
        *nbl = 0; nbl++;
        m_nblocks++;
        
    }
    
}

// initialize weights
void Evonet::initWeights()
{
  int i;
  int j;
  int t;
  int b;
  int n;
  int* nbl;
  double range;
  int nnetworks;
  bool LSTM;
    
  // in case of homogeneous networks we need a single set of parameter
  if (m_heterogeneous == 0)
     nnetworks = 1;
    else
     nnetworks = m_nnetworks;
  
  // cparameter
  j = 0;
  for(n=0; n < nnetworks; n++)
    {
    // Initialize biases to 0.0
    if (m_bias)
    {
        // Bias are initialized to 0.0
        for (i = 0; i < (m_nhiddens + m_noutputs); i++)
        {
            cgenotype[j] = 0.0;
            j++;
        }
    }
    // Initialize weights of connection blocks
    for (b = 0, nbl = m_netblock; b < m_nblocks; b++)
    {
        // connection block (2 is for LSTM)
        if (*nbl == 0 || *nbl == 2)
        {
            if (*nbl == 2)
                LSTM = true;
               else
                LSTM = false;
            
            switch (m_wInit)
            {
                // xavier initialization
                // gaussian distribution with range = (radq(2.0 / (ninputs + noutputs))
                case 0:
                    int nin;
                    int nout;
                    // ninput and noutput of the current block
                    nin = *(nbl + 4);
                    nout = *(nbl + 2);
                    // if previous and/or next block include the same receiving neurons we increase ninputs accordingly
                    // connection block are always preceded by update block, so we can check previous values
                    if ((*(nbl + 5) == 0) && ((*(nbl + 1) == *(nbl + 6)) && (*(nbl + 2) == *(nbl + 7))))
                        nin += *(nbl + 9);
                    if ((*(nbl - 5) == 0) && ((*(nbl - 4) == *(nbl + 1)) && (*(nbl - 3) == *(nbl + 2))))
                        nin += *(nbl - 1);
                    // compute xavier range
                    range = sqrt(2.0 / (nin + nout));
                    for (t = 0; t < *(nbl + 2); t++)
                    {
                        for (i = 0; i < *(nbl + 4); i++)
                        {
                            if (LSTM)
                            {
                                for (int ii=0; ii < 4; ii++)
                                {
                                 cgenotype[j] = netRng->getGaussian(range, 0.0);
                                 j++;
                                }
                            }
                            else
                            {
                              cgenotype[j] = netRng->getGaussian(range, 0.0);
                              j++;
                            }
                        }
                    }
                break;
                // normalization of incoming weights as in salimans et al. (2017)
                // in case of linear output, use a smaller range for the last layer
                // we assume that the last layer corresponds to the last connection block followed by the last update block
                // compute the squared sum of gaussian numbers in order to scale the weights
                // equivalent to the following python code for tensorflow:
                // out = np.random.randn(*shape).astype(np.double32)
                // out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
                // where randn extract samples from Gaussian distribution with mean 0.0 and std 1.0
                // std is either 1.0 or 0.01 depending on the layer
                // np.square(out).sum(axis=0, keepdims=True) computes the squared sum of the elements in out
                case 1:
                   {
                    double *wSqSum = new double[*(nbl + 2)];
                    int k;
                    int cnt;
                       range = 1.0; // std
                    if (m_outType == 3 && b == (m_nblocks - 2))
                        range = 0.01; // std for layers followed by linear outputs
                    for (t = 0; t < *(nbl + 2); t++)
                        wSqSum[t] = 0.0;
                    // Index storing the genotype block to be normalized (i.e., the starting index)
                    k = j;
                    // Counter of weights
                    cnt = 0;
                    for (t = 0; t < *(nbl + 2); t++)
                    {
                        for (i = 0; i < *(nbl + 4); i++)
                        {
                            // Extract weights from Gaussian distribution with mean 0.0 and std 1.0
                            cgenotype[j] = netRng->getGaussian(1.0, 0.0);
                            // Update square sum of weights
                            wSqSum[t] += (cgenotype[j] * cgenotype[j]);
                            j++;
                            // Update counter of weights
                            cnt++;
                        }
                    }
                    // Normalize weights
                    j = k;
                    t = 0;
                    i = 0;
                    while (j < (k + cnt))
                    {
                        cgenotype[j] *= (range / sqrt(wSqSum[t])); // Normalization factor
                        j++;
                        i++;
                        if (i % *(nbl + 4) == 0)
                            // Move to next sum
                            t++;
                    }
                    // We delete the pointer
                    delete[] wSqSum;
                   }
                break;
                // normal gaussian distribution with range netWrange
                case 2:
                    // the range is specified manually and is the same for all layers
                    for (t = 0; t < *(nbl + 2); t++)
                    {
                        for (i = 0; i < *(nbl + 4); i++)
                        {
                            if (LSTM)
                            {
                              for (int ii=0; ii < 4; ii++)
                                {
                                    cgenotype[j] = netRng->getGaussian(range, 0.0);
                                    j++;
                                }
                            }
                            else
                            {
                             cgenotype[j] = netRng->getDouble(-m_wrange, m_wrange);
                             j++;
                            }
                        }
                    }
                break;
                default:
                    // unrecognized initialization mode
                    printf("ERROR: unrecognized initialization mode: %d \n", m_wInit);
                break;
            }
        }
        nbl = (nbl + 5);
    }
    // parameters for the diagonal gaussian output
    if (m_randAct == 2)
    {
     for (i=0; i < m_noutputs; i++)
        {
         cgenotype[j] = 0.0;
         j++;
        }
	}
		
    /* print sum of absolute incoming weight
    j = 0;
    if (m_bias)
    {
        for (i = 0; i < (m_nhiddens + m_noutputs); i++)
            j++;
    }
    double sum;
    for (b = 0, nbl = m_netblock; b < m_nblocks; b++)
    {
     printf("block %d type %d\n", b, *nbl);
     if (*nbl == 0)
     {
      for(t = 0; t < *(nbl + 2); t++)
        {
          sum = 0;
          for(i = 0; i < *(nbl + 4); i++)
          {
            sum += fabs(cgenotype[j]);
            j++;
          }
          printf("block %d neuron %d sum-abs incoming weights %f\n", b, t, sum);
        }
     }
      nbl = (nbl + 5);
    }
    */
    }
}


// set the normalization phase (0 = do nothing, 1 = collect data to update normalization vectors)
void Evonet::normphase(int phase)
{
   m_normphase = phase;
}

// collect data for normalization
void Evonet::collectNormalizationData()
{
	int i;
	for (i = 0; i < m_ninputs; i++)
       //printf("%.2f ", cobservation[i]);
	
	for (i = 0; i < m_ninputs; i++)
	{
		m_sum[i] += cobservation[i];
		m_sumsq[i] += (cobservation[i] * cobservation[i]);
		//printf("%.2f ", m_sum[i]);
	}
	//printf("\n");
	// Update counter
	m_count++;
}

// compute normalization vectors
void Evonet::updateNormalizationVectors()
{
	int i;
	int ii;
	double cStd;
	
	//printf("%.2f ]", m_count);
	for (i = 0; i < m_ninputs; i++)
	{
		m_mean[i] = m_sum[i] / m_count;
		//printf("%.2f ", m_mean[i]);
		cStd = (m_sumsq[i] / m_count - (m_mean[i] * m_mean[i]));
		if (cStd < 0.01)
			cStd = 0.01;
		m_std[i] = sqrt(cStd);
		//printf("%.2f  ", m_std[i]);
	}
	//printf("\n");
	// copy nornalization vectors on the cnormalization vector that is used for saving data
	ii = 0;
	for (i = 0; i < m_ninputs; i++)
	  {
	    cnormalization[ii] = m_mean[i];
	    ii++;
	  }
	for (i = 0; i < m_ninputs; i++)
	  {
	    cnormalization[ii] = m_std[i];
	    ii++;
	  }
}

// restore a loaded normalization vector
void Evonet::setNormalizationVectors()
{

  int i;
  int ii;
	
  if (m_normalize == 1)
  {
	ii = 0;
	for (i = 0; i < m_ninputs; i++)
	  {
	    m_mean[i] = cnormalization[ii];
	    ii++;
	  }
	for (i = 0; i < m_ninputs; i++)
	  {
	    m_std[i] = cnormalization[ii];
	    ii++;
	  }
	for (i = 0; i < m_ninputs; i++)
	   {
		 //printf("%.2f %.2f  ", m_mean[i], m_std[i]);
 	   }
  }
}

// reset normalization vector
void Evonet::resetNormalizationVectors()
{

  if (m_normalize == 1)
   {
    m_count = 0.01;                  // to avoid division by 0
    int i;
    for (i = 0; i < m_ninputs; i++)
    {
        m_sum[i] = 0.0;
        m_sumsq[i] = 0.01;           // eps
        m_mean[i] = 0.0;
        m_std[i] = 1.0;
    }
   }
}
