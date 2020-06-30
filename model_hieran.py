import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.autograd import Variable
import numpy as np

#proper encoding

class VRNN(nn.Module):
    def __init__(self, x_dim, h_dim, h2_dim, z_dim, z2_dim, n_layers, bias=False):
        super(VRNN, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
  

		#feature-extracting transformations for trajectory+ context
        self.phi_x = nn.Sequential(
			nn.Linear(x_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
        self.phi_z = nn.Sequential(
			nn.Linear(z_dim, h_dim),
			nn.ReLU())
        
        #second layer encoder
        self.phi_x1 = nn.Sequential(
            nn.Linear(h_dim, h2_dim),
            nn.ReLU(),
            nn.Linear(h2_dim, h2_dim),
            nn.ReLU())
        self.phi_z1 = nn.Sequential(
            nn.Linear(z2_dim, h2_dim),
            nn.ReLU())
    
        #trajectory encoder
        self.enc = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())
            
        
        #trajectory encoder 2
        self.enc1 = nn.Sequential(
            nn.Linear(h_dim + h_dim, h2_dim),
            nn.ReLU(),
            nn.Linear(h2_dim, h2_dim),
            nn.ReLU())
        
        self.enc_mean1 = nn.Linear(h2_dim, z2_dim)
        self.enc_std1 = nn.Sequential(
            nn.Linear(h2_dim, z2_dim),
            nn.Softplus())
       
  

		#prior 1
        self.prior = nn.Sequential(
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
			nn.Linear(h_dim, z_dim),
			nn.Softplus())
        
        #prior 2
        self.prior2 = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.prior_mean2 = nn.Linear(h_dim, z2_dim)
        self.prior_std2 = nn.Sequential(
            nn.Linear(h_dim, z2_dim),
            nn.Softplus())
        
   
   

		#decoder
        self.dec = nn.Sequential(
			nn.Linear(h_dim + h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
        self.dec_std = nn.Sequential(
			nn.Linear(h_dim, x_dim),
			nn.Softplus())
		#self.dec_mean = nn.Linear(h_dim, x_dim)
        self.dec_mean = nn.Sequential(
			nn.Linear(h_dim, x_dim),
			nn.Sigmoid())
   
        #decoder 2
        self.dec1 = nn.Sequential(
            nn.Linear(h2_dim + h2_dim, h2_dim),
            nn.ReLU(),
            nn.Linear(h2_dim, h2_dim),
            nn.ReLU())
        self.dec_std1 = nn.Sequential(
            nn.Linear(h2_dim, h_dim),
            nn.Softplus())
        #self.dec_mean = nn.Linear(h_dim, x_dim)
        self.dec_mean1 = nn.Sequential(
            nn.Linear(h2_dim, h_dim),
            nn.Sigmoid())
   
   
   
   
   

		#recurrence
        self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)
        #######################################################

    def forward(self, x):
        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        kld_loss = 0
        nll_loss = 0
        mse_loss = 0
        step = 0

        h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim ))
        
        for t in range(x.size(0)):
            phi_x_t = self.phi_x(x[t])
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)
            
			#prior 1
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)



			#sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

			
            #prior 2
            #prior_t2 = self.prior2(h[-1])
            prior_t2 = self.prior2(phi_z_t)
            prior_mean_t2 = self.prior_mean2(prior_t2)
            prior_std_t2 = self.prior_std2(prior_t2)
            #####################################################
            #second layer
            
            #phi_x_t1 = self.phi_x1(dec_t)
            #enc_t1 = self.enc1(torch.cat([enc_t, h[-1]], 1))
            enc_t1 = self.enc1(torch.cat([phi_z_t, h[-1]], 1))
            enc_mean_t1 = self.enc_mean1(enc_t1)
            enc_std_t1  = self.enc_std1(enc_t1)
            
            z_t1 = self._reparameterized_sample(enc_mean_t1, enc_std_t1)
            phi_z_t1 = self.phi_z1(z_t1)
            
            dec_t1 = self.dec1(torch.cat([phi_z_t1, h[-1]], 1))
            dec_mean_t1 =  self.dec_mean1(dec_t1)
            dec_std_t1  =  self.dec_std1(dec_t1)
            
            
            
            #decoder
            dec_t = self.dec(torch.cat([dec_t1, h[-1]], 1))
            dec_mean_t =  self.dec_mean(dec_t)
            dec_std_t  =  self.dec_std(dec_t)
            
            
            
            
            
            
            
            
            

			#recurrence
            #out, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
            out, h  = self.rnn(torch.cat([phi_x_t, phi_z_t1], 1).unsqueeze(0), h)
            #va = out[-1][out[-1].shape[0]-1];va =  va.detach()
            out = dec_mean_t;va = out[out.shape[0]-1];va =  va.detach()
            #print(va)
            if t == 0:
                pred = va.reshape(1,x.shape[2])
            else:
                pred = np.vstack((pred, va.reshape(1,x.shape[2]) ))

			#computing losses
            
            kld_loss += self._kld_gauss(enc_mean_t1, enc_std_t1, prior_mean_t2, prior_std_t2)
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
		
            nll_loss += self._nll_bernoulli(dec_mean_t, x[t] )
            step = step + 1
            
     
        return (kld_loss/x.shape[0] ,nll_loss/x.shape[0], pred)

    
    def sample(self, seq_len):
        sample = torch.zeros(seq_len, self.x_dim)

        h = Variable(torch.zeros(self.n_layers, 1, self.h_dim))
        for t in range(seq_len):

			#prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

			#sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)
			
			#decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
	

            phi_x_t = self.phi_x(dec_mean_t)

			#recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data
	
        return sample


    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)


    def _init_weights(self, stdv):
        pass


    def _reparameterized_sample(self, mean, std):
		
        eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mean)
        
    def _kl_anneal_function(self, step):
          return float(1/(1+np.exp(-0.0025*(step - 2500))))


    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
	

        kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) +
			(std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
			std_2.pow(2) - 1)
        return	0.5 * torch.sum(kld_element)


    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x*torch.log(theta) + (1-x)*torch.log(1-theta))
        

    def _nll_gauss(self, mean, std, x):
        pass
