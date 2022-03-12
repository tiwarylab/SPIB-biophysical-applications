##__________________________________________________________________________________________________##
relevency analysis script
##__________________________________________________________________________________________________##

import numpy as np
import torch
import SPIB
import SPIB_training
import torch.nn.functional as F
import os

class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

torch.set_num_threads(2)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

RC_dim=2
output_dim=24
data_shape=(21,)
neuron_num1=512
neuron_num2=512
batch_size = 1024
UpdateLabel = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
default_device = torch.device("cpu")
encoder_type = 'Nonlinear'

IB = SPIB.SPIB(encoder_type, RC_dim, output_dim, data_shape, device, UpdateLabel, neuron_num1, neuron_num2)
IB.reset_representative(representative_inputs)
IB.to(device)
## Bias MD Reweighed
BiasReweighed = True

## Load trajectory data and weights
traj_data = np.load("traj_data.npy")
weights_data = np.load("traj_weights.npy)
## Load time delayed SPIB predicted labels
future_labels = np.load("time_delayed_labels.npy")
labels = torch.from_numpy(lab).float().to(default_device)   
                  
past_data = torch.from_numpy(traj_data).float().to(default_device)
past_weights = torch.from_numpy(weights).float().to(default_device)

# pass through VAE
with torch.no_grad():
    ## Comment out the 3 lines below to analyze predictions with more than two converged states. Here, we analyze transition state between SPIB states 7, and 11.
    labels[:,11]=labels[:,11:].sum(dim=-1)
    labels[:,7]=labels[:,:11].sum(dim=-1)
    labels = labels[:,[7,11]]
                  
    ## Calculate the relevance based on the deterministic reconstruction error
    
    # obtain the original reconstruction error
    reconstruction_error0=0
    for i in range(0, len(past_data), batch_size):
        
        batch_inputs = past_data[i:i+batch_size].to(device)
        data_targets = labels[i:i+batch_size].to(device)
        weights = past_weights[i:i+batch_size].to(device)
        # pass through VAE
        # log_prediction, z_sample, z_mean, z_logvar = self.forward(batch_inputs)
        z_mean, z_logvar = IB.encode(batch_inputs)    
        outputs = IB.decoder(z_mean)
        outputs[:,11]=torch.log(outputs[:,11:].exp().sum(dim=-1))
        outputs[:,7]=torch.log(outputs[:,:11].exp().sum(dim=-1))
        outputs = outputs[:,[7,11]]
        if BiasReweighed == False:
          reconstruction_error0 += torch.mean(torch.sum(-data_targets*outputs, dim=1))*len(batch_inputs)
        elif BiasReweighed ==True:
          reconstruction_error0 += torch.sum(weights*torch.sum(-data_targets*outputs, dim=1))
        
    if BiasReweighed == False:
        reconstruction_error0/=len(past_data) 
        
    elif BiasReweighed == True:
        reconstruction_error0/=past_weights.sum()
    
    reconstruction_error0 = reconstruction_error0.data.cpu().numpy()
    ## Saved data arrays contain sines and cosines of 6 different angles. Since, sines and cosines of the same angle is correlated we will directly consider the relevance of these angles.
    relevance=np.zeros(data_shape[0]-6)
    for k in range(data_shape[0]-6):
        reconstruction_error = 0
        
        ## Uniformly sample k-th OP in [0,1]
        if k>8:
          k_angle = np.zeros((traj_data.shape[0]))
          k_angle = np.arctan2(traj_data[:,9+2*(k-9)],traj_data[:,9+2*(k-9)+1])
          k_angle_torch = torch.from_numpy(k_angle).float().to(default_device)
          new_OP =  k_angle_torch.min() + (k_angle_torch.max()-k_angle_torch.min())*torch.rand([past_data.shape[0]])
        else:
           new_OP = past_data[:,k].min() + (past_data[:,k].max()-past_data[:,k].min())*torch.rand([past_data.shape[0]])
    
        for i in range(0, len(past_data), batch_size):           
            batch_inputs = past_data[i:i+batch_size].clone()
            
            ## Replace k-th OP
            if k>8:
              batch_inputs[:,9+2*(k-9)] = np.sin(new_OP[i:i+batch_size] )
              batch_inputs[:,9+2*(k-9)+1] = np.cos(new_OP[i:i+batch_size] )
            else:
              batch_inputs[:,k] = new_OP[i:i+batch_size]
            
            batch_inputs = batch_inputs.to(device)
            data_targets = labels[i:i+batch_size].to(device)
            weights = past_weights[i:i+batch_size].to(device)
            ## Pass through VAE
            ## Log_prediction, z_sample, z_mean, z_logvar = self.forward(batch_inputs)
            z_mean, z_logvar = IB.encode(batch_inputs)
            outputs = IB.decoder(z_mean)
            outputs[:,11]=torch.log(outputs[:,11:].exp().sum(dim=-1))
            outputs[:,7]=torch.log(outputs[:,:11].exp().sum(dim=-1))
            outputs = outputs[:,[7,11]]
            if BiasReweighed == False:
              reconstruction_error += torch.mean(torch.sum(-data_targets*outputs, dim=1))*len(batch_inputs)
            elif BiasReweighed == True:
              reconstruction_error += torch.sum(weights*torch.sum(-data_targets*outputs, dim=1))
            
        if BiasReweighed == False:
          reconstruction_error/=len(past_data)
        elif BiasReweighed == True:
          reconstruction_error/=past_weights.sum()
        
        relevance[k] = reconstruction_error.data.cpu().numpy() - reconstruction_error0
        
    np.save('short_OP_relevance.npy',relevance)
