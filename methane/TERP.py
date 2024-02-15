#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import matplotlib.pyplot as plt 
import torch
import SPIB
import test_SPIB_training as training
import time
import os
import constraints
import subprocess
import matplotlib
from matplotlib import colors as c
print(os.getcwd())


def reconstruct_SPIB(beta, gamma, bandwidth, nn, lag, RC_dim = 2, pseduo_dim = 10, output_dim = 10, data_shape = (2,),
                    encoder_type = 'Nonlinear', seed = '0', lr = 0.0001000, path = './'):

    neuron_num1 = nn
    neuron_num2 = nn
    device = torch.device("cpu") #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Update Label
    ## Remember to update both directories
    UpdateLabel = True
    prefix_00 = path + str(bandwidth) + '/neurons/' + str(nn)
    representative_inputs_path = prefix_00 + '/Unweighted_d=%d_t=%d_b=%.9f_gamma=%.9f_learn=%.6f_representative_inputs%d.npy' % (RC_dim, lag, beta, gamma, lr, seed)
    prefix = prefix_00 + "/Unweighted_d=%d_t=%d_b=%.9f_gamma=%.9f_learn=%.6fcpt%d/IB_final_cpt.pt" % (RC_dim, lag, beta, gamma, lr, seed)

    print(prefix)
    restore_path=prefix
    representative_inputs = torch.tensor(np.load(representative_inputs_path))
    index = 1
    encoder_type = 'Nonlinear'
    IB = SPIB.SPIB(encoder_type, RC_dim, output_dim, data_shape, device, UpdateLabel, neuron_num1, neuron_num2)
    IB.reset_representative(representative_inputs)
    checkpoint=torch.load(restore_path)
    IB.load_state_dict(checkpoint['state_dict'])
    weight0=IB.encoder[0].weight.cpu().data.numpy()
    weight1=IB.encoder[2].weight.cpu().data.numpy()
    weight2=IB.encoder_mean.weight.cpu().data.numpy()
    bias0=IB.encoder[0].bias.cpu().data.numpy()
    bias1=IB.encoder[2].bias.cpu().data.numpy()
    bias2=IB.encoder_mean.bias.cpu().data.numpy()
    return IB


params = {'legend.fontsize': 16,
          'figure.figsize': (8, 6),
         'axes.labelsize': 30,
         'axes.titlesize':25,
         'xtick.labelsize':25,'ytick.labelsize':25,
         'axes.linewidth':4,
         'xtick.major.width':3,'ytick.major.width':3,
         'xtick.minor.width':1,'ytick.minor.width':1,
         'xtick.major.size':5,'ytick.major.size':5,
         'xtick.minor.size':3,'ytick.minor.size':3,
         'xtick.direction':'in','ytick.direction':'in'
         }
plt.rcParams.update(params)


# In[12]:

# load from file
file_path = '../analysis/big/sparse/delta_variational/1e-1/neurons/64/'
b = float(np.loadtxt('b.txt'))
beta = b

nreps = 1
seed = 0
lr = 0.0001000
z1_bar_list = []
z_list = []
z1_thermo_list = []
z1_filtered_thermo_list = []
traj_labels_list = []
for nn in [64]:
    for lag in [5]:
        for beta in [1e-2]:
            for gamma in [1e-1]:
                for bandwidth in ["1e-1"]:
                    print(nn, beta, gamma, lag, bandwidth)
                    path = '../analysis/big/sparse/delta_variational/'


                    prefix='../analysis/big/sparse/delta_variational/' + bandwidth + '/neurons/' + str(nn) + '/Unweighted_d=1_t=' + str(lag) + '_b=%.9f_gamma=%.9f_learn=%.6f' % (beta, gamma, lr)
                    #print(prefix)
                    repeat = str(seed)
                    IB = reconstruct_SPIB(beta, gamma, bandwidth, nn, lag, RC_dim = 1, pseduo_dim = 10, output_dim = 10, data_shape = (6,),
                    encoder_type = 'Nonlinear', seed = seed, path = path)


frames = np.loadtxt('random_frames.txt')
print(frames.shape)


beta = 1e-2
gamma = 1e-1
bandwidth = "1e-1"
nn = 64
RC_dim = 1
pseudo_dim = 10
output_dim = 10
data_shape = (6,)

neuron_num1 = nn
neuron_num2 = nn
encoder_type = "Nonlinear"

device = torch.device("cpu") #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Update Label
## Remember to update both directories
UpdateLabel = True
prefix_00 = file_path
representative_inputs_path = prefix + '_representative_inputs0.npy' 
prefix = prefix + "cpt0/IB_final_cpt.pt"


restore_path=prefix
representative_inputs = torch.tensor(np.load(representative_inputs_path))
index = 1
encoder_type = 'Nonlinear'

IB.reset_representative(representative_inputs)
checkpoint=torch.load(restore_path)
IB.load_state_dict(checkpoint['state_dict'])
weight0=IB.encoder[0].weight.cpu().data.numpy()
weight1=IB.encoder[2].weight.cpu().data.numpy()
weight2=IB.encoder_mean.weight.cpu().data.numpy()
bias0=IB.encoder[0].bias.cpu().data.numpy()
bias1=IB.encoder[2].bias.cpu().data.numpy()
bias2=IB.encoder_mean.bias.cpu().data.numpy()


# run TERP on each of the frames
clase = "8"
TERP_path = './'
random_frames = np.array(np.loadtxt('frames_level_set_0.txt'), dtype = int)
for frame in random_frames:
    frame = int(frame)
    subprocess.call('mkdir -pv DATA', shell = True)
    subprocess.call('mkdir -pv %s' % (TERP_path + 'class_' + clase + '/frame_' + str(frame)), shell = True)
    subprocess.call('python ' + TERP_path + 'TERP_neighborhood_generator.py -seed 1 --progress_bar -input_numeric ../sparse_traj_data.npy -num_samples 5000 -index ' + str(frame), shell = True)
    # feed TERP predictions to the SPIB model
    TERP_predictions = TERP_path + 'DATA/make_prediction_numeric.npy'
    predictions = np.load(TERP_predictions)
    outputs, z_sample, z_mean, z_logvar = IB.forward(torch.Tensor(predictions))
    predicted_prob = torch.exp(outputs)
    np.save(TERP_path + 'DATA/TERP_predictions.npy', predicted_prob.detach().numpy())
    subprocess.call('python ' + TERP_path + 'TERP_optimizer_01.py -TERP_input DATA/TERP_numeric.npy -blackbox_prediction ' + TERP_path + 'DATA/TERP_predictions.npy -explain_class '+ clase, shell = True)
    subprocess.call('python ' + TERP_path + 'TERP_neighborhood_generator.py -seed 1 --progress_bar -input_numeric ../sparse_traj_data.npy -num_samples 5000 -index ' + str(frame) + ' -selected_features TERP_results/selected_features.npy', shell = True)
    # feed TERP predictions to the SPIB model
    TERP_predictions = TERP_path + 'DATA_2/make_prediction_numeric.npy'
    predictions = np.load(TERP_predictions)
    outputs, z_sample, z_mean, z_logvar = IB.forward(torch.Tensor(predictions))
    predicted_prob = torch.exp(outputs)
    np.save(TERP_path + 'DATA_2/TERP_predictions.npy', predicted_prob.detach().numpy())
    subprocess.call('python TERP_optimizer_02.py -TERP_input DATA_2/TERP_numeric.npy -blackbox_prediction DATA_2/TERP_predictions.npy -selected_features TERP_results/selected_features.npy', shell = True)
    #subprocess.call('python ' + TERP_path + 'TERP_model.py -ncores 4 -TERP_numeric DATA/TERP_numeric.npy -pred_proba ' + TERP_path + 'frame_' + str(frame) + '/TERP_predictions.npy -iterations 20000 -k_max 4 --saveall -explain_class '+ clase, shell = True)
    subprocess.call('mv -v DATA ' + TERP_path + 'class_' + clase + '/frame_' + str(frame) + '/', shell = True)
    subprocess.call('mv -v DATA_2 ' + TERP_path + 'class_' + clase + '/frame_' + str(frame) + '/', shell = True)
    subprocess.call('mv -v TERP_results ' + TERP_path + 'class_' + clase + '/frame_' + str(frame) + '/', shell = True)
    subprocess.call('mv -v TERP_results_2 ' + TERP_path + 'class_' + clase + '/frame_' + str(frame) + '/', shell = True)
