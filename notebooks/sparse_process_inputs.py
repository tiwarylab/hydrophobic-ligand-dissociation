#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


params = {'legend.fontsize': 25,
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


# In[3]:


b = .40110705547310577193


# In[41]:


beta = b
timestep = 0.2
#big_colvar = []
#big_driver_colvar = []
#big_weights = []
with open('./COLVAR') as f:
	tmp = []
	for line in f:
		if line[0] == '#':
			pass
		else:
			if len(line.split()) == 13:
				tmp.append(line.split())
			else:
				pass
colvar = np.array(tmp, dtype = float)[::-1]
tmp = np.copy(colvar)
dummy = []
dummy.append(colvar[0,:])
counter = 0
for k in range(1, len(colvar)):
	if np.allclose(colvar[k,0] - dummy[counter][0], -timestep):
		dummy.append(colvar[k,:])
		counter += 1
colvar = np.array(dummy)[::-1]
driver_colvar = np.genfromtxt('./driver/COLVAR', usecols = (1, 2, 3, 4, 5, 6))
#dummy, unique_indices = np.unique(colvar[:,0][::-1], return_index = True)
#unique_indices = unique_indices[::-1]
#colvar = colvar[unique_indices]
weights = np.exp(beta * colvar[:,-2])
#if i == 2:
#	big_driver_colvar = driver_colvar
#	big_colvar = colvar[:,1:]
#	big_weights = weights[:len(driver_colvar),None]
#else:
#	big_driver_colvar = np.concatenate((big_driver_colvar, driver_colvar))
#	big_colvar = np.concatenate((big_colvar, colvar[:,1:]))
#	big_weights = np.concatenate((big_weights, weights[:len(driver_colvar),None]))
nrg = np.loadtxt('e.txt')
if len(nrg) > len(driver_colvar):
	np.save('./sparse_traj_data.npy', driver_colvar[::5])
	np.save('./sparse_weights.npy', weights[:len(driver_colvar)][::5])
	np.savetxt('sparse_e.txt', nrg[:len(driver_colvar)][::5])
	print(len(driver_colvar[::5]), len(weights[:len(driver_colvar)][::5]), len(nrg[:len(driver_colvar)][::5]))
else:
	np.save('./sparse_traj_data.npy', driver_colvar[:len(nrg)][::5])
	np.save('./sparse_weights.npy', weights[:len(nrg)][::5])
	np.savetxt('sparse_e.txt', nrg[::5])
	print(len(driver_colvar[:len(nrg)][::5]), len(weights[:len(nrg)][::5]), len(nrg[::5]))

print(len(colvar), len(driver_colvar))



# In[4]:


# load from file
beta = b
timestep = 0.2

driver_colvar = np.load('./sparse_traj_data.npy')
weights = np.load('./sparse_weights.npy')



print(len(weights), len(driver_colvar))


# In[5]:


d = np.sqrt(driver_colvar[:,2]**2 + driver_colvar[:,1]**2 + driver_colvar[:,0]**2)
rho = driver_colvar[:,3]
z = driver_colvar[:,2]
Nwp = driver_colvar[:,-2]
Nwm = driver_colvar[:,-1]
#N_combined = driver_colvar[:,-1]


import deeptime

try:
	cc = np.load('./clustercenters.npy')
	model = deeptime.clustering.ClusterModel(n_clusters = cc.shape[0], cluster_centers = cc, converged = True)
	dtrajs = model.transform(np.column_stack([z, Nwm]))
except FileNotFoundError:
	clusters = deeptime.clustering.Kmeans(10)
	clusters.fit(np.column_stack([z, Nwm]))
	cc = clusters.model.cluster_centers
	dtrajs = clusters.transform(np.column_stack([z, Nwm]))
	np.save('./clustercenters.npy', cc)

traj_labels = np.zeros((len(dtrajs), dtrajs.max() + 1), dtype = int)
for k, label in enumerate(dtrajs):
	traj_labels[k, label] = 1
	
np.save('./sparse_init_traj_labels.npy', traj_labels)
