#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import deeptime


torsions = np.load('colvar.npy')


clusters = deeptime.clustering.KMeans(10)


# In[11]:


clusters.fit(np.column_stack([torsions[:,2], torsions[:,5]]))


# In[12]:


cc = clusters.model.cluster_centers


dtrajs = clusters.model.transform(np.column_stack([torsions[:,2], torsions[:,5]]))


# In[15]:


traj_labels = np.zeros((len(dtrajs), dtrajs.max() + 1), dtype = int)
for k, label in enumerate(dtrajs):
    traj_labels[k, label] = 1


# In[16]:


np.save('./init_traj_labels.npy', traj_labels)

