#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import mdtraj
import deeptime
import jacobian
import constraints
import util
import torch
import os
from tqdm import tqdm
print(os.getcwd())


# In[12]:


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


beta = 0.4011057683822763


# In[ ]:


# load from file
'''b = beta
timestep = 0.2
dtrajs_list = []
for i in range(2, 6):
    print(i)
    driver_colvar = np.load('../hocky/prod/biased/%d/colvar.npy' % (i))
    weights = np.load('../hocky/prod/biased/%d/weights.npy' % (i))
    
    
    rho = driver_colvar[:,3]
    z = driver_colvar[:,2]
    Nw = driver_colvar[:,-1]
    
    if i == 2:
        clusters = deeptime.clustering.Kmeans(1000, max_iter = 1000)
        clusters.fit(np.column_stack([z, Nw]))
        cc = clusters.model.cluster_centers
        big_colvar = driver_colvar
        big_weights = weights
        big_dtrajs = clusters.model.transform(np.column_stack([z, Nw]))
    dtrajs = clusters.model.transform(np.column_stack([z, Nw]))
    dtrajs_list.append(dtrajs)
    
    
    big_colvar = np.vstack((big_colvar, driver_colvar))
    big_weights = np.concatenate((big_weights, weights))
    big_dtrajs = np.concatenate((big_dtrajs, dtrajs))'''


# In[6]:


np.save('./z_Nw_clustercenters.npy', cc)
np.save('./big_dtrajs.npy', big_dtrajs)
for i in range(2, 6):
    np.save('../hocky/prod/biased/%d/dtrajs.npy' % (i), dtrajs_list[i - 2])


# In[4]:


# load from file
b = beta
timestep = 0.2
dtrajs_list = []
for i in range(2, 6):
    print(i)
    driver_colvar = np.load('../hocky/prod/biased/%d/colvar.npy' % (i))[::5]
    weights = np.load('../hocky/prod/biased/%d/weights.npy' % (i))[::5]
    dtrajs = np.load('../hocky/prod/biased/%d/dtrajs.npy' % (i))[::5]
    
    rho = driver_colvar[:,3]
    z = driver_colvar[:,2]
    Nw = driver_colvar[:,-1]
    
    if i == 2:
        big_colvar = driver_colvar
        big_weights = weights
        big_dtrajs = dtrajs
    else:
        big_colvar = np.vstack((big_colvar, driver_colvar))
        big_weights = np.concatenate((big_weights, weights))
        big_dtrajs = np.concatenate((big_dtrajs, dtrajs))
    dtrajs_list.append(dtrajs)


# In[5]:


cc = np.load('./z_Nw_clustercenters.npy')


# In[23]:


histo, xbins, ybins = np.histogram2d(big_colvar[:,2], big_colvar[:,-1], bins = 50, weights = big_weights)

xbins = (xbins[1:] + xbins[:-1]) / 2
ybins = (ybins[1:] + ybins[:-1]) / 2

histo = histo.T / histo.sum()

xx, yy = np.meshgrid(xbins, ybins)

fes = -(1 / b) * np.log(histo)
fes = fes - fes.min()

im = plt.contourf(xx, yy, fes, levels = 25)
cbar = plt.colorbar(im)
plt.contour(xx, yy, fes, levels = 10, colors = 'k')
plt.scatter(cc[:,0], cc[:,1], edgecolor = 'w', c = 'w')

plt.xlabel('z, nm')
plt.ylabel(r'N$_W$, nm')
cbar.set_label('free energy, kJ / mol')
plt.xlim((0.8, 1.7))
plt.ylim((0, 37))
plt.show()
plt.close()


# In[52]:


histo, bins = np.histogram(big_colvar[:,2], bins = 50, weights = big_weights)
bins = (bins[1:] + bins[:-1]) / 2.
histo = histo / histo.sum()
fes = -(1 / b) * np.log(histo)
fes = fes - fes.min()

plt.plot(10 * bins, fes / 4.184, lw = 2, c = 'k')
plt.xlabel(r'z, $\AA$')
plt.ylabel('free energy, kcal / mol')
plt.show()
plt.close()


# In[133]:


timescale_list = []
for lag in [1, 2, 3, 4, 5, 10]:
    estimator = deeptime.decomposition.TICA(var_cutoff = 0.9, lagtime = lag).fit(big_colvar)
    model = estimator.fetch_model()
    tICs = model.transform(big_colvar)
    timescale_list.append(model.timescales())
    
    histo, xbins, ybins = np.histogram2d(tICs[:,0], tICs[:,1], bins = 50, weights = big_weights)

    xbins = (xbins[1:] + xbins[:-1]) / 2
    ybins = (ybins[1:] + ybins[:-1]) / 2

    histo = histo.T / histo.sum()

    xx, yy = np.meshgrid(xbins, ybins)

    fes = -(1 / b) * np.log(histo)
    fes = fes - fes.min()

    im = plt.contourf(xx, yy, fes, levels = 25)
    cbar = plt.colorbar(im)
    plt.contour(xx, yy, fes, levels = 10, colors = 'k')
    plt.xlabel('tIC 1')
    plt.ylabel('tIC 2')
    cbar.set_label('free energy, kJ / mol')
    #plt.xlim((0.8, 1.7))
    #plt.ylim((0, 37))
    plt.show()
    plt.close()


# In[129]:


timescales = np.array(timescale_list)


# In[132]:


plt.plot([1, 5, 10, 25, 50, 100], timescales[:,0])
#plt.xscale('log')
plt.yscale('log')


# In[6]:


model = deeptime.markov.msm.MaximumLikelihoodMSM().fit(dtrajs_list, lagtime = 2).fetch_model()


# In[9]:


histo, xbins, ybins = np.histogram2d(big_colvar[:,2], big_colvar[:,-1], bins = 50, weights = big_weights)

xbins = (xbins[1:] + xbins[:-1]) / 2
ybins = (ybins[1:] + ybins[:-1]) / 2

histo = histo.T / histo.sum()

xx, yy = np.meshgrid(xbins, ybins)

fes = -(1 / b) * np.log(histo)
fes = fes - fes.min()

plt.contourf(xx, yy, fes, levels = 25)
plt.contour(xx, yy, fes, levels = 10, colors = 'k')

psi2 = model.eigenvectors_right()[:,1]
psi2_scaled = (psi2 - psi2.min()) / (psi2.max() - psi2.min())

im = plt.scatter(cc[:,0], cc[:,1], c = psi2_scaled, cmap = 'RdYlBu')
cbar = plt.colorbar(im)
plt.xlabel('z, nm')
plt.ylabel(r'$N_W$')
cbar.set_label(r'$\psi_2$')
#plt.xlim((0.8, 1.7))
#plt.ylim((0.0, 1.2))
plt.show()
plt.close()


# In[10]:


histo, xbins, ybins = np.histogram2d(big_colvar[:,2], big_colvar[:,-1], bins = 50, weights = big_weights)

xbins = (xbins[1:] + xbins[:-1]) / 2
ybins = (ybins[1:] + ybins[:-1]) / 2

histo = histo.T / histo.sum()

xx, yy = np.meshgrid(xbins, ybins)

fes = -(1 / b) * np.log(histo)
fes = fes - fes.min()

plt.contourf(xx, yy, fes, levels = 25)
plt.contour(xx, yy, fes, levels = 10, colors = 'k')

psi3 = model.eigenvectors_right()[:,2]
psi3_scaled = (psi3 - psi3.min()) / (psi3.max() - psi3.min())

im = plt.scatter(cc[:,0], cc[:,1], c = psi3_scaled, cmap = 'RdYlBu')
cbar = plt.colorbar(im)
plt.xlabel('z, nm')
plt.ylabel(r'$N_W$')
cbar.set_label(r'$\psi_3$')
#plt.xlim((0.8, 1.7))
#plt.ylim((0.0, 1.2))
plt.show()
plt.close()


# In[11]:


psi = model.eigenvectors_right()


# In[12]:


psi2_traj = psi[:,1][big_dtrajs]
psi3_traj = psi[:,2][big_dtrajs]


# In[13]:


big_z = np.column_stack([psi[:,1][big_dtrajs], psi[:,2][big_dtrajs]])


# In[42]:


hist = np.histogram2d(big_z[:,0],big_z[:,1],bins = 50, weights = big_weights)
histo = hist[0] / hist[0].sum()
xbins = (hist[1][:-1] + hist[1][1:]) / 2
ybins = (hist[2][:-1] + hist[2][1:]) / 2

fes  = -np.log(histo).T
fes = fes - fes.min()

kT = 2.4931
xx, yy = np.meshgrid(xbins, ybins)
im = plt.contourf(xx, yy, kT * fes, levels = 25, cmap = 'coolwarm')
plt.contour(xx, yy, kT * fes, levels = 25, colors = 'k')
cbar = plt.colorbar(im)
cbar.set_label('free energy, kJ / mol')
plt.xlabel(r"z$_1$")
plt.ylabel(r"z$_2$")
plt.tight_layout()
#plt.savefig(prefix + '_latent_space_fes.pdf', dpi = 300)
plt.show()
plt.close()


# In[14]:


plt.plot(big_z[:,0])
plt.plot(big_z[:,1])


# In[9]:


x = big_colvar[:,2]
y = big_colvar[:,-1]


# In[15]:


big_z_hat = np.column_stack([psi2_scaled[big_dtrajs], psi3_scaled[big_dtrajs]])


# In[45]:


hist=np.histogram2d(x,y,bins=100)
hist_RC=np.histogram2d(x,y,bins=[hist[1],hist[2]], weights = big_z_hat[:,0])

fig, ax = plt.subplots()

RC=np.divide(hist_RC[0],hist[0])

im=ax.contourf(RC.T, extent=[hist_RC[1][0],hist_RC[1][-1],hist_RC[2][0],hist_RC[2][-1]],levels=10,
                    cmap='RdYlBu')
cb1 = fig.colorbar(im,ax=ax)
cb1.set_label('RC 1')

plt.xlabel(r"$z$, nm")
plt.ylabel(r"N$_{w,pocket}$")
plt.title(r'$\gamma$ = 0.0', fontsize = 16)
plt.tight_layout()
#plt.savefig(prefix + '_RC_projection1.pdf', dpi = 300)
plt.show()
plt.close()


# In[46]:


hist=np.histogram2d(x,y,bins=100)
hist_RC=np.histogram2d(x,y,bins=[hist[1],hist[2]], weights = big_z_hat[:,1])

fig, ax = plt.subplots()

RC=np.divide(hist_RC[0],hist[0])

im=ax.contourf(RC.T, extent=[hist_RC[1][0],hist_RC[1][-1],hist_RC[2][0],hist_RC[2][-1]],levels=10,
                    cmap='RdYlBu')
cb1 = fig.colorbar(im,ax=ax)
cb1.set_label('RC 2')

plt.xlabel(r"z, nm")
plt.ylabel(r"N$_{w,pocket}$")
plt.title(r'$\gamma$ = 0.0', fontsize = 16)
plt.tight_layout()
#plt.savefig(prefix + '_RC_projection1.pdf', dpi = 300)
plt.show()
plt.close()


# In[16]:


# coarse-grain with PCCA+ and compare to SPIB metastable states

pcca = model.pcca(n_metastable_sets = 3)


# In[17]:


data = np.column_stack([x, y])


# In[18]:


import matplotlib as mpl
fig, axes = plt.subplots(1, 3, figsize=(15, 10))

for i in range(len(axes)):
    ax = axes[i]
    ax.set_title(f"Metastable set {i+1} assignment probabilities")

    ax.scatter(*data[::10].T, c=pcca.memberships[np.concatenate([*dtrajs_list])[::10], i], cmap=plt.cm.Blues)
norm = mpl.colors.Normalize(vmin=0, vmax=1)
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.Blues), ax=axes, shrink=.8);


# In[25]:


np.save('../hocky/prod/biased/pcca_memberships.npy', pcca.memberships)
np.save('./pcca_memberships.npy', pcca.memberships)


# In[19]:


coarse_dtraj = pcca.memberships[np.concatenate([*dtrajs_list])].argmax(1)


# In[20]:


pcca_traj_labels = np.zeros((len(coarse_dtraj), coarse_dtraj.max() + 1), dtype = int)
for k, label in enumerate(coarse_dtraj):
    pcca_traj_labels[k, label] = 1


# In[21]:


pcca_predictions = pcca.memberships[np.concatenate([*dtrajs_list])]


# In[22]:


avg_z_pcca = np.zeros(np.max(coarse_dtraj) + 1)
counter = np.zeros_like(avg_z_pcca)
z_dist_pcca = [[], [], []]
for k, label in enumerate(coarse_dtraj):
    avg_z_pcca[label] += big_colvar[k,2]
    counter[label] += 1
    z_dist_pcca[label].append(big_colvar[k,2])

avg_z_pcca = avg_z_pcca / counter


# In[99]:


histo, bins = np.histogram(z_dist_pcca[0], bins = 50)

histo = histo / histo.sum()
bins = (bins[1:] + bins[:-1]) / 2.

plt.plot(bins, histo, lw = 2, c = 'b')

histo, bins = np.histogram(z_dist_pcca[1], bins = 50)

histo = histo / histo.sum()
bins = (bins[1:] + bins[:-1]) / 2.

plt.plot(bins, histo, lw = 2, c = 'cyan')

histo, bins = np.histogram(z_dist_pcca[2], bins = 50)

histo = histo / histo.sum()
bins = (bins[1:] + bins[:-1]) / 2.

plt.plot(bins, histo, lw = 2, c = 'orange')
plt.xlabel('z, nm')
plt.ylabel('P(z)')
plt.show()
plt.close()


# In[55]:


fig, ax0 = plt.subplots()
ax0.plot(coarse_dtraj, c = 'k', lw = 0, marker = 'o')
ax1 = plt.twinx(ax0)
ax1.plot(big_colvar[:,2], lw = 2, c = 'r', alpha = 0.5, zorder = -10)
ax0.set_ylabel('state label')
ax1.set_ylabel('z, nm', c = 'r')
ax1.tick_params(labelcolor = 'r', color = 'r')
ax0.set_xlabel('time step')
plt.tight_layout()
plt.show()
plt.close()


# In[24]:


avg_Nw_pcca = np.zeros(np.max(coarse_dtraj) + 1)
counter = np.zeros_like(avg_Nw_pcca)
Nw_dist_pcca = [[], [], []]
for k, label in enumerate(coarse_dtraj):
    avg_Nw_pcca[label] += big_colvar[k,-1]
    counter[label] += 1
    Nw_dist_pcca[label].append(big_colvar[k,-1])

avg_Nw_pcca = avg_Nw_pcca / counter


# In[27]:


histo, bins = np.histogram(Nw_dist_pcca[0], bins = 50)

histo = histo / histo.sum()
bins = (bins[1:] + bins[:-1]) / 2.

plt.plot(bins, histo, lw = 2, c = 'b')

histo, bins = np.histogram(Nw_dist_pcca[1], bins = 50)

histo = histo / histo.sum()
bins = (bins[1:] + bins[:-1]) / 2.

plt.plot(bins, histo, lw = 2, c = 'cyan')

histo, bins = np.histogram(Nw_dist_pcca[2], bins = 50)

histo = histo / histo.sum()
bins = (bins[1:] + bins[:-1]) / 2.

plt.plot(bins, histo, lw = 2, c = 'orange')
plt.xlabel(r'N$_W$')
plt.ylabel(r'P(N$_W$)')
plt.show()
plt.close()


# In[25]:


fig, ax0 = plt.subplots()
ax0.plot(coarse_dtraj, c = 'k', lw = 0, marker = 'o')
ax1 = plt.twinx(ax0)
ax1.plot(big_colvar[:,-1], lw = 2, c = 'r', alpha = 0.5, zorder = -10)
ax0.set_ylabel('state label')
ax1.set_ylabel(r'N$_W$', c = 'r')
ax1.tick_params(labelcolor = 'r', color = 'r')
ax0.set_xlabel('time step')
plt.tight_layout()
plt.show()
plt.close()


# In[28]:


avg_Nw_pcca = np.zeros(np.max(coarse_dtraj) + 1)
counter = np.zeros_like(avg_Nw_pcca)
Nw_dist_pcca = [[], [], []]
for k, label in enumerate(coarse_dtraj):
    avg_Nw_pcca[label] += big_colvar[k,-2]
    counter[label] += 1
    Nw_dist_pcca[label].append(big_colvar[k,-2])

avg_Nw_pcca = avg_Nw_pcca / counter


# In[29]:


histo, bins = np.histogram(Nw_dist_pcca[0], bins = 50)

histo = histo / histo.sum()
bins = (bins[1:] + bins[:-1]) / 2.

plt.plot(bins, histo, lw = 2, c = 'b')

histo, bins = np.histogram(Nw_dist_pcca[1], bins = 50)

histo = histo / histo.sum()
bins = (bins[1:] + bins[:-1]) / 2.

plt.plot(bins, histo, lw = 2, c = 'cyan')

histo, bins = np.histogram(Nw_dist_pcca[2], bins = 50)

histo = histo / histo.sum()
bins = (bins[1:] + bins[:-1]) / 2.

plt.plot(bins, histo, lw = 2, c = 'orange')
plt.xlabel(r'N$_{\mathrm{W, C60}}$')
plt.ylabel(r'P(N$_{\mathrm{W, C60}}$)')
plt.show()
plt.close()


# In[37]:


fig, ax0 = plt.subplots()
ax0.plot(coarse_dtraj, c = 'k', lw = 0, marker = 'o')
ax1 = plt.twinx(ax0)
ax1.plot(big_colvar[:,-2], lw = 2, c = 'r', alpha = 0.5, zorder = -10)
ax0.set_ylabel('state label')
ax1.set_ylabel(r'N$_{\mathrm{W, C60}}$', c = 'r')
ax1.tick_params(labelcolor = 'r', color = 'r')
ax0.set_xlabel('time step')
plt.tight_layout()
plt.show()
plt.close()


# In[69]:


from matplotlib import colors as c

labels=traj_labels
hist=plt.hist2d(x,y,bins=100)


state_num=labels.shape[1]
state_labels=np.arange(state_num)

hist_state=np.zeros([state_num]+list(hist[0].shape))

for i in range(state_num):
    hist_state[i]=plt.hist2d(x,y,bins=[hist[1],hist[2]],weights=pcca_predictions[:,i])[0]

label_map50=np.argmax(hist_state,axis=0).astype(float)
label_map50[hist[0]==0]=np.nan
plt.close()
fig, ax = plt.subplots()

fmt = mpl.ticker.FuncFormatter(lambda x, pos: state_labels[x])
tickz = np.arange(0,len(state_labels))

cMap = c.ListedColormap(plt.cm.tab20.colors[0:3])
im=ax.pcolormesh(hist[1], hist[2], label_map50.T, cmap=cMap, vmin=-0.5, vmax=len(state_labels)-0.5)
cb1 = fig.colorbar(im,ax=ax,format=fmt, ticks=tickz)

# plot labels of occupied states
dummy = labels.sum(0)
xloc = -0.75
#for n, i in enumerate(dummy):
#    if i != 0:
#        ax.text(xloc,0,str(n),horizontalalignment='center',verticalalignment='center',fontsize=64)
#        xloc += 0.2

plt.xlabel(r"z, nm")
plt.ylabel(r"N$_{w,pocket}$")
plt.tight_layout()
#plt.savefig(prefix + '_metastable_decomposition.pdf', dpi = 300)
plt.show()
plt.close()


# In[16]:


# import SPIB state assignments

import matplotlib
from matplotlib import colors as c
nreps = 4
seed = 0
lr = 0.0001000
z1_bar_list = []
z2_bar_list = []
for nn in [64]:
    for lag in [75]:
        for beta in [1e-5]:
            for gamma in [1e-1]:
                for bandwidth in ["1e-1"]:
                    print(nn, beta, gamma, lag, bandwidth)
                    path = '../hocky/prod/biased/analysis/sparse/delta_variational/'


                    prefix='../hocky/prod/biased/analysis/sparse/delta_variational/' + bandwidth + '/neurons/' + str(nn) + '/Unweighted_d=2_t=' + str(lag) + '_b=%.9f_gamma=%.9f_learn=%.6f' % (beta, gamma, lr)
                    repeat = str(seed)
                    #print('loading labels...')
                    dummy = []
                    for i in range(nreps):
                        dummy.append(np.load(prefix + "_traj" + str(i) + "_labels" + repeat + ".npy"))
                    traj_labels = np.concatenate(dummy, axis = 0)
                    
                    repeat = str(seed)
                    #print('loading labels...')
                    dummy = []
                    for i in range(nreps):
                        dummy.append(np.load(prefix + "_traj" + str(i) + "_data_prediction" + repeat + ".npy"))
                    spib_predictions = np.concatenate(dummy, axis = 0)
                    
                    dummy = []
                    for i in range(nreps):
                        dummy.append(np.load(prefix + "_traj" + str(i) + "_mean_representation" + repeat + ".npy"))
                    z_mean = np.concatenate(dummy, axis = 0)


# In[20]:


tmp = np.copy(traj_labels)
tmp[:,0] = traj_labels[:,1]
tmp[:,1] = traj_labels[:,3]
tmp[:,2] = traj_labels[:,2]

traj_labels = tmp[:,0:3]


tmp = np.copy(spib_predictions)
tmp[:,0] = spib_predictions[:,1]
tmp[:,1] = spib_predictions[:,3]
tmp[:,2] = spib_predictions[:,2]

spib_predictions = tmp[:,0:3]
spib_dtrajs = spib_predictions.argmax(1)


# In[13]:


from matplotlib import colors as c
import matplotlib as mpl

labels=traj_labels
hist=plt.hist2d(x,y,bins=100)


state_num=labels.shape[1]
state_labels=np.arange(state_num)

hist_state=np.zeros([state_num]+list(hist[0].shape))

for i in range(state_num):
    hist_state[i]=plt.hist2d(x,y,bins=[hist[1],hist[2]],weights=labels[:,i])[0]

label_map50=np.argmax(hist_state,axis=0).astype(float)
label_map50[hist[0]==0]=np.nan
plt.close()
fig, ax = plt.subplots()

fmt = mpl.ticker.FuncFormatter(lambda x, pos: state_labels[x])
tickz = np.arange(0,len(state_labels))

cMap = c.ListedColormap(plt.cm.tab20.colors[0:3])
im=ax.pcolormesh(hist[1], hist[2], label_map50.T, cmap=cMap, vmin=-0.5, vmax=len(state_labels)-0.5)
cb1 = fig.colorbar(im,ax=ax,format=fmt, ticks=tickz)

# plot labels of occupied states
dummy = labels.sum(0)
xloc = -0.75
#for n, i in enumerate(dummy):
#    if i != 0:
#        ax.text(xloc,0,str(n),horizontalalignment='center',verticalalignment='center',fontsize=64)
#        xloc += 0.2

plt.xlabel(r"z, nm")
plt.ylabel(r"N$_{w,pocket}$")
plt.tight_layout()
#plt.savefig(prefix + '_metastable_decomposition.pdf', dpi = 300)
plt.show()
plt.close()


# In[21]:


from matplotlib import colors as c
import matplotlib as mpl

labels=traj_labels
hist=plt.hist2d(z_mean[:,0],z_mean[:,1],bins=100)


state_num=labels.shape[1]
state_labels=np.arange(state_num)

hist_state=np.zeros([state_num]+list(hist[0].shape))

for i in range(state_num):
    hist_state[i]=plt.hist2d(z_mean[:,0],z_mean[:,1],bins=[hist[1],hist[2]],weights=labels[:,i])[0]

label_map50=np.argmax(hist_state,axis=0).astype(float)
label_map50[hist[0]==0]=np.nan
plt.close()
fig, ax = plt.subplots()

fmt = mpl.ticker.FuncFormatter(lambda x, pos: state_labels[x])
tickz = np.arange(0,len(state_labels))

cMap = c.ListedColormap(plt.cm.tab20.colors[0:3])
im=ax.pcolormesh(hist[1], hist[2], label_map50.T, cmap=cMap, vmin=-0.5, vmax=len(state_labels)-0.5)
cb1 = fig.colorbar(im,ax=ax,format=fmt, ticks=tickz)

# plot labels of occupied states
dummy = labels.sum(0)
xloc = -0.75
#for n, i in enumerate(dummy):
#    if i != 0:
#        ax.text(xloc,0,str(n),horizontalalignment='center',verticalalignment='center',fontsize=64)
#        xloc += 0.2

plt.xlabel(r"z$_1$")
plt.ylabel(r"z$_2$")
plt.tight_layout()
#plt.savefig(prefix + '_metastable_decomposition.pdf', dpi = 300)
plt.show()
plt.close()


# In[47]:


avg_z_spib = np.zeros(spib_dtrajs.max() + 1)
z_dist_spib = [[], [], []]
counter = np.zeros_like(avg_z_spib)
for k, label in enumerate(spib_dtrajs):
    avg_z_spib[label] += big_colvar[k,2]
    counter[label] += 1
    z_dist_spib[label].append(big_colvar[k,2])
avg_z_spib = avg_z_spib / counter


# In[95]:


histo, bins = np.histogram(z_dist_spib[0], bins = 50)

histo = histo / histo.sum()
bins = (bins[1:] + bins[:-1]) / 2.

plt.plot(bins, histo, lw = 2, c = 'b')

histo, bins = np.histogram(z_dist_spib[1], bins = 50)

histo = histo / histo.sum()
bins = (bins[1:] + bins[:-1]) / 2.

plt.plot(bins, histo, lw = 2, c = 'cyan')

histo, bins = np.histogram(z_dist_spib[2], bins = 50)

histo = histo / histo.sum()
bins = (bins[1:] + bins[:-1]) / 2.

plt.plot(bins, histo, lw = 2, c = 'orange')
plt.xlabel('z, nm')
plt.ylabel('P(z)')
plt.show()
plt.close()


# In[52]:


help(plt.plot)


# In[54]:


fig, ax0 = plt.subplots()
ax0.plot(spib_dtrajs, c = 'k', lw = 0, marker = 'o')
ax1 = plt.twinx(ax0)
ax1.plot(big_colvar[:,2], lw = 2, c = 'r', alpha = 0.5, zorder = -10)
ax0.set_ylabel('state label')
ax1.set_ylabel('z, nm', c = 'r')
ax1.tick_params(labelcolor = 'r', color = 'r')
ax0.set_xlabel('time step')
ax0.set_xscale('log')
plt.tight_layout()
plt.show()
plt.close()


# In[36]:


avg_Nw_spib = np.zeros(np.max(coarse_dtraj) + 1)
counter = np.zeros_like(avg_Nw_spib)
Nw_dist_spib = [[], [], []]
for k, label in enumerate(spib_dtrajs):
    avg_Nw_spib[label] += big_colvar[k,-1]
    counter[label] += 1
    Nw_dist_spib[label].append(big_colvar[k,-1])

avg_Nw_spib = avg_Nw_spib / counter

histo, bins = np.histogram(Nw_dist_spib[0], bins = 50)

histo = histo / histo.sum()
bins = (bins[1:] + bins[:-1]) / 2.

plt.plot(bins, histo, lw = 2, c = 'b')

histo, bins = np.histogram(Nw_dist_spib[1], bins = 50)

histo = histo / histo.sum()
bins = (bins[1:] + bins[:-1]) / 2.

plt.plot(bins, histo, lw = 2, c = 'cyan')

histo, bins = np.histogram(Nw_dist_spib[2], bins = 50)

histo = histo / histo.sum()
bins = (bins[1:] + bins[:-1]) / 2.

plt.plot(bins, histo, lw = 2, c = 'orange')
plt.xlabel(r'N$_W$')
plt.ylabel(r'P(N$_W$)')
plt.show()
plt.close()

fig, ax0 = plt.subplots()
ax0.plot(coarse_dtraj, c = 'k', lw = 0, marker = 'o')
ax1 = plt.twinx(ax0)
ax1.plot(big_colvar[:,-2], lw = 2, c = 'r', alpha = 0.5, zorder = -10)
ax0.set_ylabel('state label')
ax1.set_ylabel(r'N$_W$', c = 'r')
ax1.tick_params(labelcolor = 'r', color = 'r')
ax0.set_xlabel('time step')
plt.tight_layout()
plt.show()
plt.close()


# In[38]:


avg_Nw_spib = np.zeros(np.max(coarse_dtraj) + 1)
counter = np.zeros_like(avg_Nw_spib)
Nw_dist_spib = [[], [], []]
for k, label in enumerate(spib_dtrajs):
    avg_Nw_spib[label] += big_colvar[k,-2]
    counter[label] += 1
    Nw_dist_spib[label].append(big_colvar[k,-2])

avg_Nw_spib = avg_Nw_spib / counter

histo, bins = np.histogram(Nw_dist_spib[0], bins = 50)

histo = histo / histo.sum()
bins = (bins[1:] + bins[:-1]) / 2.

plt.plot(bins, histo, lw = 2, c = 'b')

histo, bins = np.histogram(Nw_dist_spib[1], bins = 50)

histo = histo / histo.sum()
bins = (bins[1:] + bins[:-1]) / 2.

plt.plot(bins, histo, lw = 2, c = 'cyan')

histo, bins = np.histogram(Nw_dist_spib[2], bins = 50)

histo = histo / histo.sum()
bins = (bins[1:] + bins[:-1]) / 2.

plt.plot(bins, histo, lw = 2, c = 'orange')
plt.xlabel(r'N$_{\mathrm{W, C60}}$')
plt.ylabel(r'P(N$_{\mathrm{W, C60}}$)')
plt.show()
plt.close()

fig, ax0 = plt.subplots()
ax0.plot(coarse_dtraj, c = 'k', lw = 0, marker = 'o')
ax1 = plt.twinx(ax0)
ax1.plot(big_colvar[:,-2], lw = 2, c = 'r', alpha = 0.5, zorder = -10)
ax0.set_ylabel('state label')
ax1.set_ylabel(r'N$_{\mathrm{W, C60}}$', c = 'r')
ax1.tick_params(labelcolor = 'r', color = 'r')
ax0.set_xlabel('time step')
plt.tight_layout()
plt.show()
plt.close()


# In[60]:


plt.plot(traj_labels.argmax(1) - coarse_dtraj)
plt.show()
plt.close()


# In[47]:


labels_diff = traj_labels - pcca_traj_labels


# In[49]:


labels=labels_diff
hist=plt.hist2d(x,y,bins=100)


state_num=labels.shape[1]
state_labels=np.arange(state_num)

hist_state=np.zeros([state_num]+list(hist[0].shape))

for i in range(state_num):
    hist_state[i]=plt.hist2d(x,y,bins=[hist[1],hist[2]],weights=labels[:,i])[0]

label_map50=np.argmax(hist_state,axis=0).astype(float)
label_map50[hist[0]==0]=np.nan
plt.close()
fig, ax = plt.subplots()
tmp = np.histogram2d(x,y,bins = 50)
histo = tmp[0] / tmp[0].sum()
xbins = (tmp[1][:-1] + tmp[1][1:]) / 2
ybins = (tmp[2][:-1] + tmp[2][1:]) / 2

fes  = -np.log(histo).T
fes = fes - fes.min()

kT = 2.4931
xx, yy = np.meshgrid(xbins, ybins)
plt.contourf(xx, yy, kT * fes, levels = 25, cmap = 'coolwarm')
#plt.contour(xx, yy, kT * fes, levels = np.linspace(0, 35, 25), colors = 'k')
fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: state_labels[x])
tickz = np.arange(0,len(state_labels))

im=ax.pcolormesh(hist[1], hist[2], label_map50.T, cmap=plt.cm.binary, vmin=-0.5, vmax=len(state_labels)-0.5,
                alpha = 0.75)
#cb1 = fig.colorbar(im,ax=ax,format=fmt, ticks=tickz)

plt.xlabel(r"z, nm")
plt.ylabel(r"N$_{w,pocket}$")
plt.tight_layout()
#plt.savefig(prefix + '_metastable_decomposition.pdf', dpi = 300)
plt.show()
plt.close()


# In[76]:


np.save('../hocky/prod/biased/pcca_predictions.npy', pcca_predictions)
np.save('../hocky/prod/biased/spib_predictions.npy', spib_predictions)


# In[84]:


np.save('../hocky/prod/biased/msm_z.npy', np.column_stack([psi2_traj, psi3_traj]))


# In[ ]:





# In[74]:


histo, bins = np.histogram(pcca_predictions.ravel(), bins = 50)

pcca_histo = histo / histo.sum()
bins = (bins[1:] + bins[:-1]) / 2.

plt.plot(bins, pcca_histo, lw = 3, c = 'k', label = 'msm + PCCA+')

histo, bins = np.histogram(spib_predictions.ravel(), bins = 50)

spib_histo = histo / histo.sum()
bins = (bins[1:] + bins[:-1]) / 2.

plt.plot(bins, spib_histo, lw = 3, c = 'r', label = 'spib')
plt.yscale('log')
plt.xlabel(r'$\chi$')
plt.ylabel(r'P($\chi$)')
plt.legend()
plt.show()
plt.close()


# In[72]:


pcca_histo.sum()


# In[73]:


for i in range(3):
    histo, bins = np.histogram(pcca_predictions[:,i], bins = 50)

    pcca_histo = histo / histo.sum()
    bins = (bins[1:] + bins[:-1]) / 2.

    plt.plot(bins, pcca_histo, lw = 3, c = 'k', label = 'msm + PCCA+')

    histo, bins = np.histogram(spib_predictions[:,i], bins = 50)

    spib_histo = histo / histo.sum()
    bins = (bins[1:] + bins[:-1]) / 2.

    plt.plot(bins, spib_histo, lw = 3, c = 'r', label = 'spib')
    plt.yscale('log')
    plt.xlabel(r'$\chi_{%s}$' % (i + 1))
    plt.ylabel(r'P($\chi_{%s}$)' % (i + 1))
    plt.legend()
    plt.show()
    plt.close()


# In[66]:


np.sum(pcca_histo * np.log(pcca_histo / spib_histo))


# In[69]:


np.sum(spib_histo * np.log(spib_histo / pcca_histo))


# In[61]:


histo, bins = np.histogram(spib_predictions.ravel(), bins = 50)

histo = histo / histo.sum()
bins = (bins[1:] + bins[:-1]) / 2.

plt.plot(bins, histo)
plt.yscale('log')
plt.xlabel(r'$\chi$')
plt.ylabel(r'P($\chi$)')
plt.show()
plt.close()


# In[25]:


hist=np.histogram2d(big_z[:,0],big_z[:,1],bins=100)
hist_RC=np.histogram2d(big_z[:,0],big_z[:,1],bins=[hist[1],hist[2]], weights = x)

fig, ax = plt.subplots()

RC=np.divide(hist_RC[0],hist[0])

im=ax.contourf(RC.T, extent=[hist_RC[1][0],hist_RC[1][-1],hist_RC[2][0],hist_RC[2][-1]],levels=10,
                    cmap='RdYlBu')
cb1 = fig.colorbar(im,ax=ax)
cb1.set_label('z, nm')

plt.xlabel(r"$\psi_2$")
plt.ylabel(r"$\psi_3$")
plt.title(r'$\gamma$ = 0.0', fontsize = 16)
plt.tight_layout()
#plt.savefig(prefix + '_RC_projection1.pdf', dpi = 300)
plt.show()
plt.close()


# In[26]:


hist=np.histogram2d(big_z[:,0],big_z[:,1],bins=100)
hist_RC=np.histogram2d(big_z[:,0],big_z[:,1],bins=[hist[1],hist[2]], weights = y)

fig, ax = plt.subplots()

RC=np.divide(hist_RC[0],hist[0])

im=ax.contourf(RC.T, extent=[hist_RC[1][0],hist_RC[1][-1],hist_RC[2][0],hist_RC[2][-1]],levels=10,
                    cmap='RdYlBu')
cb1 = fig.colorbar(im,ax=ax)
cb1.set_label(r'$N_{W, pocket}$')

plt.xlabel(r"$\psi_2$")
plt.ylabel(r"$\psi_3$")
plt.title(r'$\gamma$ = 0.0', fontsize = 16)
plt.tight_layout()
#plt.savefig(prefix + '_RC_projection1.pdf', dpi = 300)
plt.show()
plt.close()


# In[11]:


# energy and entropy

# load from file
b = beta
timestep = 0.2
nrg_list = []
for i in range(2, 6):
    print(i)
    nrg = np.loadtxt('../hocky/prod/biased/%d/e.txt' % (i))
    
    
    
    if i == 2:
        big_energy = nrg
    nrg_list.append(nrg)
    
    big_energy = np.concatenate((big_energy, nrg))


# In[12]:


d = np.sqrt(big_colvar[:,0]**2 + big_colvar[:,1]**2 + big_colvar[:,2]**2)


# In[10]:


data = np.column_stack([d, Nw])


# In[34]:


help(np.histogram2d)


# In[13]:


def kde_1d(data, bin_centers, binwidth = 1., weights = None):
    hist = np.zeros(len(bin_centers))
    for i, bin_center in enumerate(bin_centers):
        #print(i, bin_center)
        if weights is None:
            for k, x in enumerate(data):
                hist[i] += np.exp(-(x - bin_center)**2 / (2 * binwidth*binwidth))
        else:
            for k, x in enumerate(data):
                hist[i] += weights[k] * np.exp(-(x - bin_center)**2 / (2 * binwidth*binwidth))   
    if weights is None:
        return hist / float(k + 1)
    else:
        return hist / (weights.sum())
    
def kde_2d(data, xbins, ybins, binwidth = 1., weights = None):
    
    hist = np.zeros((len(xbins), len(ybins)))
    for i, xbin in tqdm(enumerate(xbins)):
        for j, ybin in enumerate(ybins):
            if weights is None:
                for k, x in enumerate(data):
                    hist[i, j] += np.exp(-((x[0] - xbin)**2 + (x[1] - ybin)**2) / (2 * binwidth*binwidth))
            else:
                for k, x in enumerate(data):
                    hist[i, j] += weights[k] * np.exp(-((x[0] - xbin)**2 + (x[1] - ybin)**2) / (2 * binwidth*binwidth))   
    if weights is None:
        return hist / float(k + 1)
    else:
        return hist / (weights.sum())


# In[14]:


def kde(data, bin_centers, binwidth = 1., weights = None):
    hist = np.zeros(len(bin_centers))
    for i, bin_center in enumerate(bin_centers):
        #print(i, bin_center)
        if weights is None:
            for k, x in enumerate(data):
                hist[i] += np.exp(-(x - bin_center)**2/ (2 * binwidth*binwidth))
        else:
            for k, x in enumerate(data):
                hist[i] += weights[k] * np.exp(-(x - bin_center)**2/ (2 * binwidth*binwidth))   
    if weights is None:
        return hist / float(k + 1)
    else:
        return hist / (weights.sum())

def jacobian_2d(data, z, z_bins = 50, nbins = 50, beta = 1, print_figures = False, weights = None, savepath = './'): 
	'''
	Calculates the Jacobian of a nonlinear transformation of the input trajectory; assumes the latent space is two-dimensional.
	---------
	INPUT:
	 
	data (N, n): Input trajectory of size N frames by n features. Right now, n must be equal 2
	z (N, m): Trajectory of the latent space variables of size N frames by m features
	z_bins (int): Number of bins for histogramming in the latent space
					   
	nbins (int): Number of bins for histogramming in the input space
					   
	print_figures (bool): Boolean argument specifying whether the projections of the latent space coordinates
						  onto the input surface should be written to file (True) or not (False)
	savepath (str): Path for saving figures. Only used if print_figures = True
					   
	OUTPUT:
	z1_hist (nbins, nbins): Projection of z onto the two-dimensional input space
	z2_hist (nbins, nbins): Projection of z onto the two-dimensional input space
	dzdx1, dzdx2 (nbins, nbins): Gradient of z with respect to the first input
	dzdy1, dzdy2 (nbins, nbins): Gradient of z with respect to the second input
	dzdx1_traj, dzdx2_traj (N,): Trajectory of the gradient with respect to the first input
	dzdy1_traj, dzdy2_traj (N,): Trajectory of the gradient with respect to the second input
	'''

	xbins = np.linspace(data[:,0].min() - 1e-9, data[:,0].max() + 1e-9, nbins + 2)
	ybins = np.linspace(data[:,1].min() - 1e-9, data[:,1].max() + 1e-9, nbins + 2)
	hist = np.histogram2d(data[:,0], data[:,1], bins = [xbins, ybins], weights = weights)

	dx = xbins[1] - xbins[0]
	dy = ybins[1] - ybins[0]

	# histogram
	if weights is None:
		z1_hist, dummy0, dummy1 = np.histogram2d(data[:,0], data[:,1], weights = z[:,0], bins = [xbins, ybins])
		z1_hist = np.divide(z1_hist, hist[0])
		z2_hist, xbins, ybins = np.histogram2d(data[:,0], data[:,1], weights = z[:,1], bins = [xbins, ybins])
		z2_hist = np.divide(z2_hist, hist[0])
	else:
		z1_hist, dummy0, dummy1 = np.histogram2d(data[:,0], data[:,1], weights = weights *  z[:,0], bins = [xbins, ybins])
		z1_hist = np.divide(z1_hist, hist[0])
		z2_hist, xbins, ybins = np.histogram2d(data[:,0], data[:,1], weights = weights * z[:,1], bins = [xbins, ybins])
		z2_hist = np.divide(z2_hist, hist[0])
	dzdx1, dzdy1 = np.gradient(z1_hist)
	dzdx2, dzdy2 = np.gradient(z2_hist)

	xx, yy = np.meshgrid((xbins[1:] + xbins[:-1]) / 2, (ybins[1:] + ybins[:-1]) / 2)

	if print_figures:
		os.makedirs(savepath + 'figures', exist_ok = True)
		im = plt.contourf(xx, yy, z1_hist.T, cmap = 'bwr', levels = 10)
		plt.contour(xx, yy, z1_hist.T, colors = 'k', levels = 10)
		cbar = plt.colorbar(im)
		cbar.set_label(r'z$_1$', fontsize = 16)
		plt.xticks(size = 12)
		plt.yticks(size = 12)
		plt.xlabel('x', fontsize = 16)
		plt.ylabel('y', fontsize = 16)
		plt.tight_layout()
		plt.savefig(savepath + 'figures/z1_projected.pdf', dpi = 300)
		plt.show()
		plt.close()

		im = plt.contourf(xx, yy, z2_hist.T, cmap = 'bwr', levels = 10)
		plt.contour(xx, yy, z2_hist.T, colors = 'k', levels = 10)
		cbar = plt.colorbar(im)
		cbar.set_label(r'z$_2$', fontsize = 16)
		plt.xticks(size = 12)
		plt.yticks(size = 12)
		plt.xlabel('x', fontsize = 16)
		plt.ylabel('y', fontsize = 16)
		plt.tight_layout()
		plt.savefig(savepath + 'figures/z2_projected.pdf', dpi = 300)
		plt.show()
		plt.close()

		im = plt.contourf(xx, yy, dzdx1.T, cmap = 'bwr', levels = 10)
		plt.contour(xx, yy, dzdx1.T, colors = 'k', levels = 10)
		cbar = plt.colorbar(im)
		cbar.set_label(r'$\frac{\partial z_1}{\partial x}$', fontsize = 16)
		plt.xticks(size = 12)
		plt.yticks(size = 12)
		plt.xlabel('x', fontsize = 16)
		plt.ylabel('y', fontsize = 16)
		plt.tight_layout()
		plt.savefig(savepath + 'figures/dz1dx_proj.pdf', dpi = 300)
		plt.show()
		plt.close()

		im = plt.contourf(xx, yy, dzdy1.T, cmap = 'bwr', levels = 10)
		plt.contour(xx, yy, dzdy1.T, colors = 'k', levels = 10)
		cbar = plt.colorbar(im)
		cbar.set_label(r'$\frac{\partial z_1}{\partial y}$', fontsize = 16)
		plt.xticks(size = 12)
		plt.yticks(size = 12)
		plt.xlabel('x', fontsize = 16)
		plt.ylabel('y', fontsize = 16)
		plt.tight_layout()
		plt.savefig(savepath + 'figures/dz1dy.pdf', dpi = 300)
		plt.show()
		plt.close()

		im = plt.contourf(xx, yy, dzdx2.T, cmap = 'bwr', levels = 10)
		plt.contour(xx, yy, dzdx2.T, colors = 'k', levels = 10)
		cbar = plt.colorbar(im)
		cbar.set_label(r'$\frac{\partial z_2}{\partial x}$', fontsize = 16)
		plt.xticks(size = 12)
		plt.yticks(size = 12)
		plt.xlabel('x', fontsize = 16)
		plt.ylabel('y', fontsize = 16)
		plt.tight_layout()
		plt.savefig(savepath + 'figures/dz2dx.pdf', dpi = 300)
		plt.show()
		plt.close()

		im = plt.contourf(xx, yy, dzdy2.T, cmap = 'bwr', levels = 10)
		plt.contour(xx, yy, dzdy2.T, colors = 'k', levels = 10)
		cbar = plt.colorbar(im)
		cbar.set_label(r'$\frac{\partial z_2}{\partial y}$', fontsize = 16)
		plt.xticks(size = 12)
		plt.yticks(size = 12)
		plt.xlabel('x', fontsize = 16)
		plt.ylabel('y', fontsize = 16)
		plt.tight_layout()
		plt.savefig(savepath + 'figures/dz2dy.pdf', dpi = 300)
		plt.show()
		plt.close()


	xhist, xbins = np.histogram(data[:,0], bins = nbins + 1)
	yhist, ybins = np.histogram(data[:,1], bins = nbins + 1)

	xbins = (xbins[1:] + xbins[:-1]) / 2
	ybins = (ybins[1:] + ybins[:-1]) / 2

	xbin_traj = np.digitize(data[:,0], xbins) - 1
	ybin_traj = np.digitize(data[:,1], ybins) - 1
	dtrajs = np.column_stack([xbin_traj, ybin_traj])
    
	bin_list = np.column_stack([xbin_traj, ybin_traj])

	dzdx1_traj = dzdx1[dtrajs[:,0], dtrajs[:,1]]
	dzdy1_traj = dzdy1[dtrajs[:,0], dtrajs[:,1]]
	dzdx2_traj = dzdx2[dtrajs[:,0], dtrajs[:,1]]
	dzdy2_traj = dzdy2[dtrajs[:,0], dtrajs[:,1]]

	#for k, xbin, ybin in list_enumerate(bin_list):
	#	#if k % 10000 == 0: print(k)
	#	dzdx1_traj[k] = dzdx1[xbin, ybin]
	#	dzdy1_traj[k] = dzdy1[xbin, ybin]
	#	dzdx2_traj[k] = dzdx2[xbin, ybin]
	#	dzdy2_traj[k] = dzdy2[xbin, ybin]

	# give nans zero weight in the succeding analysis
	dzdx1_traj = np.nan_to_num(dzdx1_traj)
	dzdy1_traj = np.nan_to_num(dzdy1_traj)
	dzdx2_traj = np.nan_to_num(dzdx2_traj)
	dzdy2_traj = np.nan_to_num(dzdy2_traj)

			
	return z1_hist, z2_hist, dzdx1, dzdy1, dzdx2, dzdy2, dzdx1_traj, dzdy1_traj, dzdx2_traj, dzdy2_traj


# In[16]:


data = np.column_stack([big_colvar[:,2], big_colvar[:,-1]])


# In[17]:


z1_hist, z2_hist, dzdx1, dzdy1, dzdx2, dzdy2, dzdx1_traj, dzdy1_traj, dzdx2_traj, dzdy2_traj = jacobian_2d(data, data, print_figures = False, weights = big_weights)


# In[18]:


dzdy1_traj


# In[20]:


weights


# In[20]:


z1_bins, z2_bins, dG1, dG2, dU1, dU2, dS1, dS2 = jacobian.calc_energy_entropy_2d(data, data, np.ones(len(data)), np.ones(len(data)), np.ones(len(data)), np.ones(len(data)), beta * big_energy, weights = big_weights, nbins = 50, beta = 1, print_figures = False, NPT = False, KDE = False,
						  bandwidth1 = 0.1, bandwidth2 = 0.1, savepath = './')

z1_bins = (z1_bins[1:] + z1_bins[:-1]) / 2
z2_bins = (z2_bins[1:] + z2_bins[:-1]) / 2

plt.plot(z1_bins, dG1 / beta, lw = 2, c = 'k')
plt.plot(z1_bins, dU1 / beta, lw = 2, c = 'r')
plt.plot(z1_bins, -dS1 / beta, lw = 2, c = 'b')
plt.xlabel('z, nm')
plt.ylabel(r'$\Delta$G/k$_B$T, $\Delta$U/k$_B$T, -$\Delta S/k_B$')
plt.show()
plt.close()

plt.plot(z2_bins, dG2 / beta, lw = 2, c = 'k')
plt.plot(z2_bins, dU2 / beta, lw = 2, c = 'r')
plt.plot(z2_bins, -dS2 / beta, lw = 2, c = 'b')
plt.xlabel(r'N$_{W, pocket}$')
plt.ylabel(r'$\Delta$G/k$_B$T, $\Delta$U/k$_B$T, -$\Delta S/k_B$')
plt.show()
plt.close()


# In[23]:


z1_bins, z2_bins, dG1, dG2, dU1, dU2, dS1, dS2 = jacobian.calc_energy_entropy_2d(data, data, dzdx1_traj, dzdy1_traj, dzdx2_traj, dzdy2_traj, beta * big_energy, weights = big_weights, nbins = 50, beta = 1, print_figures = False, NPT = False, KDE = True,
						  bandwidth1 = 0.1, bandwidth2 = 0.1, savepath = './')

z1_bins = (z1_bins[1:] + z1_bins[:-1]) / 2
z2_bins = (z2_bins[1:] + z2_bins[:-1]) / 2

plt.plot(z1_bins, dG1 / beta, lw = 2, c = 'k')
plt.plot(z1_bins, dU1 / beta, lw = 2, c = 'r')
plt.plot(z1_bins, -dS1 / beta, lw = 2, c = 'b')
plt.show()
plt.close()

plt.plot(z2_bins, dG2 / beta, lw = 2, c = 'k')
plt.plot(z2_bins, dU2 / beta, lw = 2, c = 'r')
plt.plot(z2_bins, -dS2 / beta, lw = 2, c = 'b')
plt.show()
plt.close()


# In[39]:


z1_bins, z2_bins, dG1, dG2, dU1, dU2, dS1, dS2 = jacobian.calc_energy_entropy_2d(data, big_z, dzdx1_traj, dzdy1_traj, dzdx2_traj, dzdy2_traj, beta * big_energy, weights = None, nbins = 50, beta = 1, print_figures = False, NPT = False, KDE = False,
						  bandwidth1 = 0.1, bandwidth2 = 0.1, savepath = './')

z1_bins = (z1_bins[1:] + z1_bins[:-1]) / 2
z2_bins = (z2_bins[1:] + z2_bins[:-1]) / 2

plt.plot(z1_bins, dG1 / beta, lw = 2, c = 'k')
plt.plot(z1_bins, dU1 / beta, lw = 2, c = 'r')
plt.plot(z1_bins, -dS1 / beta, lw = 2, c = 'b')
plt.xlabel(r'$\psi_2$, nm')
plt.ylabel(r'$\Delta$G/k$_B$T, $\Delta$U/k$_B$T, -$\Delta S/k_B$')
plt.show()
plt.close()

plt.plot(z2_bins, dG2 / beta, lw = 2, c = 'k')
plt.plot(z2_bins, dU2 / beta, lw = 2, c = 'r')
plt.plot(z2_bins, -dS2 / beta, lw = 2, c = 'b')
plt.xlabel(r'$\psi_3$')
plt.ylabel(r'$\Delta$G/k$_B$T, $\Delta$U/k$_B$T, -$\Delta S/k_B$')
plt.show()
plt.close()


# In[38]:


z1_bins, z2_bins, dG1, dG2, dU1, dU2, dS1, dS2 = jacobian.calc_energy_entropy_2d(data, big_z, dzdx1_traj, dzdy1_traj, dzdx2_traj, dzdy2_traj, beta * big_energy, weights = None, nbins = 50, beta = 1, print_figures = False, NPT = False, KDE = True,
						  bandwidth1 = 0.1, bandwidth2 = 0.1, savepath = './')

z1_bins = (z1_bins[1:] + z1_bins[:-1]) / 2
z2_bins = (z2_bins[1:] + z2_bins[:-1]) / 2

plt.plot(z1_bins, dG1 / beta, lw = 2, c = 'k')
plt.plot(z1_bins, dU1 / beta, lw = 2, c = 'r')
plt.plot(z1_bins, -dS1 / beta, lw = 2, c = 'b')
plt.xlabel(r'$\psi_2$')
plt.ylabel(r'$\Delta$G/k$_B$T, $\Delta$U/k$_B$T, -$\Delta S/k_B$')
plt.show()
plt.close()

plt.plot(z2_bins, dG2 / beta, lw = 2, c = 'k')
plt.plot(z2_bins, dU2 / beta, lw = 2, c = 'r')
plt.plot(z2_bins, -dS2 / beta, lw = 2, c = 'b')
plt.xlabel(r'$\psi_3$')
plt.ylabel(r'$\Delta$G/k$_B$T, $\Delta$U/k$_B$T, -$\Delta S/k_B$')
plt.show()
plt.close()


# In[34]:


# tICA
timescale_list = []
for lag in [100]:
    estimator = deeptime.decomposition.TICA(var_cutoff = 0.9, lagtime = lag).fit(big_colvar)
    model = estimator.fetch_model()
    tICs = model.transform(big_colvar)
    timescale_list.append(model.timescales())
    
    histo, xbins, ybins = np.histogram2d(tICs[:,0], tICs[:,1], bins = 50, weights = big_weights)

    xbins = (xbins[1:] + xbins[:-1]) / 2
    ybins = (ybins[1:] + ybins[:-1]) / 2

    histo = histo.T / histo.sum()

    xx, yy = np.meshgrid(xbins, ybins)

    fes = -(1 / b) * np.log(histo)
    fes = fes - fes.min()

    im = plt.contourf(xx, yy, fes, levels = 25)
    cbar = plt.colorbar(im)
    plt.contour(xx, yy, fes, levels = 10, colors = 'k')
    plt.xlabel('tIC 1')
    plt.ylabel('tIC 2')
    cbar.set_label('free energy, kJ / mol')
    #plt.xlim((0.8, 1.7))
    #plt.ylim((0, 37))
    plt.show()
    plt.close()


# In[31]:


hist=np.histogram2d(x,y,bins=100)
hist_RC=np.histogram2d(x,y,bins=[hist[1],hist[2]], weights = tICs[:,0])

fig, ax = plt.subplots()

RC=np.divide(hist_RC[0],hist[0])

im=ax.contourf(RC.T, extent=[hist_RC[1][0],hist_RC[1][-1],hist_RC[2][0],hist_RC[2][-1]],levels=10,
                    cmap='RdYlBu')
cb1 = fig.colorbar(im,ax=ax)
cb1.set_label('tIC 1')

plt.xlabel(r"$z$, nm")
plt.ylabel(r"N$_{w,pocket}$")
plt.title(r'$\gamma$ = 0.0', fontsize = 16)
plt.tight_layout()
#plt.savefig(prefix + '_RC_projection1.pdf', dpi = 300)
plt.show()
plt.close()


# In[32]:


hist=np.histogram2d(x,y,bins=100)
hist_RC=np.histogram2d(x,y,bins=[hist[1],hist[2]], weights = tICs[:,1])

fig, ax = plt.subplots()

RC=np.divide(hist_RC[0],hist[0])

im=ax.contourf(RC.T, extent=[hist_RC[1][0],hist_RC[1][-1],hist_RC[2][0],hist_RC[2][-1]],levels=10,
                    cmap='RdYlBu')
cb1 = fig.colorbar(im,ax=ax)
cb1.set_label('tIC 2')

plt.xlabel(r"$z$, nm")
plt.ylabel(r"N$_{w,pocket}$")
plt.title(r'$\gamma$ = 0.0', fontsize = 16)
plt.tight_layout()
#plt.savefig(prefix + '_RC_projection1.pdf', dpi = 300)
plt.show()
plt.close()


# In[36]:


tICs[:,:2].shape


# In[37]:


z1_bins, z2_bins, dG1, dG2, dU1, dU2, dS1, dS2 = jacobian.calc_energy_entropy_2d(data, tICs[:,:2], dzdx1_traj, dzdy1_traj, dzdx2_traj, dzdy2_traj, beta * big_energy, weights = None, nbins = 50, beta = 1, print_figures = False, NPT = False, KDE = False,
						  bandwidth1 = 1.0, bandwidth2 = 1.0, savepath = './')

z1_bins = (z1_bins[1:] + z1_bins[:-1]) / 2
z2_bins = (z2_bins[1:] + z2_bins[:-1]) / 2

plt.plot(z1_bins, dG1 / beta, lw = 2, c = 'k')
plt.plot(z1_bins, dU1 / beta, lw = 2, c = 'r')
plt.plot(z1_bins, -dS1 / beta, lw = 2, c = 'b')
plt.show()
plt.close()

plt.plot(z2_bins, dG2 / beta, lw = 2, c = 'k')
plt.plot(z2_bins, dU2 / beta, lw = 2, c = 'r')
plt.plot(z2_bins, -dS2 / beta, lw = 2, c = 'b')
plt.show()
plt.close()


# In[40]:


z1_bins, z2_bins, dG1, dG2, dU1, dU2, dS1, dS2 = jacobian.calc_energy_entropy_2d(data, tICs[:,:2], dzdx1_traj, dzdy1_traj, dzdx2_traj, dzdy2_traj, beta * big_energy, weights = None, nbins = 50, beta = 1, print_figures = False, NPT = False, KDE = True,
						  bandwidth1 = 0.1, bandwidth2 = 0.1, savepath = './')

z1_bins = (z1_bins[1:] + z1_bins[:-1]) / 2
z2_bins = (z2_bins[1:] + z2_bins[:-1]) / 2

plt.plot(z1_bins, dG1 / beta, lw = 2, c = 'k')
plt.plot(z1_bins, dU1 / beta, lw = 2, c = 'r')
plt.plot(z1_bins, -dS1 / beta, lw = 2, c = 'b')
plt.show()
plt.close()

plt.plot(z2_bins, dG2 / beta, lw = 2, c = 'k')
plt.plot(z2_bins, dU2 / beta, lw = 2, c = 'r')
plt.plot(z2_bins, -dS2 / beta, lw = 2, c = 'b')
plt.show()
plt.close()


# In[10]:


# load from file
b = beta
timestep = 0.2
dtrajs_list = []
for i in range(2, 6):
    print(i)
    driver_colvar = np.load('../hocky/prod/biased/%d/colvar.npy' % (i))
    
    
    rho = driver_colvar[:,3]
    z = driver_colvar[:,2]
    Nw = driver_colvar[:,-1]
    d = np.sqrt(driver_colvar[:,0]**2 + driver_colvar[:,1]**2 + driver_colvar[:,2]**2)
    
    if i == 2:
        clusters = deeptime.clustering.Kmeans(100, max_iter = 1000)
        clusters.fit(np.column_stack([d, Nw]))
        cc = clusters.model.cluster_centers
        big_colvar = driver_colvar
        big_weights = weights
        big_dtrajs = clusters.model.transform(np.column_stack([d, Nw]))
    dtrajs = clusters.model.transform(np.column_stack([d, Nw]))
    dtrajs_list.append(dtrajs)
    labels = np.zeros((len(dtrajs), dtrajs.max() + 1))
    for k, label in enumerate(dtrajs):
        labels[k,label] = 1
        
    np.save('../hocky/prod/biased/%d/init_labels100.npy' % (i), labels)
    
    
    big_colvar = np.vstack((big_colvar, driver_colvar))
    big_dtrajs = np.concatenate((big_dtrajs, dtrajs))


# In[12]:


labels.shape


# In[13]:


len(big_colvar)


# In[ ]:




