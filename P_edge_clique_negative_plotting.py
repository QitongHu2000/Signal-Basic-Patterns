# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.integrate import odeint
import warnings
import pickle
warnings.filterwarnings('ignore')

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
a=1.2
b=1.0
eta = 0.3

times = load_dict('P_edge_clique_P_'+str(int(100*a))+'_'+str(int(100*b))).tolist()[0]
weights = load_dict('P_edge_clique_P_weight_'+str(int(100*a))+'_'+str(int(100*b)))

colors = [ '#6A468F', '#92A8D7', '#004C65']
fig=plt.figure(figsize=(11,6))
ax=fig.add_subplot(111)

ax.scatter(weights,times,s=250,marker = 's',c=colors[0],label = 'Simulation',alpha =0.8)
ax.loglog(weights,times,c=colors[0],linewidth=4,linestyle='-', alpha = 0.7)

times= np.power(weights, 1.0/a-1.0-b/a)  * (np.power(weights[0], 1.0/a-1.0)/np.power(weights[0], 1.0/a-1.0-b/a)) #np.array((final_times[:,2]).T.tolist()[0])/1000
ax.scatter(weights,times,s=200,c=colors[1],label = r'$d_i^{\theta_J-\theta_Q}$')
ax.loglog(weights,times,c=colors[1],linewidth=4,linestyle='-', alpha = 1.0)

times= np.power(weights, 1.0/a-1.0)#np.array((final_times[:,2]).T.tolist()[0])/1000
ax.scatter(weights,times,s=200,c= 'gray',label = r'$d_i^{\theta_J}$')
ax.loglog(weights,times,c='gray',linewidth=4,linestyle='-', alpha = 0.8)

ax.tick_params(axis='both',which='both',direction='out',width=1,length=10, labelsize=25)
bwith=1
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

plt.xlabel(r"$d_i$",fontsize=35)
plt.ylabel(r"$\tau_{i}$",fontsize=35)
plt.legend(fontsize=15,loc=1,bbox_to_anchor=(1.6,1))
plt.tight_layout()
plt.savefig('P_edge_clique_P_'+str(int(100*a))+'_'+str(int(100*b))+'.pdf',dpi=300)
plt.show()