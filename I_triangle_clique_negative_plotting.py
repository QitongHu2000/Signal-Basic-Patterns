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
B=0.1
C=2
eta = 0.3

times = load_dict('I_triangle_clique_I_'+str(int(100*B))+'_'+str(int(100*C))).tolist()[0]
weights = load_dict('I_triangle_clique_I_weight_'+str(int(100*B))+'_'+str(int(100*C)))

colors = ['#EECB8E', '#DC8910','#83272E']
fig=plt.figure(figsize=(6,6))
ax=fig.add_subplot(111)

ax.scatter(weights,times,s=250,marker = 's',c=colors[0],label = 'Simulation',alpha =0.8)
ax.loglog(weights,times,c=colors[0],linewidth=4,linestyle='-', alpha = 0.7)

times= np.power(weights * 1.0 , -3/2)  * (np.power(weights[0] * 1.0, -1)/np.power(weights[0] * 1.0, -3/2))#np.array((final_times[:,2]).T.tolist()[0])/1000
ax.scatter(weights,times,s=200,c=colors[1],label = r'$d_i^{\theta}$')
ax.loglog(weights,times,c=colors[1],linewidth=4,linestyle='-', alpha = 1.0)

times= np.power(weights * 1.0, -1)#np.array((final_times[:,2]).T.tolist()[0])/1000
ax.scatter(weights,times,s=200,c= 'gray')
ax.loglog(weights,times,c='gray',linewidth=4,linestyle='-', alpha = 0.8)

times= np.power(weights * 1.0, 0) * (np.power(weights[0] * 1.0, -1)/np.power(weights[0] * 1.0, 0))#np.array((final_times[:,2]).T.tolist()[0])/1000
ax.scatter(weights,times,s=200,c= colors[2],label = r'$d_i^{\theta+1}$')
ax.loglog(weights,times,c=colors[2],linewidth=4,linestyle='-', alpha = 0.8)

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
plt.tight_layout()
plt.savefig('I_triangle_clique_I_'+str(int(100*B))+'_'+str(int(100*C))+'.pdf',dpi=300)
plt.show()