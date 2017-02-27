# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 20:34:20 2016


%reset -f
%pylab
%clear
%load_ext autoreload
%autoreload 2

@author: isaac
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import plots
reload(plots)
from plots import bmap, rcj, tl, tickout, four_plot


# %% Get some colors


import brewer2mpl
new_bmap = brewer2mpl.get_map('Set1', 'Qualitative', 9).mpl_colors
new_bmap.pop(5)

more_colors = brewer2mpl.get_map('Set2', 'Qualitative', 8).mpl_colors

new_bmap += more_colors


# %%

d = pd.read_csv('./Data/epsilon.csv')
mass = d[u'Mass (g)'].values
eps = d[u'epsilon'].values
eps_snake = d[u'eps from c from sqrt(Ws)'].values
Ws = d[u'Wing loading (N/m^2)']

all_labels = d[u'Label']
labels = d[u'Label'].unique().tolist()
markers = ['o', 'v', '^',  'p', 's', 'd']


fig, ax = plt.subplots()

eps_const = .04597
eps_const = (1.18 * 9.81) / (2 * 10**2.4)
eps_const = (1.18 * 9.81) / (2 * np.e**2.1)
eps_const = (1.18 * 9.81) / (2 * 2.1)
ax.axhline(eps_const, color='gray', lw=1)

m_min = 120
mth = np.linspace(m_min, 1400, 10)
eth = mth**(.11)
#eth -= eth[0]
#eth += .01
offset = eth[0] - .01
eth = 10**(np.log10(eth) - offset)
mgeom = np.r_[m_min, 1400]
egeom = np.r_[.01, .01]  # np.r_[eps.mean(), eps.mean()]
#ax.loglog(mth, 10**(np.log10(eth) - 2.25), c='gray')
ax.loglog(mth, eth, c='gray')
ax.loglog(mgeom, egeom, c='gray')

for i in np.arange(len(labels)):
    label = labels[i]
    marker = markers[i]
    idx = np.where(all_labels == label)[0]

    # ax.loglog(mass[idx], np.sqrt(eps[idx]), marker, label=label)
    ax.loglog(mass[idx], eps[idx], marker, c=new_bmap[i], ms=10,
              mew=0, mec='w', label=label)

#    if label == 'Snake':
#        ax.loglog(mass[idx], np.sqrt(eps_snake[idx]), marker)

ax.legend(loc='upper right', frameon=True, framealpha=.2, ncol=2)
ax.set_xlabel('mass (g)', fontsize=16)
ax.set_ylabel(r'$\epsilon$    ', fontsize=16, rotation=0)
#ax.set_aspect('equal', adjustable='box')
[ttl.set_size(16) for ttl in ax.get_xticklabels()]
[ttl.set_size(16) for ttl in ax.get_yticklabels()]

# https://stackoverflow.com/questions/21920233/matplotlib-log-scale-tick-label-number-formatting
#from matplotlib.ticker import ScalarFormatter
#for axis in [ax.xaxis, ax.yaxis]:
#    axis.set_major_formatter(ScalarFormatter())
from matplotlib import ticker
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))

rcj(ax)
tl(fig)

fig.savefig('Figures/figure8_epsilon.pdf', transparent=True)
