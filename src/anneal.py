#!/usr/bin/env python
# coding: utf-8

# In[78]:


from scipy import optimize
import utils
import numpy as np
import time


# In[79]:


def fitness(psnr, bitrate, bt):
    penalty = max(0, (bitrate / bt - 1) * 10)
    return -(psnr / 20 - penalty)


# In[80]:


target_bitrates = {
     'hall': [
        (300, 36.57),
         (350, 0.0),
         (400, 0.0),
         (450, 0.0)
    ],
    'foreman': [
        (500, 0.0),
        (600, 0.0),
        (700, 0.0),
        (800, 0.0)
    ],
    'football_cif': [
        (1500, 0.0),
        (1600, 0.0),
        (1700, 0.0),
        (1800, 0.0),
    ],
    'container': [
        (600, 0.0),
        (700, 0.0),
        (800, 0.0),
        (900, 0.0),
    ],
}


def ff(qps, *args):
    bt, table, frames, res = args
    psnr, bitrate = utils.calculate_metrics(table, qps, frames)
    return psnr, bitrate


# In[84]:


def f(qps, *args):
    psnr, bitrate = ff(qps, *args)
    bt, table, frames, res = args
    return fitness(psnr, bitrate, bt)


# In[85]:


frames = utils.read_frames()

with open('../stats/anneal_intra.txt', 'w') as out:
    for inp in target_bitrates.keys():
        fn = utils.get_filename(inp)

        res = (352, 288)
        fr_n = frames[fn]

        tab = '../tables/' + fn + '.pickle'
        i_table = utils.read_table_pickle(tab)

        for b, x in target_bitrates[fn]:
            t = time.time()
            x0 = [25] * fr_n
            bounds = [(0, 51)] * fr_n
            params = (b, i_table, fr_n, res)
            res = optimize.dual_annealing(f, bounds, args=params, maxiter=12000)
            print(inp, b, ff(res['x'], *params), (time.time() - t) / 60, file=out)
            print(inp, b, ff(res['x'], *params), (time.time() - t) / 60)
            print(res['x'], file=out)



