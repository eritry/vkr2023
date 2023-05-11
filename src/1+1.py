import random
import utils
import metrics
import time
import numpy as np
from tqdm import tqdm

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


def fitness(psnr, bitrate, bt):
    penalty = max(0, (bitrate / bt - 1) * 30)
    return psnr / 20 - penalty


def ff(qps, *args):
    bt, table, frames, res = args
    psnr, bitrate = utils.calculate_metrics(table, qps, frames)
    return psnr, bitrate


def f(qps, *args):
    psnr, bitrate = ff(qps, *args)
    bt, table, frames, res= args
    return fitness(psnr, bitrate, bt)


def get_inversed_bits(x, p):
    inversed_x = []
    for frame_bits in x:
        inversed_bits = []
        for i in range(len(frame_bits)):
            c = random.random()
            inversed_bits.append(1 - frame_bits[i] if c < p else frame_bits[i])
        inversed_x.append(inversed_bits)
    return inversed_x


def get_qp_value(x):
    c = 0
    p = 1
    for i in range(len(x)):
        c += p * x[i]
        p *= 2
    return min(c, 51)


def get_qp_list(x):
    qps = []
    for bits in x:
        qps.append(get_qp_value(bits))
    return qps    


def step(x, params):
    qps = get_qp_list(x)

    inv_x = get_inversed_bits(x, 0.005)
    inv_qps = get_qp_list(inv_x)
    
    fitn = f(qps, *params)
    inv_fitn = f(inv_qps, *params)
    
    if inv_fitn > fitn: return inv_x
    return x


frames = utils.read_frames()

with open('../stats/1+1_intra.txt', 'w') as out:
    for inp in target_bitrates.keys():
        fn = utils.get_filename(inp)

        res = (352, 288)
        fr_n = frames[fn]

        tab = '../tables/' + fn + '.pickle'
        i_table = utils.read_table_pickle(tab)

        for b, x in target_bitrates[fn]:

            t = time.time()
            params = (b, i_table, fr_n, res)
            x = [[random.randint(0, 1) for _ in range(6)] for _ in range(fr_n)]

            for epoch in range(1000000):
                # tepoch.set_description(f"Epoch {epoch}")
                x = step(x, params)
                    # tepoch.set_postfix(fitness=ff(get_qp_list(x), *params))
            print(inp, b, ff(get_qp_list(x), *params), (time.time() - t) / 60)
            print(inp, b, ff(get_qp_list(x), *params), (time.time() - t) / 60, file=out)
            print(get_qp_list, file=out)

