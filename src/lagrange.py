import utils
import collections
import numpy as np
import time

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


tables = {}


def get_table_path(file_name):
    return '../tables/' + file_name + '.pickle'


frames = utils.read_frames()
for f in target_bitrates.keys():
    fn = utils.get_filename(f)
    tables[fn] = utils.read_table_pickle(get_table_path(fn))


def get_bytes_sum(bitrate, fr_n):
    return bitrate / 8 * 1024 * (fr_n / 30)


def func(m, file_name):
    global tables
    table = tables[file_name]
    qps = []
    sz = 0
    for frame in range(frames[file_name]):
        best_qp = -1
        min_s = float('inf')
        min_sz = float('inf')
        for qp in range(0, 52):
            s = table[qp]['mse'][frame] + m * table[qp]['bytes'][frame]
            if s < min_s or (s == min_s and min_sz < table[qp]['bytes'][frame]):
                min_s = s
                min_sz = table[qp]['bytes'][frame]
                best_qp = qp
        sz += table[best_qp]['bytes'][frame]
        qps.append(best_qp)
    return qps, sz


def binary_search(file_name, bits_sum, l=0, r=1, max_steps=100):
    n = 0
    while abs(r - l) > 1e-9 and n < max_steps:
        n += 1
        m = (l + r) / 2
        qps, sz = func(m, file_name)
        if sz <= bits_sum:
            r = m
        else:
            l = m
    return r


def calculate_metrics(fn, qps):
    global tables, frames
    table = tables[fn]
    bts = 0
    mse_arr = []
    for j in range(len(qps)):
        qp = qps[j]
        bts += table[qp]['bytes'][j]
        mse_arr.append(table[qp]['mse'][j])
    bitrate = bts / (frames[fn] / 30) / 1024 * 8
    psnr_y = 10 * np.log10((255 * 255) / (sum(mse_arr) / len(mse_arr)))
    return psnr_y, bitrate


best_qps = collections.defaultdict(dict)
lambdas = collections.defaultdict(dict)
for f in target_bitrates.keys():
    fn = utils.get_filename(f)
    for i in range(len(target_bitrates[f])):
        bytes_size = get_bytes_sum(target_bitrates[f][i][0], frames[fn])
        lambda_opt = binary_search(fn, bytes_size)
        best_qps[fn][i], _ = func(lambda_opt, fn)
        lambdas[fn][i] = lambda_opt

with open('../stats/lagrange_intra.txt', 'w') as out:
    for f in target_bitrates.keys():
        t = time.time()
        fn = utils.get_filename(f)
        for i in range(len(target_bitrates[f])):
            psnr, bitrate = calculate_metrics(fn, best_qps[fn][i])
            print(fn, target_bitrates[f][i][0], (psnr, bitrate), time.time() - t)
            print(fn, target_bitrates[f][i][0], (psnr, bitrate), time.time() - t, file=out)
            print(best_qps[fn][i], file=out)
