import utils
import collections
import numpy as np

target_bitrates = {
    'stefan_cif.yuv': [(3549.0703, 30.3902),]
    #                    (3751.2552, 30.9156),
    #                    (5030.4583, 34.1569),
    #                    (6231.3307, 36.8507), ],
    # 'foreman.yuv': [(1232.0508, 30.8496),
    #                 (1561.782, 32.3449),
    #                 (2121.3531, 34.3978),
    #                 (2557.575, 35.7613), ],
    # 'football_cif.yuv': [(2066.744, 31.9217),
    #                      (2286.6193, 32.6293),
    #                      (2494.8056, 33.3437),
    #                      (2700.2335, 34.0063), ],
    # 'container.yuv': [(1488.8469, 31.5765),
    #                   (1979.0609, 33.6011),
    #                   (2177.9664, 34.3326),
    #                   (2407.2797, 35.0005), ],
    # 'hall.yuv': [(998.1867, 31.4239),
    #              (1189.0406, 32.8684),
    #              (1507.6609, 34.4131),
    #              (1977.2789, 36.8544), ],
}

tables = {}


def get_table_path(file_name):
    return '../tables/' + file_name + '.txt'


frames = utils.read_frames()
for f in target_bitrates.keys():
    fn = utils.get_filename(f)
    tables[fn] = utils.read_table(get_table_path(fn), frames[fn])


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
            s = table[qp][frame]['mse'] + m * table[qp][frame]['bytes']
            if s < min_s or (s == min_s and min_sz < table[qp][frame]['bytes']):
                min_s = s
                min_sz = table[qp][frame]['bytes']
                best_qp = qp
        sz += table[best_qp][frame]['bytes']
        qps.append(best_qp)
    return qps, sz


def binary_search(file_name, bits_sum, l=0, r=1, max_steps=100):
    n = 0
    while abs(r - l) > 1e-9 and n < max_steps:
        n += 1
        m = (l + r) / 2
        qps, sz = func(m, file_name)
        # print(sz)
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
        bts += table[qp][j]['bytes']
        mse_arr.append(table[qp][j]['mse'])
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

for f in target_bitrates.keys():
    fn = utils.get_filename(f)
    frames[fn] = 60
    print(fn, ":[")
    for i in range(len(target_bitrates[f])):
        psnr, bitrate = calculate_metrics(fn, best_qps[fn][i])
        # print('\t(', target_bitrates[f][i][0], ", ", round(psnr, 4), "),", sep = '')
        print('\t(', target_bitrates[f][i][0], ", ", round(psnr, 4), "),", sep='')

    print("],")