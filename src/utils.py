#!/usr/bin/env python
# coding: utf-8
import collections
import pickle
import metrics
import numpy as np
# In[1]:


import subprocess
import os

# считывание записанной заранее информации о количестве кадров
def read_frames(input_file='../dataset/info/frames.info'):
    frames = {}
    for line in open(input_file, 'r'):
        v = line.split()
        frames[v[0]] = int(v[1])
    return frames

# запуск декодера
def decode(input_path, output_path):
    process = subprocess.Popen(
        ['/home/itmo/work/yandex/tools/JM/bin/ldecod_static', '-i', input_path, '-o', output_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout, stderr


# запуск кодера, параметры нужны чтобы он брал ровно те qp, которые укащаны в qpfile
def encode(input_path, output_path, resolution, frames, preset='veryslow', bitrate=None, qpfile=None, qp=None,
           keyint=None):
    command = ['/home/itmo/work/yandex/tools/x264/x264',
               input_path,
               '--output', output_path,
               '--input-res', str(resolution[0]) + 'x' + str(resolution[1]),
               '--preset', preset,
               '--fps', '30',
               '--frames', str(frames),
               '--output-depth', '8',
               '--input-depth', '8',
               '--verbose',
               '--ref', '1',
               '--bframes', '0',
               # '--psnr',
               '--aq-mode', '0',
               '--no-mbtree'
               ]

    if qp is not None: command.extend(['--qp', str(qp)])
    if qpfile is not None: command.extend(['--qpfile', qpfile])
    if bitrate is not None: command.extend(['--bitrate', str(bitrate)])
    if keyint is not None: command.extend(['--no-scenecut', '--keyint', str(keyint)])
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout, stderr

# извлечение имени файла
def get_filename(path):
    return path.split('/')[-1].split('.')[0]


def mkdir(path):
    process = subprocess.Popen(['mkdir', path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    # print(stdout)


def rmdir(path):
    process = subprocess.Popen(['rm', '-r', path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    # print(stdout)


def rm(path):
    process = subprocess.Popen(['rm', path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()


# In[9]:


def touch(path):
    process = subprocess.Popen(['touch', path],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    # print(stdout, stderr)


# In[10]:

def calculate_bitrate(file_path, frames, fps=30):
    return os.path.getsize(file_path) / (frames / fps) / 1024 * 8


def calculate_metrics(table, qps, frames):
    bts = 0
    mse_arr = []
    for j in range(len(qps)):
        qp = int(qps[j])
        bts += table[qp]['bytes'][j]
        mse_arr.append(table[qp]['mse'][j])
    bitrate = bts / (frames / 30) / 1024 * 8
    psnr_y = 10 * np.log10((255 * 255) / (sum(mse_arr) / len(mse_arr)))
    return psnr_y, bitrate


# заставляет x264 выдать точно нужный битрейт, сам он не справляется например в случае ippppppp
def get_exact_bitrate(inp, enc, dec, res, fr_n, bitr):
    bitrate_target = bitr
    bitrate_x = 0
    for p in [100]:
        while bitrate_x < bitr:
            bitrate_target += p
            encode(inp, enc, res, fr_n, bitrate=round(bitrate_target), keyint=fr_n)
            decode(enc, dec)
            psnr_x, _ = metrics.calculate_psnr(inp, dec, res, fr_n)
            bitrate_x = calculate_bitrate(enc, fr_n)
        while bitrate_x > bitr:
            bitrate_target -= p
            encode(inp, enc, res, fr_n, bitrate=round(bitrate_target), keyint=fr_n)
            decode(enc, dec)
            psnr_x, _ = metrics.calculate_psnr(inp, dec, res, fr_n)
            bitrate_x = calculate_bitrate(enc, fr_n)

    encode(inp, enc, res, fr_n, bitrate=round(bitrate_target), keyint = fr_n)
    out, err = decode(enc, dec)
    out = [s.split() for s in str(out).split('\\n')]
    b = out.index(['Frame', 'POC', 'Pic#', 'QP', 'SnrY', 'SnrU', 'SnrV', 'Y:U:V', 'Time(ms)']) + 1
    e = out.index(['--------------------', 'Average', 'SNR', 'all', 'frames', '------------------------------'])
    out = out[b + 1:e]
    qp = []
    for s in out:
        if s[1] == 'P': qp.append(int(s[5]))
        else: qp.append(int(s[3]))
    psnr_x264, _ = metrics.calculate_psnr(inp, dec, res, fr_n)
    bitrate_x264 = calculate_bitrate(enc, fr_n)
    print("core: ", qp)
    return psnr_x264, bitrate_x264, qp

# в качестве опорного вектора возвращает вектор полученный из x264
def get_core_vector(inp, enc, dec, res, fr_n, b):
    _, _, core_vec = get_exact_bitrate(inp, enc, dec, res, fr_n, b)
    return core_vec

# считывание предпосчитанных таблиц
def read_table_pickle(path):
    with open(path, 'rb') as handle:
        table = pickle.load(handle)
    return table
