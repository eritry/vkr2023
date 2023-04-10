#!/usr/bin/env python
# coding: utf-8
import collections
# In[1]:


import subprocess
import os


# In[2]:

def read_frames(input_file='../dataset/info/frames.info'):
    frames = {}
    for line in open(input_file, 'r'):
        v = line.split()
        frames[v[0]] = int(v[1])
    return frames


# In[3]:


def decode(input_path, output_path):
    process = subprocess.Popen(['../tools/JM/bin/ldecod_static', '-i', input_path, '-o', output_path],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout, stderr


# In[6]:


def encode(input_path, output_path, resolution, frames, preset='veryslow', bitrate=None, qpfile=None, qp=None, keyint=None):
    command = ['../tools/x264/x264',
               input_path,
               '--output', output_path,
               '--input-res', str(resolution[0]) + 'x' + str(resolution[1]),
               '--preset', preset,
               '--frames', str(frames),
               '--output-depth', '8',
               '--input-depth', '8',
               '--verbose',
               '--ref', '1',
               '--bframes', '0',
               ]

    if qp is not None: command.extend(['--qp', str(qp)])
    if qpfile is not None: command.extend(['--qpfile', qpfile])
    if bitrate is not None: command.extend(['--bitrate', str(bitrate)])
    if keyint is not None: command.extend(['--no-scenecut', '--keyint', str(keyint)])
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout, stderr


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


# In[1]:


def get_core_vector(encoded_path, decoded_path):
    out, err = decode(encoded_path, decoded_path)
    out = [s.split() for s in str(out).split('\\n')]
    b = out.index(['Frame', 'POC', 'Pic#', 'QP', 'SnrY', 'SnrU', 'SnrV', 'Y:U:V', 'Time(ms)']) + 1
    e = out.index(['--------------------', 'Average', 'SNR', 'all', 'frames', '------------------------------'])
    out = out[b + 1:e]
    core_vec = [s[3] for s in out]
    return core_vec

def read_table(path, frames):
    table = [[{} for _ in range(frames + 1)] for _ in range(52)]
    cnt = collections.defaultdict(int)
    for line in open(path, 'r'):
        v = line.split(';')
        for j in range(len(v)):
            k = v[j].split()
            if len(k) == 0: continue
            table[int(k[0])][int(k[3])]['bytes'] = int(k[1])
            table[int(k[0])][int(k[3])]['mse'] = float(k[2])
            cnt[k[0]] += 1
    return table
