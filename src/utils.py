#!/usr/bin/env python
# coding: utf-8

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
    # print(stdout, stderr)


# In[6]:


def encode(input_path, output_path, resolution, frames, preset='ultrafast', bitrate=None, qpfile=None):
    command = ['../tools/x264/x264', 
               input_path, 
               '--output', output_path, 
               '--input-res', str(resolution[0])+'x'+str(resolution[1]),
               '--preset', preset,  
               '--frames', str(frames), 
               '--fps', '30', 
               '--output-depth', '8',
               '--input-depth', '8']
    if qpfile is not None: command.extend(['--qpfile', qpfile])
    if bitrate is not None: command.extend(['--bitrate', str(bitrate)])
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    # print(stderr)


# In[7]:


def get_filename(path):
    return path.split('/')[-1].split('.')[0]
    


# In[8]:


def mkdir(path):
    process = subprocess.Popen(['mkdir', path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    # print(stdout)


# In[9]:


def touch(path):
    process = subprocess.Popen(['touch', path],
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    # print(stdout, stderr)


# In[10]:


def calculate_bitrate(file_path, frames, fps=30):
    return os.path.getsize(file_path) / (frames / fps) / 1000 * 8


# In[ ]:




