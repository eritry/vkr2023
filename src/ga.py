#!/usr/bin/env python
# coding: utf-8

# In[6]:


import random
import copy
import metrics
import wandb

import utils
import numpy as np
import os
import multiprocessing
from tqdm import tqdm as tqdm

# In[7]:


STAT_FILE = ""


# In[8]:


class GA:
    def __init__(self, file_path, frames_number, population_size=60, cross_prob=0.85,
                 resolution=(352, 288), bitrate=1200, hof=5, alpha=0.01, init_file=None, stat_file=None):
        self.population_size = population_size
        self.cross_prob = cross_prob
        self.mutation_prob = 1 - cross_prob
        self.hof = hof
        self.bitrate = bitrate
        self.alpha = alpha

        self.file = file_path
        self.filename = utils.get_filename(self.file)
        self.frames = frames_number
        self.res = resolution

        self.population = np.array([])
        self.next_population = np.array([])

        self.bitrates = np.zeros(self.population_size)
        self.psnrs = np.zeros(self.population_size)
        self.epoch = 0

        if init_file is None:
            self.init_population()
        else:
            self.read_population(init_file)

        self.stat_file = stat_file

    def init_population(self):
        self.next_population = np.array(np.array(
            [np.array(
                [random.randint(0, 81) for _ in range(self.frames)]
            ) for _ in range(self.population_size)]))

    def read_population(self, file_path):
        population = []
        for line in open(file_path, 'r'):
            v = line.split()
            population.append(np.array(v))
        self.next_population = np.array(population)

    def clear_statistics(self):
        self.bitrates = np.zeros(self.population_size)
        self.psnrs = np.zeros(self.population_size)

    def get_metrics(self):
        good_psnrs = self.psnrs[self.bitrates < 1200]
        return {
            'bitrate/average': np.average(self.bitrates),
            'bitrate/min': np.min(self.bitrates),
            'psnr/average': np.average(self.psnrs),
            'psnr/max': np.max(self.psnrs),
            'psnr_good/average': np.mean(good_psnrs) if len(good_psnrs) > 0 else 0,
            'psnr_good/max': np.max(good_psnrs) if len(good_psnrs) > 0 else 0,
            'psnr_good/cnt': len(good_psnrs),
        }

    def dump_statistics(self):
        print(self.epoch, end=' ')
        cur_metrics = self.get_metrics()
        wandb.log(cur_metrics)
        for k, v in cur_metrics.items():
            print(f'{k}:\t{v}')

    def dump_statistics_file(self):
        for k, v in self.get_metrics().items():
            print(f'{k}:\t{v}', file=STAT_FILE)

    def get_random_index(self, n):
        return random.randint(0, n - 1)

    def get_two_random_indices(self, n):
        i = self.get_random_index(n)
        j = self.get_random_index(n)
        while i == j: j = self.get_random_index(n)
        if i > j: i, j = j, i
        return i, j

    def mutate(self, ind):
        p = copy.deepcopy(self.population[ind])
        p[self.get_random_index(self.frames)] = random.randint(0, 81)
        return p

    def crossover(self, fp, sp):
        parents = [fp, sp]
        fchild = []
        schild = []
        for i in range(self.frames):
            r = random.randint(0, 1)
            fchild.append(self.population[parents[r]][i])
            schild.append(self.population[parents[1 - r]][i])
        return np.array(fchild), np.array(schild)

    def write_qpfile(self, qp_path, qps):
        with open(qp_path, 'w') as out:
            for j in range(self.frames):
                print(j, 'I', qps[j], file=out)

    def run_individual(self, i):
        qp_path = '../tmp/' + self.filename + "_" + str(i) + ".qp"
        encoded_path = '../tmp/' + self.filename + "_" + str(i) + ".264"
        decoded_path = '../tmp/' + self.filename + "_" + str(i) + ".yuv"

        self.write_qpfile(qp_path, self.population[i])

        utils.encode(self.file, encoded_path, resolution=self.res, frames=self.frames, qpfile=qp_path)
        utils.decode(encoded_path, decoded_path)

        psnr_y, pnsr_u, psnr_v, psnr_yuv = metrics.calculate_psnr(self.file, decoded_path, self.res, self.frames)
        bitrate = utils.calculate_bitrate(encoded_path, self.frames)

        os.remove(qp_path)
        os.remove(encoded_path)
        os.remove(decoded_path)

        return psnr_yuv, bitrate

    def fitness(self, psnr, bitrate):
        penalty = max(0, bitrate / self.bitrate) * self.alpha * self.epoch
        return psnr / 50 - penalty

    def make_children(self, parents):
        i, j = self.get_two_random_indices(len(parents))
        p = random.random()
        if p >= self.cross_prob:
            fc, sc = self.crossover(parents[i], parents[j])
            return [fc, sc]
        else:
            fc = self.mutate(parents[i])
            return [fc]

    def step(self):
        self.clear_statistics()
        self.population = self.next_population

        fitness = np.zeros(self.population_size)
        with multiprocessing.Pool() as pool:
            results = pool.map(self.run_individual, [i for i in range(self.population_size)])

        for i in range(self.population_size):
            fitness[i] = self.fitness(*results[i])
            self.psnrs[i], self.bitrates[i] = results[i]

        next_population = []
        if self.hof > 0:
            hof = fitness.argsort()[-self.hof:][::-1]
            next_population.extend(self.population[hof])

        fitness += min(fitness)
        s = sum(fitness)

        best_parents = []
        rs = [random.randint(0, int(s + 1)) for _ in range(self.population_size // 3)]

        for r in rs:
            cur_s = 0
            i = 0
            while cur_s <= r and i < len(fitness):
                cur_s += fitness[i]
                i += 1
            best_parents.append(i - 1)

        while len(next_population) < self.population_size:
            next_population.extend(self.make_children(best_parents))

        next_population = next_population[:self.population_size]

        self.next_population = np.array(next_population)

        self.dump_statistics()
        if self.stat_file is not None: self.dump_statistics_file()

        self.epoch += 1

    def fit(self, steps=200):
        wandb.init(project="vkr", name=str(self.population_size), reinit=True)
        for _ in tqdm(range(steps)):
            self.step()

for pop, st in [(10, 10), (100, 300), (900, 500), (1200, 500)]:
    STAT_FILE = open(f"../stefan_cif_stat_{pop}.pop", 'w')
    g = GA('../dataset/stefan_cif.yuv', frames_number=90, population_size=pop, stat_file=True)
    g.fit(steps=st)
    STAT_FILE.close()
    with open(f"../stefan_cif_{pop}.pop", 'w') as out:
        for line in g.population: print(*line, file=out)
