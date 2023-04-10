#!/usr/bin/env python
# coding: utf-8
import collections
import sys
from multiprocessing import Pool
# In[6]:
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import compose, pipeline
import random
import copy
import metrics
import wandb
import time

import utils
import numpy as np
from tqdm import tqdm as tqdm
from sklearn.linear_model import LinearRegression

# In[7]:


STAT_FILE = ""
WANDB = ""


def get_psnr(mse):
    return 10 * np.log10((255 * 255) / mse)


def get_mse(psnr):
    return 255 * 255 / (10 ** (psnr / 10))


def close_plots():
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs: plt.close(fig)


def get_random_index(n):
    return random.randint(0, n - 1)


def get_two_random_indices(n):
    i = get_random_index(n)
    j = get_random_index(n)
    while i == j: j = get_random_index(n)
    if i > j: i, j = j, i
    return i, j


class GA:
    def __init__(self, file_path, frames_number, population_size=60, cross_prob=0.7,
                 resolution=(352, 288), bitrate=1200, hof=1, alpha=0.1, table=None,
                 core_vectors=None, init_file=None, stat_file=None, gop_size=4, train_epochs=100):
        self.table = table
        self.population_size = population_size
        self.cross_prob = cross_prob
        self.mutation_prob = 1 - cross_prob
        self.hof = hof
        self.bitrate = bitrate
        self.alpha = alpha
        self.bad_precision = 0
        self.steps = 0
        self.gop = gop_size
        self.train_epochs = train_epochs
        self.timer = 0
        self.fitted = set()

        self.estim = {}
        self.X = {}
        self.y = {}
        self.count_dots = {}

        self.estim['psnr'] = {}
        self.estim['size'] = {}
        self.X['psnr'] = collections.defaultdict(list)
        self.y['psnr'] = collections.defaultdict(list)
        self.X['size'] = collections.defaultdict(list)
        self.y['size'] = collections.defaultdict(list)
        self.count_dots['psnr'] = collections.defaultdict(dict)
        self.count_dots['size'] = collections.defaultdict(dict)
        self.delta_psnr = []
        self.delta_size = []

        self.file = file_path
        self.filename = utils.get_filename(self.file)
        self.frames = frames_number
        self.res = resolution
        self.best_one = []

        self.population = np.array([])
        self.next_population = np.array([])

        self.bitrates = np.zeros(self.population_size)
        self.psnrs = np.zeros(self.population_size)
        self.psnrs_pred = []
        self.bitrates_pred = []

        self.psnrs_diffs = []
        self.bitrates_diffs = []

        self.epoch = 0
        self.core_vectors = core_vectors

        if init_file is None:
            self.init_population()
        else:
            self.read_population(init_file)

        self.stat_file = stat_file

    def init_population(self):
        ones = np.array(np.array(
            [np.array(
                [i] * self.frames
            ) for i in range(20, 42)]))

        good_randoms = np.array(np.array(
            [np.array(
                [random.randint(20, 42) for _ in range(self.frames)]
            ) for _ in range(min(self.population_size // 10, self.population_size - len(ones)))]))

        randoms = np.array(np.array(
            [np.array(
                [random.randint(0, 51) for _ in range(self.frames)]
            ) for _ in range(self.population_size - len(ones) - len(good_randoms))]))

        print(ones)
        print(good_randoms)
        print(randoms)
        if len(good_randoms > 0):
            self.next_population = np.concatenate((ones, good_randoms), axis=0)
        if len(randoms > 0):
            self.next_population = np.concatenate((self.next_population, randoms), axis=0)

        if self.core_vectors is not None:
            for i in range(len(self.core_vectors)):
                self.next_population[-i - 1] = np.array(self.core_vectors[i])

    # read initial population from 'file_path' file
    def read_population(self, file_path):
        population = []
        for individual in open(file_path, 'r'):
            v = individual.split()
            population.append(np.array(v))
        self.next_population = np.array(population)

    # clear statistic for after last epoch
    def clear_statistics(self):
        self.bitrates = np.zeros(self.population_size)
        self.psnrs = np.zeros(self.population_size)
        self.delta_size = []
        self.delta_psnr = []
        self.bitrates_pred = []
        self.psnrs_pred = []

    # get dict of metrics for wandb
    def get_metrics(self):
        good_psnrs = self.psnrs[self.bitrates <= self.bitrate]
        return {
            'bitrate/average': np.average(self.bitrates),
            'bitrate/min': np.min(self.bitrates),
            'psnr/average': np.average(self.psnrs),
            'psnr/max': np.max(self.psnrs),
            'psnr_good/average': np.mean(good_psnrs) if len(good_psnrs) > 0 else 0,
            'psnr_good/max': np.max(good_psnrs) if len(good_psnrs) > 0 else 0,
            'psnr_good/cnt': len(good_psnrs),
            'delta/psnr_diffs': sum(self.psnrs_diffs) / len(self.psnrs_diffs),
            'delta/bitrate_diffs': sum(self.bitrates_diffs) / len(self.bitrates_diffs),
            'time/epoch_time': round((time.time() - self.timer) / 60, 4)
        }

    # get plots for wandb
    def get_plots(self):
        plots = {}

        for what in ['psnr', 'size']:
            lens = []
            for i in range(self.frames):
                for qp in range(0, 52):
                    lens.append(((len(self.X[what][(i, qp)])), (i, qp)))
            frames_to_show = sorted(lens, reverse=True)[:6]
            for i in range(len(frames_to_show)):
                item = frames_to_show[i][1]
                f = plt.figure()
                plt.plot(self.X[what][item], self.y[what][item], 'ro')
                estim_x = [j for j in range(10, 60)]
                estim_y = [self.predict_estim(item, what, ex) for ex in estim_x]
                plt.plot(estim_x, estim_y)
                plt.ylabel("current frame " + what)
                plt.xlabel("previous frame psnr " + "\nframe=" + str(item[0]) + ", qp=" + str(item[1]))
                plots['estimations/' + what + '/' + str(i)] = f

        return plots

    # check if this epoch for train
    def check_train(self):
        return self.epoch < self.train_epochs or self.epoch % self.train_epochs == 0

    # dump all statistics for last epoch
    def dump_statistics(self):
        print(self.epoch, end=' ')
        print(self.best_one)

        if self.check_train():
            self.psnrs_diffs.append(np.average(np.abs(np.array(self.psnrs_pred) - self.psnrs)))
            if len(self.psnrs_diffs) == 3: self.psnrs_diffs.pop(0)

            self.bitrates_diffs.append(np.average(abs(np.array(self.bitrates_pred) - self.bitrates)))
            if len(self.bitrates_diffs) == 3: self.bitrates_diffs.pop(0)

        cur_metrics = self.get_metrics()

        if self.check_train():
            cur_metrics.update(self.get_plots())

        wandb.log(cur_metrics)

        if self.check_train():
            close_plots()

        for k, v in cur_metrics.items():
            print(f'{k}:\t{v}')

        with open(f"../stats/{fn}_{int(b)}_{pop_size}.best", 'a') as out:
            print('[', *self.best_one, ']', file=out, sep=',')
            print(self.epoch, self.bad_precision, file=out)

    # mutation function: after some epochs, range for mutation value decreases
    def mutate(self, ind):
        p = copy.deepcopy(self.population[ind])
        ind = get_random_index(self.frames)

        if self.epoch < 90:
            val = p[ind]
            p[ind] = random.randint(val * 9 // 10, min(51, val * 11 // 10))
        elif self.epoch < 150:
            val = p[ind]
            p[ind] = random.randint(val * 14 // 15, min(51, val * 16 // 15))
        else:
            val = p[ind]
            c = 1 if random.randint(0, 1) == 0 else -1
            p[ind] = min(51, max(0, val + c))

        return p

    # crossover function: child = weighted sum of parents
    def crossover(self, fp, sp):
        fparent = copy.deepcopy(self.population[fp])
        sparent = copy.deepcopy(self.population[sp])
        a = random.random()
        return np.round(fparent * a + (1 - a) * sparent).astype(int)
        # i = self.get_random_index(self.frames)
        # return np.concatenate((fparent[:i], sparent[i:])), np.concatenate((sparent[:i], fparent[i:]))
        # parents = [fp, sp]
        # fchild = []
        # schild = []
        # for i in range(self.frames):
        #     r = random.randint(0, 1)
        #     fchild.append(self.population[parents[r]][i])
        #     schild.append(self.population[parents[1 - r]][i])
        # return np.array(fchild), np.array(schild)

    # fill qpfile
    def write_qpfile(self, qp_path, qps):
        with open(qp_path, 'w') as out:
            for j in range(self.frames):
                c = 'I' if j % self.gop == 0 else 'P'
                print(j, c, qps[j], file=out)

    # add value to data for estimation table
    def add(self, k, what, X, y):
        a = round(X) if what == 'psnr' else X // 1000
        b = round(y)
        square = (a, b)
        if square not in self.count_dots[what][k].keys():
            self.count_dots[what][k][square] = 0

        if self.count_dots[what][k][square] < 5:
            self.X[what][k].append(X)
            self.y[what][k].append(y)
            self.count_dots[what][k][square] += 1

    # predict estimation table value
    def predict_estim(self, k, what, x_val):
        x_val = np.array([x_val]).reshape(-1, 1)

        x_data = np.array(self.X[what][k]).reshape(-1, 1)
        y_data = np.array(self.y[what][k]).reshape(-1, 1)

        if self.check_train() and len(x_data) > 0:
            if k not in self.estim[what].keys():
                self.estim[what][k] = compose.TransformedTargetRegressor(
                    regressor=pipeline.Pipeline([
                        ('scaler', StandardScaler()),
                        ('regressor', LinearRegression()),
                    ]),
                    transformer=StandardScaler(),
                )
            self.estim[what][k].fit(x_data, y_data)
            self.fitted.add((what, k))

        if (what, k) not in self.fitted:
            return x_val[0][0]

        return self.estim[what][k].predict(x_val)[0][0]

    # process individual i: run encoder and decoder and return results for frames
    def process(self, i):
        qp_vec = tuple(self.population[i])

        qp = '../tmp/QP' + str(i) + '.txt'
        encoded = '../tmp/encoded' + str(i) + '.yuv'
        decoded = '../tmp/decoded' + str(i) + '.yuv'

        self.write_qpfile(qp, qp_vec)
        _, err = utils.encode(self.file, encoded, self.res, self.frames, qpfile=qp, keyint=self.gop)
        if "x264 [error]: can't parse qpfile" in str(err):
            print("QP PARSE ERROR")
            sys.exit(1)
        out = [s.split() for s in str(err).split('\\n')]

        sz = []  # frame sizes in bytes
        for p in out:
            if 'frame=' in p:
                sz.append(int(p[11].split('=')[1]))

        utils.decode(encoded, decoded)

        _, mse = metrics.calculate_psnr(self.file, decoded, self.res, self.frames)

        utils.rm(encoded)
        utils.rm(decoded)
        utils.rm(qp)

        return mse, sz

    # get information about I-frame
    def i_info(self, ind, qp, what):
        return self.table[qp][ind][what]

    # run individual based on real mse and sz from encoder/decoder
    def run_individual(self, i, mse, sz):
        psnr_est, bitrate_est = self.run_individual_estim(i)
        self.psnrs_pred.append(psnr_est)
        self.bitrates_pred.append(bitrate_est)

        qp_vec = tuple(self.population[i])

        last_psnr = 0
        for j in range(self.frames):
            if j % self.gop == 0:
                cur_psnr = get_psnr(self.i_info(0, qp_vec[0], 'mse'))
            else:
                cur_psnr = get_psnr(mse[j])
                self.add((j, qp_vec[j]), 'psnr', last_psnr, cur_psnr)
                self.add((j, qp_vec[j]), 'size', last_psnr, sz[j])
            last_psnr = cur_psnr

        bitrate = sum(sz) / (self.frames / 30) / 1024 * 8
        psnr = get_psnr(sum(mse) / len(mse))
        return psnr, bitrate

    # run individual based on estimation tables
    def run_individual_estim(self, i):
        qp_vec = tuple(self.population[i])

        mse_sum = 0
        sz_sum = 0

        last_psnr = 0
        for j in range(self.frames):
            if j % self.gop == 0:  # I-frame
                psnr_pred = get_psnr(self.i_info(0, qp_vec[0], 'mse'))
                size_pred = self.i_info(0, qp_vec[0], 'bytes')
            else:  # P-frame
                psnr_pred = self.predict_estim((j, qp_vec[j]), 'psnr', last_psnr)
                size_pred = self.predict_estim((j, qp_vec[j]), 'size', last_psnr)
            last_psnr = psnr_pred

            mse_sum += get_mse(psnr_pred)
            sz_sum += size_pred

        psnr = get_psnr(mse_sum / self.frames)
        bitrate = sz_sum / (self.frames / 30) / 1024 * 8
        return psnr, bitrate

    def fitness(self, psnr, bitrate):
        penalty = max(0, (bitrate / self.bitrate - 1) * self.alpha * self.epoch)
        return psnr / 20 - penalty

    def make_children(self, parents):
        i, j = get_two_random_indices(len(parents))
        p = random.random()
        if p >= self.cross_prob:
            fc = self.crossover(parents[i], parents[j])
        else:
            fc = self.mutate(parents[i])
        return [fc]

    def step(self):
        self.timer = time.time()
        self.clear_statistics()
        self.population = self.next_population

        fitness = np.zeros(self.population_size)

        results = []

        if self.check_train():
            with Pool() as pool:
                processed = pool.map(self.process, [i for i in range(self.population_size)])

            i = 0
            for mse, sz in processed:
                results.append(self.run_individual(i, mse, sz))
                i += 1

        else:
            for i in range(self.population_size):
                results.append(self.run_individual_estim(i))

        for i in range(self.population_size):
            fitness[i] = self.fitness(*results[i])
            self.psnrs[i], self.bitrates[i] = results[i]

        next_population = []

        # good = individuals with bitrate less than target bitrate
        goods = np.argwhere(self.bitrates <= self.bitrate).flatten()
        good_psnrs = self.psnrs[goods]

        hof = []
        # choose the best good individual
        if len(good_psnrs) > 0:
            hof = good_psnrs.argsort()[-max(1, self.hof):][::-1]
            self.best_one = self.population[goods[hof[0]]]

        # choose hall of fame from the good individuals
        if self.hof > 0 and len(good_psnrs) > 0:
            next_population.extend(self.population[goods[hof]])

        # extend hall of fame with not good individuals
        if self.hof - len(next_population) > 0:
            hof = fitness.argsort()[-(self.hof - len(next_population)):][::-1]
            next_population.extend(self.population[hof])

        # roulette to find best parents
        fitness += abs(min(fitness))
        s = sum(fitness)

        best_parents = []
        rs = [random.uniform(0, s) for _ in range(self.population_size // 3)]

        cur_s = 0
        rs.sort()
        j = 0
        for i in range(len(fitness)):
            cur_s += fitness[i]
            while j < len(rs) and cur_s >= rs[j]:
                best_parents.append(i)
                j += 1

        # fill next population
        while len(next_population) < self.population_size:
            next_population.extend(self.make_children(best_parents))

        next_population = next_population[:self.population_size]
        self.next_population = np.array(next_population)

        self.dump_statistics()
        self.epoch += 1

    def fit(self, steps=200):
        wandb.init(project="vkr", name=self.filename + ' ' +
                                       str(self.bitrate) + ' ' +
                                       str(self.population_size) + ' | ' +

                                       WANDB, reinit=True)
        self.steps = steps
        for _ in tqdm(range(steps)):
            self.step()


target_bitrates = {
    # 'stefan_cif': [
    #     (3549.0703, 32.6645),
    #         (3751.2552, 33.1836),
    #         (5030.4583, 36.3082),
    #         (6231.3307, 38.6873),
    # ],

    'hall': [
        (300, 36.57),
        (400, 0.00)
        #     (998.1867, 31.5303),
        #     (1189.0406, 32.8398),
        #     (1507.6609, 34.7217),
        #     (1977.2789, 36.804),
    ],
    'foreman': [
        (600, 0.0),
        #     # (1232.0508, 30.7664),
        #     #     (1561.782, 32.1129),
        #     #     (2121.3531, 34.046),
        #     #     (2557.575, 35.3574),
    ],
    'football_cif': [
        (1500, 31.399),
        # (2066.744, 33.4538),
        #     (2286.6193, 34.0987),
        #     (2494.8056, 34.699),
        #     (2700.2335, 35.2675),
    ],
    'container': [
        (600, 33.763),
        # (1488.8469, 32.077),
        #     (1979.0609, 33.9638),
        #     (2177.9664, 34.6095),
        #     (2407.2797, 35.2946),
    ],
}

pop_size = 60
epochs = 150

frames = utils.read_frames()
for inp in target_bitrates.keys():
    fp = '../dataset/' + inp + '.yuv'
    fn = utils.get_filename(inp)

    res = (352, 288)
    fr_n = frames[fn]

    tab = '../tables/' + fn + '.txt'
    i_table = utils.read_table(tab, frames[fn])

    for b, x in target_bitrates[inp]:
        WANDB = str(x)

        cores = None
        open(f"../stats/{fn}_{int(b)}_{pop_size}.best", 'w').close()

        g = GA(fp, frames_number=fr_n, population_size=pop_size,
               bitrate=b-20, core_vectors=cores, table=i_table,
               gop_size=60, train_epochs=15)

        g.fit(steps=epochs)

        with open(f"../stats/{fn}_{int(b)}.pop", 'w') as out:
            for line in g.population: print(*line, file=out)
