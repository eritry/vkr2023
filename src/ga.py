import collections
import sys
from multiprocessing import Pool
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

# глобальная переменная с путем до файла в который будет писаться статистика по эпохам
STAT_FILE = ""
WANDB = ""


def get_psnr(mse):
    if mse == 0: return 255
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

# file_path - путь к исходному файлу
# frames_number - количество кадров в исходном файле
# population_size - размер популяции генетического алгоритма
# cross_prob - вероятность кроссинговера, 1-cross_prob = вероятность мутации
# resolution - разрешение исходного файла
# bitrate - заданное ограничение на битрейт
# hof - hall of fame, количество наилучших индивидов, которые попадут в следующее поколение
#       без отбора
# alpha - параметр нормализации для штрафа в фитнесс-функции, чем больше, тем значимее ощущется
#       превышение целевого битрейта
# table - предпосчитанные таблицы числа бит и psnr покадрово при кодировании в интра-режиме,
#       нужны всегда (используются для всех I кадров), создаются в make_tables.py
# core_vectors - вектор опорных векторов, используется для ускорения схождения обучения,
#       можно не указывать
# gop_size - длина GOP для режима без подбора типов кадров
# train_epochs - сколько эпох предобучать линейные регрессии
# choosing_types - булевый флаг, True если режим подбора типов кадров
#
#
class GA:
    def __init__(self, file_path, frames_number, population_size=60, cross_prob=0.8,
                 resolution=(352, 288), bitrate=1200, hof=8, alpha=0.1, table=None,
                 core_vectors=None, gop_size=4, train_epochs=20, choosing_types = False):
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
        self.choosing_types = choosing_types

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

        self.best_qps = []
        self.best_types = []
        self.checked_psnr = 0
        self.checked_best_qps = []

        self.population = np.array([])
        self.next_population = np.array([])
        self.types = np.array([])
        self.next_types = np.array([])

        self.bitrates = np.zeros(self.population_size)
        self.psnrs = np.zeros(self.population_size)
        self.psnrs_pred = []
        self.bitrates_pred = []

        self.psnrs_diffs = []
        self.bitrates_diffs = []

        self.epoch = 0
        self.core_vectors = core_vectors

        self.init_population()

    def init_population(self):
        ones = np.array(np.array(
            [np.array(
                [i] * self.frames
            ) for i in range(20, 42)]))

        randoms = np.array(np.array(
            [np.array(
                [random.randint(0, 51) for _ in range(self.frames)]
            ) for _ in range(self.population_size - len(ones))]))

        print(ones)
        print(randoms)

        qps = ones
        if len(randoms > 0):
            qps = np.concatenate((ones, randoms), axis=0)

        if self.choosing_types:
            types = np.array(np.array(
                [np.array(
                    [random.randint(0, 1) for _ in range(self.frames)]
                ) for _ in range(self.population_size)]))

            self.next_types = types

        self.next_population = qps

        # добавление core_vectors в популяцию
        if self.core_vectors is not None:
            for i in range(len(self.core_vectors)):
                self.next_population[-i - 1] = np.array(self.core_vectors[i])

        self.best_types = np.array([1] * self.frames)
        self.best_qps = self.next_population[-1]

    # clear statistic after last epoch
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
        return self.epoch < self.train_epochs or self.epoch % 15 < 3 or self.steps - self.epoch < 8

    def train_types(self):
        return self.choosing_types and self.epoch % 80 >= 40

    # dump all statistics for last epoch
    def dump_statistics(self):
        print(self.epoch, end=' ')
        print(self.best_qps)

        if self.check_train():
            self.psnrs_diffs.append(np.average(np.abs(np.array(self.psnrs_pred) - self.psnrs)))
            if len(self.psnrs_diffs) == 3: self.psnrs_diffs.pop(0)

            self.bitrates_diffs.append(np.average(abs(np.array(self.bitrates_pred) - self.bitrates)))
            if len(self.bitrates_diffs) == 3: self.bitrates_diffs.pop(0)

        cur_metrics = self.get_metrics()

        # if self.check_train():
        #     cur_metrics.update(self.get_plots())

        wandb.log(cur_metrics)

        # if self.check_train():
        #     close_plots()

        for k, v in cur_metrics.items():
            print(f'{k}:\t{v}')

        with open(STAT_FILE, 'a') as out:
            print("QP: ", file=out)
            for p in self.best_qps:
                print(p, end=' ', file=out)
            print(file=out)

            if self.choosing_types:
                print("TP: ", file=out)
                for p in self.best_types:
                    print(p, end=' ', file=out)
                print(file=out)

    def mutate_types(self, i):
        p = copy.deepcopy(self.types[i])
        ind = get_random_index(self.frames)
        p[ind] = 1 - p[ind]
        return p

    # mutation function: after some epochs, range for mutation value decreases
    def mutate(self, i):
        p = copy.deepcopy(self.population[i])
        ind = get_random_index(self.frames)
        qp = p
        val = qp[ind]
        if self.epoch < 90:
            qp[ind] = random.randint(val * 9 // 10, min(51, val * 11 // 10))
        elif self.epoch < 150:
            qp[ind] = random.randint(val * 14 // 15, min(51, val * 16 // 15))
        else:
            c = 1 if random.randint(0, 1) == 0 else -1
            qp[ind] = min(51, max(0, val + c))
        return qp

    def crossover_types(self, fp, sp):
        fparent = copy.deepcopy(self.types[fp])
        sparent = copy.deepcopy(self.types[sp])
        i = get_random_index(self.frames)
        return np.concatenate((fparent[:i], sparent[i:]))

    # crossover function: child = weighted sum of parents
    def crossover(self, fp, sp):
        fparent = copy.deepcopy(self.population[fp])
        sparent = copy.deepcopy(self.population[sp])
        a = random.random()
        return np.round(fparent * a + (1 - a) * sparent).astype(int)

    # fill qpfile
    def write_qpfile(self, qp_path, qps, types=None):
        with open(qp_path, 'w') as out:
            for j in range(self.frames):
                if types is None:
                    c = 'I' if j % self.gop == 0 else 'P'
                else:
                    c = 'I' if types[j] == 0 else 'P'
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
        qp_path = '../tmp/QP' + str(i) + '.txt'
        encoded = '../tmp/encoded' + str(i) + '.yuv'
        decoded = '../tmp/decoded' + str(i) + '.yuv'

        if self.choosing_types:
            if self.train_types():
                self.write_qpfile(qp_path, self.best_qps, self.types[i])
            else:
                self.write_qpfile(qp_path, self.population[i], self.best_types)
        else:
            self.write_qpfile(qp_path, self.population[i])

        _, err = utils.encode(self.file, encoded, self.res, self.frames, qpfile=qp_path, preset='veryslow')
        if "x264 [error]: can't parse qpfile" in str(err):
            print("QP PARSE ERROR")
            sys.exit(1)
        out = [s.split() for s in str(err).split('\\n')]

        sz = []  # frame sizes in bytes
        for p in out:
            if 'bytes' in p:
                index = p.index('bytes')
                sz.append(int(p[index - 1].split('=')[1]))

        utils.decode(encoded, decoded)

        _, mse = metrics.calculate_psnr(self.file, decoded, self.res, self.frames)

        utils.rm(encoded)
        utils.rm(decoded)
        utils.rm(qp_path)

        return mse, sz

    # get information about I-frame
    def i_info(self, ind, qp, what):
        return self.table[qp][what][ind]

    # run individual based on real mse and sz from encoder/decoder
    def run_individual(self, i, mse, sz):
        psnr_est, bitrate_est = self.run_individual_estim(i)
        self.psnrs_pred.append(psnr_est)
        self.bitrates_pred.append(bitrate_est)

        if self.train_types():
            qp_vec = self.best_qps
        else:
            qp_vec = self.population[i]

        last_psnr = 0
        for j in range(self.frames):
            cur_psnr = get_psnr(mse[j])
            self.add((j, qp_vec[j]), 'psnr', last_psnr, cur_psnr)
            self.add((j, qp_vec[j]), 'size', last_psnr, sz[j])
            last_psnr = cur_psnr

        bitrate = sum(sz) / (self.frames / 30) / 1024 * 8
        psnr = get_psnr(sum(mse) / len(mse))
        return psnr, bitrate

    # run individual based on estimation tables
    def run_individual_estim(self, i):
        if self.train_types():
            qp_vec = self.best_qps
            types = self.types[i]
        else:
            qp_vec = self.population[i]
            types = self.best_types

        mse_sum = 0
        sz_sum = 0

        last_psnr = 0
        for j in range(self.frames):
            if (self.choosing_types and (j == 0 or types[j] == 0))\
                    or (not self.choosing_types and j % self.gop == 0): # I-frame
                psnr_pred = get_psnr(self.i_info(j, qp_vec[j], 'mse'))
                size_pred = self.i_info(j, qp_vec[j], 'bytes')
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
        if p <= self.cross_prob:
            if self.train_types():
                fc = self.crossover_types(parents[i], parents[j])
            else:
                fc = self.crossover(parents[i], parents[j])
        else:
            if self.train_types():
                fc = self.mutate_types(parents[i])
            else:
                fc = self.mutate(parents[i])
        return [fc]

    def step(self):
        self.timer = time.time()
        self.clear_statistics()
        self.population = self.next_population
        self.types = self.next_types

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
            best_index = goods[hof[0]]

            if self.train_types():
                self.best_types = self.types[best_index]
            else:
                self.best_qps = self.population[best_index]
                if self.check_train():
                    if self.checked_psnr < self.psnrs[best_index]:
                        self.checked_psnr = self.psnrs[best_index]
                        if self.train_types():
                            self.checked_best_types = self.types[best_index]
                            self.checked_best_qps = self.best_qps
                        else:
                            self.checked_best_types = self.best_types
                            self.checked_best_qps = self.population[best_index]


        # choose hall of fame from the good individuals
        if self.hof > 0 and len(good_psnrs) > 0:
            if self.train_types():
                next_population.extend(self.types[goods[hof]])
            else:
                # print(goods)
                # print(hof)

                next_population.extend(self.population[goods[hof].astype(int)])

        # extend hall of fame with not good individuals
        if self.hof - len(next_population) > 0:
            hof = fitness.argsort()[-(self.hof - len(next_population)):][::-1]
            if self.train_types():
                next_population.extend(self.types[hof])
            else:
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

        if self.train_types():
            self.next_population = self.population
            self.next_types = np.array(next_population)
        else:
            self.next_population = np.array(next_population)
            self.next_types = self.types

        self.dump_statistics()
        self.epoch += 1

    def fit(self, steps=200):
        self.steps = steps
        wandb.init(project="vkr", name=self.filename + ' ' +
                                       str(self.bitrate) + ' ' +
                                       str(self.population_size) + ' ' +
                                       str(self.steps) + ' | ' +
                                       WANDB, reinit=True)
        for _ in tqdm(range(steps)):
            self.step()

# задание файлов для тестирования в формате имя : ограничение на битрейт
target_bitrates = {
    'container': (100, 200),
    'akiyo': (30, 50),
}

# размер популяции
pop_size = 60

# количество эпох
epochs = 2000

# считываем информацию о количестве кадров в каждом файле
frames = utils.read_frames()

for inp in target_bitrates.keys():
    fp = '../dataset/' + inp + '.yuv'
    enc = '../encoded/cores.264'
    dec = '../decoded/cores.yuv'
    fn = utils.get_filename(inp)

    res = (352, 288) # исходное разрешение
    fr_n = frames[fn] # исходное количество кадров

    tab = '../tables/' + fn + '.pickle'
    i_table = utils.read_table_pickle(tab) # чтение таблицы

    for b in target_bitrates[inp]:
        WANDB = "ipp"
        STAT_FILE = f"../stats/ipp/{fn}_{int(b)}_{pop_size}_{fr_n}.best" # переопределение файла для статистики
        open(STAT_FILE, 'w').close() # очистка файла для статистики

        cores = [utils.get_core_vector(fp, enc, dec, res, fr_n, b)] # получение опорных векторов

        t = time.time()

        g = GA(fp, frames_number=fr_n, population_size=pop_size,
               bitrate=b, core_vectors=cores, table=i_table, choosing_types=False,
               train_epochs=35) # инициализация алгоритма

        g.fit(steps=epochs) # работа алгоритма

        # запись лучшего индивида в файл статистики
        with open(STAT_FILE, 'a') as out:
            print("CHECKED QP:", file=out)
            for p in g.checked_best_qps:
                print(p, end=' ', file=out)

            print(file=out)
            if g.choosing_types:
                print("CHECKED TP:", file=out)
                for p in g.checked_best_types:
                    print(p, end=' ', file=out)
                print(file=out)

        with open(f"../stats/ipip/{fn}_{int(b)}_{pop_size}_{fr_n}.time", 'w') as outfile:
            print((time.time() - t) / 60, file=outfile)