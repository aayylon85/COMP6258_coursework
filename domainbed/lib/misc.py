# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import hashlib
import json
import os
import copy
import itertools
import sys
from shutil import copyfile
from collections import OrderedDict, defaultdict
from numbers import Number
import operator

import numpy as np
import torch
import tqdm
from collections import Counter
import networkx as nx
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def log_write(train_log, loss_msg):
    print(loss_msg)
    with open(train_log, "a") as f:
        f.write(loss_msg + "\n")

def to_np(x):
    return x.detach().cpu().numpy()

def read_pickle(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

def write_pickle(data, name):
    with open(name, "wb") as f:
        pickle.dump(data, f)

## DiWA ##
def get_score(results, test_envs, metric_key="acc"):
    val_env_keys = []
    for i in itertools.count():
        acc_key = f'env{i}_out_' + metric_key
        if acc_key in results:
            if i not in test_envs:
                val_env_keys.append(acc_key)
        else:
            break
    assert i > 0
    return np.mean([results[key] for key in val_env_keys])

## DiWA ##
class MergeDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        super(MergeDataset, self).__init__()
        self.datasets = datasets

    def __getitem__(self, key):
        count = 0
        for d in self.datasets:
            if key - count >= len(d):
                count += len(d)
            else:
                return d[key - count]
        raise ValueError(key)

    def __len__(self):
        return sum([len(d) for d in self.datasets])

def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
        torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
        torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).mean()

class MovingAverage:

    def __init__(self, ema, oneminusema_correction=True):
        self.ema = ema
        self.ema_data = {}
        self._updates = 0
        self._oneminusema_correction = oneminusema_correction

    def update(self, dict_data):
        ema_dict_data = {}
        for name, data in dict_data.items():
            data = data.view(1, -1)
            if self._updates == 0:
                previous_data = torch.zeros_like(data)
            else:
                previous_data = self.ema_data[name]

            ema_data = self.ema * previous_data + (1 - self.ema) * data
            if self._oneminusema_correction:
                # correction by 1/(1 - self.ema)
                # so that the gradients amplitude backpropagated in data is independent of self.ema
                ema_dict_data[name] = ema_data / (1 - self.ema)
            else:
                ema_dict_data[name] = ema_data
            self.ema_data[name] = ema_data.clone().detach()

        self._updates += 1
        return ema_dict_data



def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights

def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)

def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def split_meta_train_test(minibatches, num_meta_test=1):
    n_domains = len(minibatches)
    perm = torch.randperm(n_domains).tolist()
    pairs = []
    meta_train = perm[:(n_domains-num_meta_test)]
    meta_test = perm[-num_meta_test:]

    for i,j in zip(meta_train, cycle(meta_test)):
         xi, yi = minibatches[i][0], minibatches[i][1]
         xj, yj = minibatches[j][0], minibatches[j][1]

         min_n = min(len(xi), len(xj))
         pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


def accuracy(network, loader, weights, device):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total


def predictions(network, loader, weights, device):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()

    y_all = []
    p_all = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)

            y_all.append(to_np(y))
            p_all.append(to_np(p))
            total += batch_weights.sum().item()
    network.train()
    return y_all, p_all


def _compute_appro_fisher_for_batch(batch, model, parameters, num_labels):

    with torch.autograd.set_grad_enabled(True):
        outputs = model(batch)
        target_gradients = torch.randn_like(outputs)
        n_params = len(parameters)
        fishers = [torch.zeros_like(param, requires_grad=False) for param in parameters]

        fisher = torch.autograd.grad(outputs=outputs, inputs=model.parameters(),
                                           grad_outputs=target_gradients, retain_graph=True)
        sq_grad = [g ** 2 for g in fisher]

    return sq_grad

def compute_fisher_for_model(model, dataset, num_labels):
    parameters = list(model.parameters())
    n_params = len(parameters)
    fishers = [torch.zeros_like(param) for param in parameters]
    print("fishers len", len(fishers))

    num_fishers = [len(fisher) for fisher in fishers]
    print("num_fishers", sum(num_fishers))

    n_examples = 0
    for batch, y in dataset:
        b, _, _, _ = batch.size()
        batch = batch.to("cuda")
        n_examples += b

        batch_fishers = _compute_appro_fisher_for_batch(batch, model, parameters, num_labels) # Approximate

        for i in range(n_params):
            fishers[i] += batch_fishers[i]

    for i in range(n_params):
        fishers[i] /= float(n_examples)

    fishers = [to_np(fisher.view(-1)) for fisher in fishers]

    fishers_final = []

    for fisher in fishers:
        fishers_final += list(fisher)

    fishers_final = np.asarray(fishers_final)

    return fishers_final


def get_centrality(A, name="degree"):
    G = nx.from_numpy_array(A)

    if name == "degree":
        centrality = nx.degree_centrality(G) # degree
    elif name == "betweenness":
        centrality = nx.betweenness_centrality(G, normalized=True) # betweenness
    elif name == "closeness":
        centrality = nx.closeness_centrality(G)
    elif name == "eigenvector":
        centrality = nx.eigenvector_centrality(G)
    elif name == "current_flow":
        centrality = nx.current_flow_closeness_centrality(G)
    elif name == "katz":
        centrality = nx.katz_centrality(G)
    elif name == "harmonic":
        centrality = nx.harmonic_centrality(G)

    # centrality = nx.closeness_centrality(G) # closeness
    # centrality = nx.eigenvector_centrality(G) # eigenvector
    centrality = np.asarray(list(centrality.values()), dtype=float)

    return np.around(centrality, 3)

def gradients_to_centrality(gradients, graph_thres, name="degree"):
    dis_matrix = cosine_distances(gradients)
    # dis_matrix = pairwise_distances(gradients)
    # Get percentile
    dis_matrix_flat = dis_matrix.reshape(-1)
    dis1 = np.percentile(dis_matrix_flat, graph_thres)
    # Unweighted graph
    A = np.zeros_like(dis_matrix)
    rule = (dis_matrix < dis1) & (dis_matrix != 0)
    A[rule] = 1
    centrality = get_centrality(A, name=name)
    # print(centrality)
    return centrality, A

class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)


def get_indices(models, combo):
    indices = []
    for c in combo:
        indices.append(str(models.index(c)))
    return indices

def list2str(lista):
    lista_str = [str(a) for a in lista]
    lista_str = ",".join(lista_str)
    return lista_str

def print_tree(d, acc_log, indent=0):
    for key, value in d.items():
        print('  ' * indent + str(key))
        log_write(acc_log, '  ' * indent + str(key))

        if isinstance(value, dict):
            print_tree(value, acc_log, indent + 1)
        else:
            print('  ' * (indent + 1) + str(value))
            log_write(acc_log, '  ' * (indent + 1) + str(value))

def format_accs(accs):
    accs = [round(acc * 100, 1) for acc in accs]
    return accs

def prediction_avg2(models_predictions):
    keys = list(models_predictions.keys())
    y = models_predictions[keys[0]][0][:-1]
    y = np.asarray(y).reshape(-1)

    num_classes = len(models_predictions[keys[0]][1][0][0])
    prediction = 0

    for key in keys:
        prediction_each = np.asarray(models_predictions[key][1][:-1])
        # print("prediction", prediction_each.shape)
        prediction += prediction_each

    print("predictions", prediction.shape)
    prediction = prediction.reshape(-1, num_classes)
    correct = (prediction.argmax(1) == y).sum()

    acc = correct/len(y)

    return acc


def prediction_avg(models_predictions):
    keys = list(models_predictions.keys())
    y = models_predictions[keys[0]][0][:-1]
    y = np.asarray(y)

    correct = 0
    total = 0

    prediction = 0
    for key in keys:
        prediction_each = np.asarray(models_predictions[key][1][:-1])
        # print("prediction", prediction_each.shape)
        prediction += prediction_each

    for i in range(len(y)):
        correct += (prediction[i].argmax(1) == y[i]).sum()
        total += len(y[i])

    return correct/total