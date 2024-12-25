import argparse
import os
import json
import random
import numpy as np
import torch
import torch.utils.data
from domainbed import datasets, algorithms_inference
from domainbed.lib.fast_data_loader import FastDataLoader
from domainbed.lib import misc
import copy
import networkx as nx
from networkx.algorithms import approximation as approx
from itertools import combinations
import time
import cvxpy as cp
from scipy.optimize import minimize
from einops import rearrange
import scipy
from sklearn.metrics import mutual_info_score
# from scipy.optimize import minimize
from scipy.linalg import eigh, norm

def create_splits(domain, inf_args, dataset, _filter):
    splits = []

    for env_i, env in enumerate(dataset):
        if domain == "test" and env_i != inf_args.test_env:
            continue
        elif domain == "train" and env_i == inf_args.test_env:
            continue

        if _filter == "full":
            splits.append(env)
        else:
            out_, in_ = misc.split_dataset(
                env, int(len(env) * 0.2), misc.seed_hash(inf_args.trial_seed, env_i)
            )
            if _filter == "in":
                splits.append(in_)
            elif _filter == "out":
                splits.append(out_)
            else:
                raise ValueError(_filter)

    return splits

def get_dict_folder_to_score(inf_args):
    output_folders = [
        os.path.join(output_dir, path)
        for output_dir in inf_args.output_dir.split(",")
        for path in os.listdir(output_dir)
    ]
    output_folders = [
        output_folder for output_folder in output_folders
        if os.path.isdir(output_folder) and "done" in os.listdir(output_folder) and "model_best.pkl" in os.listdir(output_folder)
    ]

    # print("output_folders", output_folders)

    dict_folder_to_score = {}
    for folder in output_folders:
        model_path = os.path.join(folder, "model_best.pkl")
        save_dict = torch.load(model_path)
        train_args = save_dict["args"]

        if train_args["dataset"] != inf_args.dataset:
            continue

        # if train_args["test_envs"] != [inf_args.test_env]:
        #     continue

        if train_args["trial_seed"] != inf_args.trial_seed and inf_args.trial_seed != -1:
             continue
        score = misc.get_score(
            json.loads(save_dict["results"]),
            [inf_args.test_env])

        # print(f"Found: {folder} with score: {score}")
        dict_folder_to_score[folder] = score

    if len(dict_folder_to_score) == 0:
        raise ValueError(f"No folders found for: {inf_args}")
    return dict_folder_to_score

def get_dict_folder_to_score_topo(inf_args, dataset, data_splits, data_names, device):
    output_folders = [
        os.path.join(output_dir, path)
        for output_dir in inf_args.output_dir.split(",")
        for path in os.listdir(output_dir)
    ]

    output_folders = [
        output_folder for output_folder in output_folders
        if os.path.isdir(output_folder) and "done" in os.listdir(output_folder) and "model_best.pkl" in os.listdir(output_folder)
    ]

    dict_folder_to_score = {}
    gradients = []

    print("output_folders", output_folders)

    for folder in output_folders:
        model_path = os.path.join(folder, "model_best.pkl")
        save_dict = torch.load(model_path)
        train_args = save_dict["args"]

        if train_args["dataset"] != inf_args.dataset:
            continue
        # if train_args["test_envs"] != [inf_args.test_env]:
        #     continue
        if train_args["trial_seed"] != inf_args.trial_seed and inf_args.trial_seed != -1:
             continue

        # load individual weights
        algorithm = algorithms_inference.ERM(
            dataset.input_shape, dataset.num_classes,
            len(dataset) - 1,
            save_dict["model_hparams"]
        )

        algorithm.load_state_dict(save_dict["model_dict"], strict=False)
        algorithm.to(device)
        algorithm.eval()
        random.seed(train_args["seed"])
        np.random.seed(train_args["seed"])
        torch.manual_seed(train_args["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        data_loaders = [
            FastDataLoader(
                dataset=split,
                batch_size=64,
                num_workers=dataset.N_WORKERS
            ) for split in data_splits
        ]

        data_evals = zip(data_names, data_loaders)
        dict_results = {}

        print("data_names", data_names)

        for name, loader in data_evals:
            if name == 'test':
                fishers = misc.compute_fisher_for_model(algorithm.network, loader, dataset.num_classes)
                gradients.append(fishers)

    return gradients


def get_wa_results(good_checkpoints, dataset, data_names, data_splits, device):

    wa_algorithm = algorithms_inference.DiWA(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - 1,
    )

    for folder in good_checkpoints:
        save_dict = torch.load(os.path.join(folder, "model_best.pkl"))
        train_args = save_dict["args"]

        # load individual weights
        algorithm = algorithms_inference.ERM(
            dataset.input_shape, dataset.num_classes,
            len(dataset) - 1,
            save_dict["model_hparams"]
        )
        algorithm.load_state_dict(save_dict["model_dict"], strict=False)
        wa_algorithm.add_weights(algorithm.network)
        del algorithm

    wa_algorithm.to(device)
    wa_algorithm.eval()
    random.seed(train_args["seed"])
    np.random.seed(train_args["seed"])
    torch.manual_seed(train_args["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_loaders = [
        FastDataLoader(
            dataset=split,
            batch_size=64,
            num_workers=dataset.N_WORKERS
        ) for split in data_splits
    ]

    data_evals = zip(data_names, data_loaders)
    dict_results = {}

    print("data_names", data_names)

    for name, loader in data_evals:
        # print(f"Inference at {name}")
        if name == "test":
            dict_results[name + "_acc"] = misc.accuracy(wa_algorithm, loader, None, device)

    dict_results["length"] = len(good_checkpoints)
    return dict_results

def get_error_matrix(good_checkpoints, dataset, data_names, data_splits, device):

    models = []

    for folder in good_checkpoints:
        save_dict = torch.load(os.path.join(folder, "model_best.pkl"))
        train_args = save_dict["args"]
        # load individual weights
        algorithm = algorithms_inference.ERM(
            dataset.input_shape, dataset.num_classes,
            len(dataset) - 1,
            save_dict["model_hparams"]
        )
        algorithm.load_state_dict(save_dict["model_dict"], strict=False)
        models.append(algorithm)

    random.seed(train_args["seed"])
    np.random.seed(train_args["seed"])
    torch.manual_seed(train_args["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_loaders = [
        FastDataLoader(
            dataset=split,
            batch_size=64,
            num_workers=dataset.N_WORKERS
        ) for split in data_splits
    ]

    data_evals = zip(data_names, data_loaders)

    for model in models:
        model.to(device)
        model.eval()

    all_errors = []

    with torch.no_grad():
        for name, loader in data_evals:
            if name == "train":
                for inputs, labels in loader:
                    batch_errors = torch.zeros((len(models), len(labels)), dtype=torch.float32)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    for i, model in enumerate(models):
                        outputs = model.predict(inputs)
                        predictions = torch.argmax(outputs, dim=1)
                        errors = predictions != labels  # 1 for error, 0 for correct
                        batch_errors[i] = errors.float()
                    all_errors.append(batch_errors)

    P = torch.cat(all_errors, dim=1)
    G_tilde = torch.matmul(P, P.T)
    G_tilde = misc.to_np(G_tilde)

    return G_tilde, P.shape[1]

def get_model_predictions(good_checkpoints, dataset, data_names, data_splits, device):

    wa_algorithm = algorithms_inference.DiWA(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - 1,
    )

    for folder in good_checkpoints:
        save_dict = torch.load(os.path.join(folder, "model_best.pkl"))
        train_args = save_dict["args"]

        # load individual weights
        algorithm = algorithms_inference.ERM(
            dataset.input_shape, dataset.num_classes,
            len(dataset) - 1,
            save_dict["model_hparams"]
        )
        algorithm.load_state_dict(save_dict["model_dict"], strict=False)
        wa_algorithm.add_weights(algorithm.network)
        del algorithm

    wa_algorithm.to(device)
    wa_algorithm.eval()
    random.seed(train_args["seed"])
    np.random.seed(train_args["seed"])
    torch.manual_seed(train_args["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_loaders = [
        FastDataLoader(
            dataset=split,
            batch_size=64,
            num_workers=dataset.N_WORKERS
        ) for split in data_splits
    ]

    data_evals = zip(data_names, data_loaders)
    dict_results = {}

    print("data_names", data_names)
    predictions = []
    for name, loader in data_evals:
        print(f"Inference at {name}")
        predictions = misc.predictions(wa_algorithm, loader, None, device)

    return predictions

def get_model_predictions2(good_checkpoints, dataset, data_names, data_splits, device):

    algorithms = []

    for folder in good_checkpoints:
        save_dict = torch.load(os.path.join(folder, "model_best.pkl"))
        train_args = save_dict["args"]
        # load individual weights
        algorithm = algorithms_inference.ERM(
            dataset.input_shape, dataset.num_classes,
            len(dataset) - 1,
            save_dict["model_hparams"]
        )
        algorithm.load_state_dict(save_dict["model_dict"], strict=False)
        algorithms.append(algorithm)


    random.seed(train_args["seed"])
    np.random.seed(train_args["seed"])
    torch.manual_seed(train_args["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_loaders = [
        FastDataLoader(
            dataset=split,
            batch_size=64,
            num_workers=dataset.N_WORKERS
        ) for split in data_splits
    ]

    data_evals = zip(data_names, data_loaders)
    dict_results = {}

    print("data_names", data_names)
    predictions = []

    for algorithm in algorithms:
        algorithm.to(device)
        algorithm.eval()
        for name, loader in data_evals:
            print(f"Inference at {name}")
            prediction = misc.predictions(algorithm, loader, None, device)
            predictions.append(prediction)

    return predictions

def get_model_predictions3(models, dataset, data_names, data_splits, device):

    algorithms = []

    for folder in models:
        save_dict = torch.load(os.path.join(folder, "model_best.pkl"))
        train_args = save_dict["args"]
        # load individual weights
        algorithm = algorithms_inference.ERM(
            dataset.input_shape, dataset.num_classes,
            len(dataset) - 1,
            save_dict["model_hparams"]
        )
        algorithm.load_state_dict(save_dict["model_dict"], strict=False)
        algorithms.append(algorithm)

    random.seed(train_args["seed"])
    np.random.seed(train_args["seed"])
    torch.manual_seed(train_args["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_loaders = [
        FastDataLoader(
            dataset=split,
            batch_size=64,
            num_workers=dataset.N_WORKERS
        ) for split in data_splits
    ]

    data_evals = zip(data_names, data_loaders)
    models_predictions = {}

    for model in models:
        models_predictions[model] = [[], []]

    for algorithm in algorithms:
        algorithm.to(device)
        algorithm.eval()

    for name, loader in data_evals:
        print(f"Inference at {name}")
        if name == 'test':
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device)
                    y = y.to(device)

                    for network, model in zip(algorithms, models):
                        p = network.predict(x)
                        models_predictions[model][0].append(misc.to_np(y))
                        models_predictions[model][1].append(misc.to_np(p))

    return models_predictions

def print_results(dict_results):
    results_keys = sorted(list(dict_results.keys()))
    misc.print_row(results_keys, colwidth=12)
    misc.print_row([dict_results[key] for key in results_keys], colwidth=12)

def restricted(dict_folder_to_score, train_log, dataset, data_names, data_splits, device):

    dict_folder_list = list(dict_folder_to_score.keys())
    sorted_checkpoints = sorted(dict_folder_to_score.keys(), key=lambda x: dict_folder_to_score[x], reverse=True)
    sorted_indexes = [dict_folder_list.index(checkpoint) for checkpoint in sorted_checkpoints]
    val_accs = [dict_folder_to_score[checkpoint] for checkpoint in sorted_checkpoints]

    misc.log_write(train_log, 'Restricted')
    # misc.log_write(train_log, 'Val accs: ' + ", ".join(val_accs))

    selected_indexes = []
    best_result = -float("inf")
    dict_best_results = {}
    accepted_model_index = []
    each_model_test_acc = []
    acc_restrict = {}

    ## incrementally add them to the WA
    for i in range(0, len(sorted_checkpoints)):
        selected_indexes.append(i)
        selected_checkpoints = [sorted_checkpoints[index] for index in selected_indexes]

        ood_results = get_wa_results(selected_checkpoints, dataset, data_names, data_splits, device)

        each_ood_results = get_wa_results([sorted_checkpoints[i]], dataset, data_names, data_splits, device)

        each_model_test_acc.append(each_ood_results["test_acc"])

        ood_results["i"] = i
        ## accept only if WA's accuracy is improved
        if ood_results["train_acc"] >= best_result:
            dict_best_results = ood_results
            ood_results["accept"] = 1
            best_result = ood_results["train_acc"]
            print(f"Accepting index {i}")
            accepted_model_index.append(sorted_indexes[i])
        else:
            ood_results["accept"] = 0
            selected_indexes.pop(-1)
            print(f"Skipping index {i}")
        print_results(ood_results)

    # print final scores
    dict_best_results["final"] = 1
    print_results(dict_best_results)

    # Record results
    acc_restrict['acc'] = dict_best_results["test_acc"]
    acc_restrict['in_acc'] = dict_best_results["train_acc"]
    acc_restrict['test_accs'] = each_model_test_acc
    acc_restrict['val_accs'] = val_accs
    acc_restrict['sorted_indexes'] = sorted_indexes
    acc_restrict['accepted_model_index'] = accepted_model_index

    accepted_model_index = [str(s) for s in accepted_model_index]
    misc.log_write(train_log, 'Accepted index: ' + ", ".join(accepted_model_index))
    misc.log_write(train_log, "Acc: " + '%.4f' % (dict_best_results["test_acc"]))

    return acc_restrict

def maximum_independent(train_log, graph_threses, dict_folder_to_score, gradients, inf_args, dataset, data_names, data_splits, device):

    independent_sets = []
    models = list(dict_folder_to_score.keys())
    acc_mis = {}

    misc.log_write(train_log, 'maximum_independent')

    for graph_thres in graph_threses:
        misc.log_write(train_log, "graph_thres: " + str(graph_thres))
        centrality, A = misc.gradients_to_centrality(gradients, graph_thres)
        G = nx.Graph(A)
        I = approx.maximum_independent_set(G)
        misc.log_write(train_log, "maximum_independent_set: " + misc.list2str(I))
        independent_sets.append(I)
        misc.draw_graph_independent(G, I, graph_thres, inf_args.output_dir, inf_args.test_env)
        # print("graph_thres", graph_thres)
        independent_models = [models[index] for index in I]
        dict_results = get_wa_results(independent_models, dataset, data_names, data_splits, device)
        print_results(dict_results)
        misc.log_write(train_log, str(dict_results["test_acc"]))
        acc_mis[graph_thres] = dict_results["test_acc"]

    acc_mis["independent_sets"] = independent_sets

    return acc_mis

def graph_centrality(graph_threses, train_log, gradients, ks, dict_folder_to_score, dataset, data_names, data_splits, device, cen_name="degree"):

    acc_centrality = {}
    models = list(dict_folder_to_score.keys())

    # misc.log_write(train_log, 'Centrality')

    for graph_thres in graph_threses:

        acc_thres = {}

        misc.log_write(train_log, "graph_thres: " + str(graph_thres))

        centrality, A = misc.gradients_to_centrality(gradients, graph_thres, name=cen_name)

        print("centrality", centrality)

        sorted_pairs = sorted(enumerate(centrality), key=lambda x: x[1])
        sorted_indices = [pair[0] for pair in sorted_pairs]
        sorted_values = [pair[1] for pair in sorted_pairs]

        acc_thres["sorted_indices"] = sorted_indices
        acc_thres["sorted_values"] = sorted_values

        # misc.log_write(train_log, "sorted_indices: " + misc.list2str(sorted_indices))
        # misc.log_write(train_log, "sorted_values: " + misc.list2str(sorted_values))

        for k in ks:
            top_k_indices = sorted_indices[:k]
            top_k_models = [models[index] for index in top_k_indices]
            start_time = time.time()
            dict_results = get_wa_results(top_k_models, dataset, data_names, data_splits, device)
            end_time = time.time()
            print("time", '%.2f' % (end_time - start_time))
            print_results(dict_results)
            acc_thres[k] = dict_results["test_acc"]
            misc.log_write(train_log, str(k) + ': ' + str(dict_results["test_acc"]))

        acc_centrality[graph_thres] = acc_thres

    return acc_centrality

def graph_centrality_ablation(graph_threses, train_log, gradients, ks, dict_folder_to_score, dataset, data_names, data_splits, device, cen_names):

    acc_all = {}
    models = list(dict_folder_to_score.keys())[:10]

    # misc.log_write(train_log, 'Centrality')

    for graph_thres in graph_threses:

        acc_thres = {}

        misc.log_write(train_log, "graph_thres: " + str(graph_thres))

        for cen_name in cen_names:

            acc_cen = {}

            misc.log_write(train_log, cen_name + ": ")

            centrality, A = misc.gradients_to_centrality(gradients, graph_thres, name=cen_name)

            sorted_pairs = sorted(enumerate(centrality), key=lambda x: x[1])
            sorted_indices = [pair[0] for pair in sorted_pairs]
            sorted_values = [pair[1] for pair in sorted_pairs]

            acc_cen["centrality"] = centrality
            acc_cen["sorted_indices"] = sorted_indices
            acc_cen["sorted_values"] = sorted_values

            misc.log_write(train_log, "centrality: " + misc.list2str(centrality))
            misc.log_write(train_log, "sorted_indices: " + misc.list2str(sorted_indices))
            misc.log_write(train_log, "sorted_values: " + misc.list2str(sorted_values))

            for k in ks:
                top_k_indices = sorted_indices[:k]
                top_k_models = [models[index] for index in top_k_indices]
                dict_results = get_wa_results(top_k_models, dataset, data_names, data_splits, device)
                print_results(dict_results)
                acc_cen[k] = dict_results["test_acc"]
                misc.log_write(train_log, str(k) + ': ' + str(dict_results["test_acc"]))

            acc_thres[cen_name] = acc_cen

        acc_all[graph_thres] = acc_thres

    return acc_all

def graph_centrality_ablation_in(graph_threses, train_log, gradients, ks, dict_folder_to_score, dataset, data_names, data_splits, device, cen_names):

    acc_all = {}
    models = list(dict_folder_to_score.keys())

    # misc.log_write(train_log, 'Centrality')

    for graph_thres in graph_threses:

        acc_thres = {}

        misc.log_write(train_log, "graph_thres: " + str(graph_thres))

        for cen_name in cen_names:

            acc_cen = {}

            misc.log_write(train_log, cen_name + ": ")

            centrality, A = misc.gradients_to_centrality(gradients, graph_thres, name=cen_name)

            sorted_pairs = sorted(enumerate(centrality), key=lambda x: x[1])
            sorted_indices = [pair[0] for pair in sorted_pairs]
            sorted_values = [pair[1] for pair in sorted_pairs]

            acc_cen["centrality"] = centrality
            acc_cen["sorted_indices"] = sorted_indices
            acc_cen["sorted_values"] = sorted_values

            misc.log_write(train_log, "centrality: " + misc.list2str(centrality))
            misc.log_write(train_log, "sorted_indices: " + misc.list2str(sorted_indices))
            misc.log_write(train_log, "sorted_values: " + misc.list2str(sorted_values))

            for k in ks:
                top_k_indices = sorted_indices[:k]
                top_k_models = [models[index] for index in top_k_indices]
                dict_results = get_wa_results(top_k_models, dataset, data_names, data_splits, device)
                print_results(dict_results)
                acc_cen[k] = dict_results["train_acc"]
                misc.log_write(train_log, str(k) + ': ' + str(dict_results["train_acc"]))

            acc_thres[cen_name] = acc_cen

        acc_all[graph_thres] = acc_thres

    return acc_all

def matrix_normalization(G, N): # N: number of training points

    tilde_G = np.zeros_like(G, dtype=float)
    # Compute tilde_G_ii
    np.fill_diagonal(tilde_G, G.diagonal() / N)
    # Compute tilde_G_ij for i != j
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            if i != j:
                tilde_G[i, j] = 0.5 * (G[i, j] / G[i, i] + G[i, j] / G[j, j])

    return tilde_G

def linear_program_reg(G_tilde, D, k, lambda_param):

    n = G_tilde.shape[0]

    # Define the optimization variables
    x = cp.Variable(n)

    # Define the objective function
    objective = cp.Minimize(cp.quad_form(x, G_tilde) + lambda_param * cp.quad_form(x, D))
    # objective = cp.Minimize(cp.quad_form(x, G_tilde))

    # Define the constraints
    constraints = [cp.sum(x) == k, x >= 0, x <= 1]

    # Formulate the problem and solve
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Output the solution
    x_value = x.value

    return x_value


def make_psd_eigen(matrix):
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # Set negative eigenvalues to 0
    eigenvalues[eigenvalues < 0] = 0

    # Reconstruct the matrix
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

def make_psd_diag(matrix, epsilon=1e-10):
    # min_eigenvalue = np.min(np.linalg.eigvalsh(matrix))
    # if min_eigenvalue < 0:
    #     matrix = matrix + np.eye(matrix.shape[0]) * (-min_eigenvalue + epsilon)
    # return matrix

    return matrix @ matrix
