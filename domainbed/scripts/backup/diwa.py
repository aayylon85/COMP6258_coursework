import torch.utils.data
from domainbed.scripts.utils import *
from itertools import combinations
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity, pairwise_distances
import warnings
import numpy as np
from itertools import combinations
from scipy.special import binom
# Ignore all warnings
warnings.filterwarnings('ignore')

from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
# from pymoo.factory import get_sampling, get_crossover, get_mutation

class EnsemblePruningProblem(ElementwiseProblem):
    def __init__(self, G_tilde, W, K):
        super().__init__(n_var=G_tilde.shape[0],
                         n_obj=2,
                         n_constr=1,
                         xl=0,
                         xu=5)
        self.G_tilde = G_tilde
        self.W = W
        self.K = K

    def _evaluate(self, x, out, *args, **kwargs):
        Z = np.outer(x, x)  # Construct Z based on decision variable x
        obj1 = np.trace(self.G_tilde @ Z)  # Objective 1: Minimize trace(G_tilde @ Z)
        obj2 = np.trace(self.W @ Z)       # Objective 2: Minimize trace(W @ Z)
        out["F"] = [obj1, obj2]
        out["G"] = [self.K - np.sum(x)]




def _get_args():
    parser = argparse.ArgumentParser(description='Domain generalization')

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--test_env', type=int)

    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--exp_dir', type=str)

    parser.add_argument('--data_dir', type=str, default="default")
    # select which checkpoints
    parser.add_argument('--weight_selection', type=str, default="uniform") # or "restricted" or

    parser.add_argument('--graph_thres', type=float, default=50)

    parser.add_argument(
        '--trial_seed',
        type=int,
        default="-1",
        help='Trial number (used for seeding split_dataset and random_hparams).'
    )

    inf_args = parser.parse_args()
    return inf_args

def main():
    inf_args = _get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Begin DiWA for: {inf_args} with device: {device}")

    if inf_args.dataset in vars(datasets):
        dataset_class = vars(datasets)[inf_args.dataset]
        dataset = dataset_class(
            inf_args.data_dir, [inf_args.test_env], hparams={"data_augmentation": False}
        )
    else:
        raise NotImplementedError

    if not os.path.exists(inf_args.exp_dir):
        os.makedirs(inf_args.exp_dir)

    # load individual folders and their corresponding scores on train_out
    dict_folder_to_score = get_dict_folder_to_score(inf_args)

    # load data: test and optionally train_out for restricted weight selection
    data_splits, data_names = [], []
    dict_domain_to_filter = {"test": "full"}

    # if inf_args.weight_selection == "restricted" or inf_args.weight_selection == "topology":
    assert inf_args.trial_seed != -1
    dict_domain_to_filter["train"] = "out"

    for domain in dict_domain_to_filter:
        _data_splits = create_splits(domain, inf_args, dataset, dict_domain_to_filter[domain])
        if domain == "train":
            data_splits.append(misc.MergeDataset(_data_splits))
        else:
            data_splits.append(_data_splits[0])
        data_names.append(domain)

    # compute score after weight averaging
    if inf_args.weight_selection == "restricted":
        # Restricted weight selection
        ## sort individual members by decreasing accuracy on train_out (val_acc)

        train_log = inf_args.dataset + "/" + inf_args.dataset+ ".log"

        if not os.path.exists(train_log):
            with open(train_log, "w") as f:
                f.write("log start!\n")

        misc.log_write(train_log, "test_env: " + str(inf_args.test_env))

        dict_folder_list = list(dict_folder_to_score.keys())
        sorted_checkpoints = sorted(dict_folder_to_score.keys(), key=lambda x: dict_folder_to_score[x], reverse=True)
        sorted_indexes = [dict_folder_list.index(checkpoint) for checkpoint in sorted_checkpoints]
        val_accs = [dict_folder_to_score[checkpoint] for checkpoint in sorted_checkpoints]

        val_accs = ['%.2f' % (100*val_acc) for val_acc in val_accs]
        misc.log_write(train_log, 'Val accs: ' + ", ".join(val_accs))

        return

        selected_indexes = []
        best_result = -float("inf")
        dict_best_results = {}
        accepted_model_index = []
        each_model_test_acc = []

        ## incrementally add them to the WA
        for i in range(0, len(sorted_checkpoints)):
            selected_indexes.append(i)
            selected_checkpoints = [sorted_checkpoints[index] for index in selected_indexes]

            ood_results = get_wa_results(
                selected_checkpoints, dataset, data_names, data_splits, device
            )

            each_ood_results = get_wa_results(
                [sorted_checkpoints[i]], dataset, data_names, data_splits, device
            )
            each_model_test_acc.append('%.2f' % (100*each_ood_results["test_acc"]))

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

        ## print final scores
        dict_best_results["final"] = 1
        print_results(dict_best_results)
        sorted_indexes = [str(s) for s in sorted_indexes]
        accepted_model_index = [str(s) for s in accepted_model_index]

        misc.log_write(train_log, 'Val ranking: ' + ", ".join(sorted_indexes))
        misc.log_write(train_log, 'Test accs: ' + ", ".join(each_model_test_acc))
        misc.log_write(train_log, 'Accepted index: ' + ", ".join(accepted_model_index))
        misc.log_write(train_log, "Final acc: "+ '%.2f' % (100*dict_best_results["test_acc"]))

    elif inf_args.weight_selection == "topology":
        # Restricted weight selection
        dict_folder_to_score_topo = get_dict_folder_to_score_topo(inf_args, dataset, data_splits, data_names, device, inf_args.graph_thres)

        return
        # sort individual members by decreasing accuracy on train_out
        sorted_checkpoints = sorted(dict_folder_to_score_topo.keys(), key=lambda x: dict_folder_to_score_topo[x], reverse=True)

        selected_indexes = []
        best_result = -float("inf")
        dict_best_results = {}
        ## incrementally add them to the WA
        for i in range(0, len(sorted_checkpoints)):
            selected_indexes.append(i)
            selected_checkpoints = [sorted_checkpoints[index] for index in selected_indexes]

            ood_results = get_wa_results(
                selected_checkpoints, dataset, data_names, data_splits, device
            )

            ood_results["i"] = i
            ## accept only if WA's accuracy is improved
            if ood_results["train_acc"] >= best_result:
                dict_best_results = ood_results
                ood_results["accept"] = 1
                best_result = ood_results["train_acc"]
                print(f"Accepting index {i}")
            else:
                ood_results["accept"] = 0
                selected_indexes.pop(-1)
                print(f"Skipping index {i}")
            print_results(ood_results)

        ## print final scores
        dict_best_results["final"] = 1
        print_results(dict_best_results)

    elif inf_args.weight_selection == "uniform":
        dict_results = get_wa_results(
            list(dict_folder_to_score.keys()), dataset, data_names, data_splits, device
        )
        print_results(dict_results)

    elif inf_args.weight_selection == "uniform_ours": # Run twice: 1) Generate figs&accs 2) Merge them
        # print("keys", dict_folder_to_score.keys())
        acc_path = os.path.join(inf_args.output_dir, "accs_" + inf_args.dataset + '_' + str(inf_args.test_env))
        graph_path = os.path.join(inf_args.output_dir, 'figs/30_'+ str(inf_args.test_env)  +'.png')

        if os.path.exists(graph_path) and os.path.exists(acc_path):
            misc.combine_figs(inf_args.output_dir, inf_args.dataset, inf_args.test_env)
            return

        accs = []
        models = list(dict_folder_to_score.keys())
        num_models = len(models)

        # Average n models
        dict_results = get_wa_results(models, dataset, data_names, data_splits, device)
        print_results(dict_results)
        accs.append(dict_results["test_acc"])

        # Average n-1 models, models-list of model names.
        for i in range(num_models):
            models_copy = copy.copy(models)
            models_copy.pop(i)
            print("models_copy", len(models_copy))

            dict_results = get_wa_results(
                models_copy, dataset, data_names, data_splits, device
            )
            print_results(dict_results)
            accs.append(dict_results["test_acc"])

        print("accs", accs)

        misc.write_pickle(accs, acc_path)
        misc.draw_acc_bar(accs, inf_args.output_dir, inf_args.dataset, inf_args.test_env)

        gradients = get_dict_folder_to_score_topo(inf_args, dataset, data_splits, data_names, device, inf_args.graph_thres)
        misc.combine_figs(inf_args.output_dir, inf_args.dataset, inf_args.test_env)

    elif inf_args.weight_selection == "maximum_independent": # Run twice: 1) Generate figs&accs_independent 2) Merge them

        # If already got graph and acc figures, merge them:
        accs_independent_path = os.path.join(inf_args.output_dir, "accs_independent" + inf_args.dataset + '_' + str(inf_args.test_env))
        accs_path = os.path.join(inf_args.output_dir, "accs_" + inf_args.dataset + '_' + str(inf_args.test_env))
        graph_path = os.path.join(inf_args.output_dir, 'figs/30_independent_' + str(inf_args.test_env) + '.png')

        if os.path.exists(accs_independent_path) and os.path.exists(graph_path):
            accs_independent = misc.read_pickle(accs_independent_path)
            accs = misc.read_pickle(accs_path)
            misc.combine_figs_max_independent(inf_args.output_dir, inf_args.dataset, inf_args.test_env, accs_independent, accs)
            return

        gradients_path = os.path.join(inf_args.output_dir, "gradients_"+inf_args.dataset + '_' + str(inf_args.test_env))
        if os.path.exists(gradients_path):
            gradients = misc.read_pickle(gradients_path)
        else:
            gradients = get_dict_folder_to_score_topo(inf_args, dataset, data_splits, data_names, device, inf_args.graph_thres)

        graph_threses = [40, 50]
        independent_sets = []
        models = list(dict_folder_to_score.keys())
        accs = []

        for graph_thres in graph_threses:
            # print("before graph_thres", graph_thres)
            centrality, A = misc.gradients_to_centrality(gradients, graph_thres)
            G = nx.Graph(A)
            I = approx.maximum_independent_set(G)
            independent_sets.append(I)
            misc.draw_graph_independent(G, I, graph_thres, inf_args.output_dir, inf_args.test_env)
            # print("graph_thres", graph_thres)
            independent_models = [models[index] for index in I]
            dict_results = get_wa_results(independent_models, dataset, data_names, data_splits, device)
            print_results(dict_results)
            accs.append(dict_results["test_acc"])

        misc.write_pickle(accs, accs_independent_path)

    elif inf_args.weight_selection == "uniform_specify":

        models = list(dict_folder_to_score.keys())
        num_models = len(models)

        models.pop(0)
        models.pop(2)
        # models.pop(6)

        dict_results = get_wa_results(models, dataset, data_names, data_splits, device)
        print_results(dict_results)

    elif inf_args.weight_selection == 'oracle': # Run twice: 1) Generate figs&accs 2) Merge them

        accs = {}
        models = list(dict_folder_to_score.keys())
        ks = [7, 8]

        accs_path = os.path.join(inf_args.output_dir, "accs_oracle_" + inf_args.dataset + '_' + str(inf_args.test_env))

        for k in ks:
            combinations_list = list(combinations(models, k))
            # Print the list of combinations
            for combo in combinations_list:
                model_indices = misc.get_indices(models, combo)
                start_time = time.time()
                dict_results = get_wa_results(combo, dataset, data_names, data_splits, device)
                end_time = time.time()
                print("time", '%.2f'%(end_time-start_time))
                print_results(dict_results)
                accs[','.join(model_indices)] = dict_results["test_acc"]

        max_pair = max(accs.items(), key=lambda x: x[1])
        train_log = inf_args.dataset + "/" + inf_args.dataset + ".log"
        misc.log_write(train_log, max_pair[0] + ': ' + str(max_pair[1]))
        misc.write_pickle(accs, accs_path)

    elif inf_args.weight_selection == "centrality":  # Run twice: 1) Generate figs&accs_independent 2) Merge them

        train_log = inf_args.dataset + "/" + inf_args.dataset + "_centrality.log"

        if not os.path.exists(train_log):
            with open(train_log, "w") as f:
                f.write("log start!\n")

        misc.log_write(train_log, "test_env: " + str(inf_args.test_env))

        gradients_path = os.path.join(inf_args.output_dir, "gradients_" + inf_args.dataset + '_' + str(inf_args.test_env))

        gradients = misc.read_pickle(gradients_path)

        graph_threses = [40, 50, 60]
        independent_sets = []
        models = list(dict_folder_to_score.keys())
        accs = []

        ks = [3, 4, 5, 6, 7, 8, 9]

        for graph_thres in graph_threses:
            # print("before graph_thres", graph_thres)
            misc.log_write(train_log, "graph_thres: " + str(graph_thres))
            names = ['degree', "betweenness"]

            for name in names:
                misc.log_write(train_log, "name: " + str(name))

                centrality, A = misc.gradients_to_centrality(gradients, graph_thres, name)
                sorted_pairs = sorted(enumerate(centrality), key=lambda x: x[1])
                sorted_indices = [pair[0] for pair in sorted_pairs]
                sorted_values = [pair[1] for pair in sorted_pairs]
                misc.log_write(train_log, "sorted_indices: " + misc.list2str(sorted_indices))
                misc.log_write(train_log, "sorted_values: " + misc.list2str(sorted_values))

                for k in ks:
                    top_k_indices = sorted_indices[:k]
                    top_k_models = [models[index] for index in top_k_indices]
                    start_time = time.time()
                    dict_results = get_wa_results(top_k_models, dataset, data_names, data_splits, device)
                    end_time = time.time()
                    print("time", '%.2f' % (end_time - start_time))
                    print_results(dict_results)
                    misc.log_write(train_log, str(k) + ': ' + str(dict_results["test_acc"]))

            # accs.append(dict_results["test_acc"])
        # misc.write_pickle(accs, accs_independent_path)

    elif inf_args.weight_selection == "all":  # Run twice: 1) Generate figs&accs_independent 2) Merge them

        train_log = os.path.join(inf_args.exp_dir, inf_args.dataset + "_" + str(inf_args.test_env) +".log")
        acc_all = {}

        if not os.path.exists(train_log):
            with open(train_log, "w") as f:
                f.write("log start!\n")

        gradients_path = os.path.join(inf_args.output_dir, "gradients_" + str(inf_args.test_env))

        if os.path.exists(gradients_path):
            gradients = misc.read_pickle(gradients_path)
        else:
            gradients = get_dict_folder_to_score_topo(inf_args, dataset, data_splits, data_names, device)
            misc.write_pickle(gradients, gradients_path)

        # Uniform, Single model, acc
        dict_results = get_wa_results(list(dict_folder_to_score.keys()), dataset, data_names, data_splits, device)
        acc_all['uniform'] = dict_results["test_acc"]
        misc.log_write(train_log, 'Uniform: ' + str(dict_results["test_acc"]))

        # Restricted
        acc_restrict = restricted(dict_folder_to_score, train_log, dataset, data_names, data_splits, device)
        acc_all['restrict'] = acc_restrict

        # MIS
        # graph_threses = [40, 50, 60]
        # acc_mis = maximum_independent(train_log, graph_threses, dict_folder_to_score, gradients, inf_args, dataset, data_names, data_splits, device)
        # acc_all['mis'] = acc_mis

        # Centrality
        if len(gradients) == 10:
            ks = [5, 6, 7, 8, 9]
        else:
            ks = [10, 12, 14, 16, 18]

        cen_names = ["degree"]

        graph_threses = [40, 50]

        acc_centrality = graph_centrality_ablation(graph_threses, train_log, gradients, ks, dict_folder_to_score, \
                                            dataset, data_names, data_splits, device, cen_names)

        acc_all['centrality'] = acc_centrality

        acc_path = os.path.join(inf_args.exp_dir, "acc_all_" + str(inf_args.test_env))
        misc.write_pickle(acc_all, acc_path)

    elif inf_args.weight_selection == "restricted_remove":  # Run twice: 1) Generate figs&accs_independent 2) Merge them

        train_log = os.path.join(inf_args.exp_dir, inf_args.dataset + "_" + str(inf_args.test_env) + "_restric.log")

        if not os.path.exists(train_log):
            with open(train_log, "w") as f:
                f.write("log start!\n")

        # Restricted
        acc_all_path = os.path.join(inf_args.exp_dir, 'acc_all_'+str(inf_args.test_env))

        if os.path.exists(acc_all_path):
            acc_all = misc.read_pickle(acc_all_path)
            acc_restrict = acc_all['restrict']
        else:
            acc_restrict = restricted(dict_folder_to_score, train_log, dataset, data_names, data_splits, device)

        accepted_models = acc_restrict['accepted_model_index']
        accepted_models = sorted(accepted_models)

        models = list(dict_folder_to_score.keys())
        models = [models[i] for i in accepted_models]
        num_models = len(models)

        accs = [acc_restrict['acc']]

        misc.log_write(train_log, 'Greedy: ' + str(acc_restrict['acc']))

        # Average n-1 models, models-list of model names.
        for i in range(num_models):
            models_copy = copy.copy(models)
            models_copy.pop(i)
            print("models_copy", len(models_copy))
            dict_results = get_wa_results(models_copy, dataset, data_names, data_splits, device)
            print_results(dict_results)
            accs.append(dict_results["test_acc"])
            misc.log_write(train_log, 'w/o '+ str(accepted_models[i]) +": " + str(dict_results["test_acc"]))

        print("accs", accs)
        acc_path = os.path.join(inf_args.exp_dir, inf_args.dataset + "_" + str(inf_args.test_env) + "_restric")
        misc.write_pickle(accs, acc_path)
        misc.draw_acc_bar_restric(accs, inf_args.exp_dir, inf_args.dataset, inf_args.test_env)

    elif inf_args.weight_selection == "centrality_ablation":  # Run twice: 1) Generate figs&accs_independent 2) Merge them

        train_log = os.path.join(inf_args.exp_dir, inf_args.dataset +"_cen_ablation.log")

        if not os.path.exists(train_log):
            with open(train_log, "w") as f:
                f.write("log start!\n")

        misc.log_write(train_log, "test_env: " + str(inf_args.test_env))

        gradients_path = os.path.join(inf_args.output_dir, "gradients_" + str(inf_args.test_env))

        # if os.path.exists(gradients_path):
        #     gradients = misc.read_pickle(gradients_path)
        # else:
        gradients = get_dict_folder_to_score_topo(inf_args, dataset, data_splits, data_names, device)
        misc.write_pickle(gradients, gradients_path)

        gradients = gradients[:10]
        # Centrality
        if len(gradients) == 10:
            # ks = [5, 6, 7, 8, 9]
            ks = list(range(1, 11))
        else:
            ks = [10, 12, 14, 16, 18]

        # cen_names = ["degree", "betweenness", "closeness", "eigenvector", "harmonic"]
        cen_names = ["degree"]

        # graph_threses = [40, 50]
        graph_threses = [40]

        acc_all = graph_centrality_ablation(graph_threses, train_log, gradients, ks, dict_folder_to_score,\
                                          dataset, data_names, data_splits, device, cen_names)

        acc_path = os.path.join(inf_args.exp_dir, "ablation_cen_" + str(inf_args.test_env))
        misc.write_pickle(acc_all, acc_path)

    elif inf_args.weight_selection == "models_ablation":  # Run twice: 1) Generate figs&accs_independent 2) Merge them

        train_log = os.path.join(inf_args.exp_dir, inf_args.dataset + "_" + str(inf_args.test_env) +"_models.log")

        if not os.path.exists(train_log):
            with open(train_log, "w") as f:
                f.write("log start!\n")

        gradients_path = os.path.join(inf_args.output_dir, "gradients_" + str(inf_args.test_env))

        # if os.path.exists(gradients_path):
        #     gradients = misc.read_pickle(gradients_path)
        # else:

        gradients = get_dict_folder_to_score_topo(inf_args, dataset, data_splits, data_names, device)
        misc.write_pickle(gradients, gradients_path)

        num_models = [i for i in range(10, 22, 2)]

        acc_all = {}

        for num_model in num_models:

            acc_model = {}

            dict_models = list(dict_folder_to_score.items())[:num_model]

            dict_models = dict(dict_models)

            # Uniform, Single model, acc
            dict_results = get_wa_results(list(dict_models.keys()), dataset, data_names, data_splits, device)
            acc_model['uniform'] = dict_results["test_acc"]

            misc.log_write(train_log, 'Num of models: ' + str(num_model))
            misc.log_write(train_log, 'Uniform: ' + str(dict_results["test_acc"]))

            # Restricted
            acc_restrict = restricted(dict_models, train_log, dataset, data_names, data_splits, device)
            acc_model['restrict'] = acc_restrict

            # Centrality
            ks = [num_model//2]

            cen_names = ["degree"]

            graph_threses = [40]

            acc_centrality = graph_centrality_ablation(graph_threses, train_log, gradients[:num_model], ks, dict_models, \
                                                dataset, data_names, data_splits, device, cen_names)

            acc_model['centrality'] = acc_centrality

            acc_all[num_model] = acc_model

        acc_path = os.path.join(inf_args.exp_dir, "acc_all_models_" + str(inf_args.test_env))
        misc.write_pickle(acc_all, acc_path)

    elif inf_args.weight_selection == "top_ablation":  # Run twice: 1) Generate figs&accs_independent 2) Merge them

        train_log = os.path.join(inf_args.exp_dir, inf_args.dataset + "_" + str(inf_args.test_env) +"_top.log")

        if not os.path.exists(train_log):
            with open(train_log, "w") as f:
                f.write("log start!\n")

        nums_data = [0.1*i for i in range(1, 11)]
        num_models = len(dict_folder_to_score)

        acc_all = {}

        for num_data in nums_data:

            acc_data = {}

            misc.log_write(train_log, "Num of data: " + str(num_data))

            # gradients_path = os.path.join(inf_args.output_dir, "gradients_" + str(num_data) + "_" + str(inf_args.test_env))
            #
            # if os.path.exists(gradients_path):
            #     gradients = misc.read_pickle(gradients_path)
            # else:
            gradients = get_dict_folder_to_score_topo2(inf_args, dataset, data_splits, data_names, device, num_data)
            # misc.write_pickle(gradients, gradients_path)

            ks = [num_models//2]

            cen_names = ["degree"]

            graph_threses = [40]

            acc_centrality = graph_centrality_ablation(graph_threses, train_log, gradients, ks, dict_folder_to_score, \
                                                dataset, data_names, data_splits, device, cen_names)
            acc_data['centrality'] = acc_centrality

            acc_all[num_data] = acc_data

        acc_path = os.path.join(inf_args.exp_dir, "acc_all_topo_" + str(inf_args.test_env))
        misc.write_pickle(acc_all, acc_path)

    elif inf_args.weight_selection == "centrality_test":  # Run twice: 1) Generate figs&accs_independent 2) Merge them

        train_log = os.path.join(inf_args.exp_dir, inf_args.dataset +"_cen_test.log")

        if not os.path.exists(train_log):
            with open(train_log, "w") as f:
                f.write("log start!\n")

        misc.log_write(train_log, "test_env: " + str(inf_args.test_env))

        gradients_path = os.path.join(inf_args.output_dir, "gradients_" + str(inf_args.test_env))

        print("model names before", list(dict_folder_to_score.keys()))
        dict_models = list(dict_folder_to_score.items())[:5]
        dict_models = dict(dict_models)
        print("model names", list(dict_models.keys()))

        if os.path.exists(gradients_path):
            gradients = misc.read_pickle(gradients_path)
        else:
            gradients = get_dict_folder_to_score_topo(inf_args, dataset, data_splits, data_names, device)
            misc.write_pickle(gradients, gradients_path)

        # Centrality
        if len(gradients) == 10:
            ks = [5]
        else:
            ks = [10]

        cen_names = ["degree"]
        graph_threses = [40]

        acc_all = graph_centrality_ablation(graph_threses, train_log, gradients, ks, dict_folder_to_score,\
                                          dataset, data_names, data_splits, device, cen_names)

    elif inf_args.weight_selection == "uniform_remove":  # Run twice: 1) Generate figs&accs_independent 2) Merge them

        train_log = os.path.join(inf_args.exp_dir, inf_args.dataset + "_" + str(inf_args.test_env) + "_uniform.log")

        if not os.path.exists(train_log):
            with open(train_log, "w") as f:
                f.write("log start!\n")

        gradients_path = os.path.join(inf_args.output_dir, "gradients_" + str(inf_args.test_env))
        gradients = get_dict_folder_to_score_topo(inf_args, dataset, data_splits, data_names, device)
        misc.write_pickle(gradients, gradients_path)

        models = list(dict_folder_to_score.keys())
        num_models = len(models)

        accs = []
        # Average n models
        dict_results = get_wa_results(models, dataset, data_names, data_splits, device)
        print_results(dict_results)
        accs.append(dict_results["test_acc"])

        misc.log_write(train_log, 'Uniform: ' + str(dict_results["test_acc"]))

        # Average n-1 models, models-list of model names.
        for i in range(num_models):
            models_copy = copy.copy(models)
            models_copy.pop(i)
            print("models_copy", len(models_copy))
            dict_results = get_wa_results(models_copy, dataset, data_names, data_splits, device)
            print_results(dict_results)
            accs.append(dict_results["test_acc"])
            misc.log_write(train_log, 'w/o '+ str(i) +": " + str(dict_results["test_acc"]))

        print("accs", accs)
        acc_path = os.path.join(inf_args.exp_dir, inf_args.dataset + "_" + str(inf_args.test_env) + "_uniform")
        misc.write_pickle(accs, acc_path)
        # misc.draw_acc_bar_restric(accs, inf_args.exp_dir, inf_args.dataset, inf_args.test_env)

    elif inf_args.weight_selection == "all_in":  # Run twice: 1) Generate figs&accs_independent 2) Merge them

        train_log = os.path.join(inf_args.exp_dir, inf_args.dataset + "_" + str(inf_args.test_env) +"_in.log")
        acc_all = {}

        if not os.path.exists(train_log):
            with open(train_log, "w") as f:
                f.write("log start!\n")

        gradients_path = os.path.join(inf_args.output_dir, "gradients_" + str(inf_args.test_env))

        if os.path.exists(gradients_path):
            gradients = misc.read_pickle(gradients_path)
        else:
            gradients = get_dict_folder_to_score_topo(inf_args, dataset, data_splits, data_names, device)
            misc.write_pickle(gradients, gradients_path)

        # Uniform, Single model, acc
        dict_results = get_wa_results(list(dict_folder_to_score.keys()), dataset, data_names, data_splits, device)
        acc_all['uniform'] = dict_results["train_acc"]
        misc.log_write(train_log, 'Uniform: ' + str(dict_results["train_acc"]))

        # Restricted
        acc_restrict = restricted(dict_folder_to_score, train_log, dataset, data_names, data_splits, device)
        acc_all['restrict'] = acc_restrict['in_acc']

        # Centrality
        if len(gradients) == 10:
            ks = [5]
        else:
            ks = [10]

        cen_names = ["degree"]

        graph_threses = [40]

        acc_centrality = graph_centrality_ablation_in(graph_threses, train_log, gradients, ks, dict_folder_to_score, \
                                            dataset, data_names, data_splits, device, cen_names)

        acc_all['centrality'] = acc_centrality

        acc_path = os.path.join(inf_args.exp_dir, "acc_all_in_" + str(inf_args.test_env))
        misc.write_pickle(acc_all, acc_path)

    elif inf_args.weight_selection == "diversity":  # Run twice: 1) Generate figs&accs_independent 2) Merge them

        train_log = os.path.join(inf_args.exp_dir, inf_args.dataset +"_div.log")

        if not os.path.exists(train_log):
            with open(train_log, "w") as f:
                f.write("log start!\n")

        models = list(dict_folder_to_score.keys())[:10]
        models_predictions = {}
        # Get predictions for each model
        for model in models:
            predictions = get_model_predictions([model], dataset, data_names, data_splits, device)
            models_predictions[model] = predictions

        # select 5 models from 10
        combinations_models = list(combinations(models, 5))
        combinations_accs = {}

        for i, combination in enumerate(combinations_models):
            print(i)
            misc.log_write(train_log, str(i))
            dict_results = get_wa_results(combination, dataset, data_names, data_splits, device)
            test_acc = dict_results["test_acc"]
            combinations_accs[combination] = test_acc
            misc.log_write(train_log, 'test acc: ' + str(test_acc))

        acc_path = os.path.join(inf_args.exp_dir, "div_" + str(inf_args.test_env))
        misc.write_pickle([models_predictions, combinations_accs], acc_path)

    elif inf_args.weight_selection == "efficiency":  # Run twice: 1) Generate figs&accs_independent 2) Merge them

        train_log = os.path.join(inf_args.exp_dir, inf_args.dataset +"_eff.log")

        if not os.path.exists(train_log):
            with open(train_log, "w") as f:
                f.write("log start!\n")

        misc.log_write(train_log, 'Test env: ' + str(inf_args.test_env))

        models = list(dict_folder_to_score.keys())
        models_predictions = {}
        # Get predictions for each model
        start_t = time.time()
        for model in models:
            predictions = get_model_predictions([model], dataset, data_names, data_splits, device)
            models_predictions[model] = predictions
        ens_acc = misc.prediction_avg(models_predictions)
        end_t = time.time()
        misc.log_write(train_log, 'ENS: ' + str(ens_acc) + " " +str((end_t-start_t)/60))

        # Uniform, Single model, acc
        start_t = time.time()
        dict_results = get_wa_results(list(dict_folder_to_score.keys()), dataset, data_names, data_splits, device)
        uniform_acc = dict_results["test_acc"]
        end_t = time.time()
        misc.log_write(train_log, 'Uniform: ' + str(uniform_acc) + " " +str((end_t-start_t)/60))

        # Restricted
        start_t = time.time()
        acc_restrict = restricted(dict_folder_to_score, train_log, dataset, data_names, data_splits, device)
        end_t = time.time()
        restrict_acc = acc_restrict['acc']
        misc.log_write(train_log, 'Restrict: ' + str(restrict_acc) + " " + str((end_t - start_t) / 60))

        # Centrality
        start_t = time.time()
        gradients = get_dict_folder_to_score_topo2(inf_args, dataset, data_splits, data_names, device, 0.1)
        num_models = len(models)
        ks = [num_models // 2]
        cen_names = ["degree"]
        graph_threses = [40]
        acc_centrality = graph_centrality_ablation(graph_threses, train_log, gradients, ks, dict_folder_to_score, \
                                                   dataset, data_names, data_splits, device, cen_names)
        end_t = time.time()
        misc.log_write(train_log, 'Cen: ' + str(acc_centrality[40]["degree"][ks[0]]) + " " + str((end_t - start_t) / 60))

    elif inf_args.weight_selection == "diversity2":  # Run twice: 1) Generate figs&accs_independent 2) Merge them

        train_log = os.path.join(inf_args.exp_dir, inf_args.dataset +"_div2.log")

        if not os.path.exists(train_log):
            with open(train_log, "w") as f:
                f.write("log start!\n")

        models = list(dict_folder_to_score.keys())[:10]
        models_predictions = get_model_predictions3(models, dataset, data_names, data_splits, device)

        acc_path = os.path.join(inf_args.exp_dir, "div2_" + str(inf_args.test_env))
        misc.write_pickle(models_predictions, acc_path)

    elif inf_args.weight_selection == "shapely":  # Run twice: 1) Generate figs&accs_independent 2) Merge them

        train_log = os.path.join(inf_args.exp_dir, inf_args.dataset + "_" + str(inf_args.test_env) +"_shapely.log")

        if not os.path.exists(train_log):
            with open(train_log, "w") as f:
                f.write("log start!\n")

        # Only consider 10 models
        models = list(dict_folder_to_score.keys())[:10]

        # All combinations from the 10 models
        combinations_accs = {}

        for num in range(1, 11):
            combinations_models = list(combinations(models, num))
            for i, combination in enumerate(combinations_models):
                print(i)
                misc.log_write(train_log, str(i))
                dict_results = get_wa_results(combination, dataset, data_names, data_splits, device)
                test_acc = dict_results["test_acc"]
                combinations_accs[combination] = test_acc
                misc.log_write(train_log, 'test acc: ' + str(test_acc))

        acc_path = os.path.join(inf_args.exp_dir, "shapley_" + str(inf_args.test_env))
        misc.write_pickle(combinations_accs, acc_path)

    elif inf_args.weight_selection == "optimization":  # Run twice: 1) Generate figs&accs_independent 2) Merge them

        train_log = os.path.join(inf_args.exp_dir, inf_args.dataset + "_" + str(inf_args.test_env) +"_opt.log")
        if not os.path.exists(train_log):
            with open(train_log, "w") as f:
                f.write("log start!\n")

        gradients_path = os.path.join(inf_args.output_dir, "gradients_" + str(inf_args.test_env))
        if os.path.exists(gradients_path) and 0:
            gradients = misc.read_pickle(gradients_path)
        else:
            gradients = get_dict_folder_to_score_topo(inf_args, dataset, data_splits, data_names, device)
            misc.write_pickle(gradients, gradients_path)

        # Distance matrix
        dis_matrix = cosine_distances(gradients)

        print("dis_matrix", len(dis_matrix))

        # Error matrix
        error_path = os.path.join(inf_args.output_dir, "error_" + str(inf_args.test_env))
        if os.path.exists(error_path):
            error_matrix, N = misc.read_pickle(error_path)
        else:
            error_matrix, N = get_error_matrix(list(dict_folder_to_score.keys()), dataset, data_names, data_splits, device)
            misc.write_pickle([error_matrix, N], error_path)

        error_matrix = matrix_normalization(error_matrix, N)
        # error_matrix = make_psd_eigen(error_matrix)
        # dis_matrix = make_psd_eigen(dis_matrix)
        # dis_matrix = make_psd_diag(dis_matrix)
        print("error_matrix", len(error_matrix))

        k = int(len(error_matrix)/2)
        # C = 1000
        # z = linear_program_reg(error_matrix, dis_matrix, k, lambda_param)
        # z = linear_program_cons(error_matrix, dis_matrix, k, C)

        lambda_params = [0, 0.01, 0.1, 1, 10, 100]
        models = list(dict_folder_to_score.keys())
        models_predictions = {}

        for lambda_param in lambda_params:
            z = linear_program_nonconvex(error_matrix, dis_matrix, k, lambda_param)
            sorted_indices = np.argsort(-z)[:k]
            selected_models = [models[i] for i in sorted_indices]
            dict_results = get_wa_results(selected_models, dataset, data_names, data_splits, device)
            test_acc = dict_results["test_acc"]
            models_predictions[lambda_param] = test_acc
            print("test_acc", test_acc, "lambda_param", lambda_param)
            misc.log_write(train_log, "lambda: " + str(lambda_param) + ' test acc: ' + str(test_acc))

        acc_path = os.path.join(inf_args.exp_dir, "opt_" + str(inf_args.test_env))
        misc.write_pickle(models_predictions, acc_path)

    elif inf_args.weight_selection == "optimization_k":  # Run twice: 1) Generate figs&accs_independent 2) Merge them

        train_log = os.path.join(inf_args.exp_dir, inf_args.dataset + "_" + str(inf_args.test_env) +"_opt_k.log")
        if not os.path.exists(train_log):
            with open(train_log, "w") as f:
                f.write("log start!\n")

        gradients_path = os.path.join(inf_args.output_dir, "gradients_" + str(inf_args.test_env))
        if os.path.exists(gradients_path):
            gradients = misc.read_pickle(gradients_path)
        else:
            gradients = get_dict_folder_to_score_topo(inf_args, dataset, data_splits, data_names, device)
            misc.write_pickle(gradients, gradients_path)

        # Error matrix
        error_path = os.path.join(inf_args.output_dir, "error_" + str(inf_args.test_env))
        if os.path.exists(error_path):
            error_matrix, N = misc.read_pickle(error_path)
        else:
            error_matrix, N = get_error_matrix(list(dict_folder_to_score.keys()), dataset, data_names, data_splits, device)
            misc.write_pickle([error_matrix, N], error_path)

        error_matrix = matrix_normalization(error_matrix, N)
        error_matrix = make_psd_eigen(error_matrix)

        # Distance matrix
        dis_matrix = cosine_similarity(gradients)
        dis_matrix = make_psd_eigen(dis_matrix)

        ks = list(range(1, len(error_matrix), 1))

        lambda_param = 1
        models = list(dict_folder_to_score.keys())
        models_predictions = {}

        for k in ks:
            z = linear_program_reg(error_matrix, dis_matrix, k, lambda_param)
            sorted_indices = np.argsort(-z)[:k]
            selected_models = [models[i] for i in sorted_indices]
            dict_results = get_wa_results(selected_models, dataset, data_names, data_splits, device)
            test_acc = dict_results["test_acc"]
            models_predictions[k] = test_acc
            # print("test_acc", test_acc, "lambda_param", lambda_param)
            misc.log_write(train_log, "k: " + str(k) + ' test acc: ' + str(test_acc))

        acc_path = os.path.join(inf_args.exp_dir, "opt_k_" + str(inf_args.test_env))
        misc.write_pickle(models_predictions, acc_path)

    elif inf_args.weight_selection == "optimization_lambda":  # Run twice: 1) Generate figs&accs_independent 2) Merge them

        train_log = os.path.join(inf_args.exp_dir, inf_args.dataset + "_" + str(inf_args.test_env) +"_opt.log")
        if not os.path.exists(train_log):
            with open(train_log, "w") as f:
                f.write("log start!\n")

        gradients_path = os.path.join(inf_args.output_dir, "gradients_" + str(inf_args.test_env))
        if os.path.exists(gradients_path):
            gradients = misc.read_pickle(gradients_path)
        else:
            gradients = get_dict_folder_to_score_topo(inf_args, dataset, data_splits, data_names, device)
            misc.write_pickle(gradients, gradients_path)

        # Error matrix
        error_path = os.path.join(inf_args.output_dir, "error_" + str(inf_args.test_env))
        if os.path.exists(error_path):
            error_matrix, N = misc.read_pickle(error_path)
        else:
            error_matrix, N = get_error_matrix(list(dict_folder_to_score.keys()), dataset, data_names, data_splits, device)
            misc.write_pickle([error_matrix, N], error_path)

        error_matrix = matrix_normalization(error_matrix, N)

        error_matrix = make_psd_eigen(error_matrix)
        print("error_matrix", len(error_matrix))

        # Distance matrix
        dis_matrix = cosine_similarity(gradients)
        print("dis_matrix", len(dis_matrix))
        dis_matrix = make_psd_eigen(dis_matrix)

        k = int(len(error_matrix)/2)

        # lambda_params = [0, 0.01, 0.1, 1, 10, 100]
        # lambda_params = [0.01, 0.1, 1, 10, 100]
        # lambda_params = [0.0001]
        # lambda_params = [0]

        lambda_params = np.linspace(0, 2, 20)
        lambda_params[0] = 0.00001

        models = list(dict_folder_to_score.keys())
        models_predictions = {}

        for lambda_param in lambda_params:
            # z = linear_program_nonconvex(error_matrix, dis_matrix, k, lambda_param)
            z = linear_program_reg(error_matrix, dis_matrix, k, lambda_param)
            sorted_indices = np.argsort(-z)[:k]

            selected_models = [models[i] for i in sorted_indices]
            dict_results = get_wa_results(selected_models, dataset, data_names, data_splits, device)
            test_acc = dict_results["test_acc"]
            models_predictions[lambda_param] = test_acc
            print("test_acc", test_acc, "lambda_param", lambda_param)
            misc.log_write(train_log, "lambda: " + str(lambda_param) + ' test acc: ' + str(test_acc))

        acc_path = os.path.join(inf_args.exp_dir, "opt_" + str(inf_args.test_env))
        misc.write_pickle(models_predictions, acc_path)

    elif inf_args.weight_selection == "optimization_sdg":  # Run twice: 1) Generate figs&accs_independent 2) Merge them

        train_log = os.path.join(inf_args.exp_dir, inf_args.dataset + "_" + str(inf_args.test_env) +"_opt22.log")
        if not os.path.exists(train_log):
            with open(train_log, "w") as f:
                f.write("log start!\n")

        gradients_path = os.path.join(inf_args.output_dir, "gradients_" + str(inf_args.test_env))
        if os.path.exists(gradients_path):
            gradients = misc.read_pickle(gradients_path)
        else:
            gradients = get_dict_folder_to_score_topo(inf_args, dataset, data_splits, data_names, device)
            misc.write_pickle(gradients, gradients_path)

        # Error matrix
        error_path = os.path.join(inf_args.output_dir, "error_" + str(inf_args.test_env))
        if os.path.exists(error_path):
            error_matrix, N = misc.read_pickle(error_path)
        else:
            error_matrix, N = get_error_matrix(list(dict_folder_to_score.keys()), dataset, data_names, data_splits, device)
            misc.write_pickle([error_matrix, N], error_path)

        error_matrix = matrix_normalization(error_matrix, N)
        error_matrix = make_psd_eigen(error_matrix)

        # Distance matrix
        dis_matrix = cosine_similarity(gradients)
        dis_matrix = make_psd_eigen(dis_matrix)

        k = int(len(error_matrix)/2)

        lambda_params = [1]

        models = list(dict_folder_to_score.keys())
        models_predictions = {}

        for lambda_param in lambda_params:
            z = linear_program_reg(error_matrix, dis_matrix, k, lambda_param)
            print("z")
            print(z)
            sorted_indices = np.argsort(-z)[:k]
            selected_models = [models[i] for i in sorted_indices]
            dict_results = get_wa_results(selected_models, dataset, data_names, data_splits, device)
            test_acc = dict_results["test_acc"]
            models_predictions[lambda_param] = test_acc
            print("test_acc", test_acc, "lambda_param", lambda_param)
            misc.log_write(train_log, "lambda: " + str(lambda_param) + ' test acc: ' + str(test_acc))

        acc_path = os.path.join(inf_args.exp_dir, "opt22_" + str(inf_args.test_env))
        misc.write_pickle(models_predictions, acc_path)

    elif inf_args.weight_selection == "optimization_topology":  # Run twice: 1) Generate figs&accs_independent 2) Merge them

        train_log = os.path.join(inf_args.exp_dir, inf_args.dataset + "_" + str(inf_args.test_env) +"_opt_topo.log")
        if not os.path.exists(train_log):
            with open(train_log, "w") as f:
                f.write("log start!\n")

        nums_data = [0.1 * i for i in range(1, 11)]
        acc_all = {}

        for num_data in nums_data:
            # Error matrix
            error_path = os.path.join(inf_args.output_dir, "error_" + str(inf_args.test_env))
            if os.path.exists(error_path):
                error_matrix, N = misc.read_pickle(error_path)
            else:
                error_matrix, N = get_error_matrix(list(dict_folder_to_score.keys()), dataset, data_names, data_splits, device)
                misc.write_pickle([error_matrix, N], error_path)

            gradients = get_dict_folder_to_score_topo2(inf_args, dataset, data_splits, data_names, device, num_data)

            error_matrix = matrix_normalization(error_matrix, N)
            error_matrix = make_psd_eigen(error_matrix)

            # Distance matrix
            dis_matrix = cosine_similarity(gradients)
            dis_matrix = make_psd_eigen(dis_matrix)

            lambda_param = 1
            k = len(error_matrix)//2

            models = list(dict_folder_to_score.keys())

            z = linear_program_reg(error_matrix, dis_matrix, k, lambda_param)
            sorted_indices = np.argsort(-z)[:k]
            selected_models = [models[i] for i in sorted_indices]
            dict_results = get_wa_results(selected_models, dataset, data_names, data_splits, device)
            test_acc = dict_results["test_acc"]
            acc_all[num_data] = test_acc
            # print("test_acc", test_acc, "lambda_param", lambda_param)
            misc.log_write(train_log, "num_data: " + str(num_data) + ' test acc: ' + str(test_acc))

        acc_path = os.path.join(inf_args.exp_dir, "opt_topo_" + str(inf_args.test_env))
        misc.write_pickle(acc_all, acc_path)

    elif inf_args.weight_selection == "optimization_brutal":  # Run twice: 1) Generate figs&accs_independent 2) Merge them

        def evaluate_subset(G_tilde, W, subset, lambda_param=1):
            # Create z vector for the subset
            z = np.zeros(G_tilde.shape[0])
            z[list(subset)] = 1
            # Calculate the objective function
            objective = z.T @ G_tilde @ z + lambda_param * z.T @ W @ z
            return objective

        def brute_force_search(G_tilde, W, K, lambda_param=1):
            N = G_tilde.shape[0]
            min_objective = 1000
            best_subset = None

            # Iterate over all combinations of size K
            for subset in combinations(range(N), K):
                objective = evaluate_subset(G_tilde, W, subset, lambda_param)
                if objective < min_objective:
                    min_objective = objective
                    best_subset = subset

            return best_subset, min_objective

        train_log = os.path.join(inf_args.exp_dir, inf_args.dataset + "_" + str(inf_args.test_env) + "_opt.log")
        if not os.path.exists(train_log):
            with open(train_log, "w") as f:
                f.write("log start!\n")

        gradients_path = os.path.join(inf_args.output_dir, "gradients_" + str(inf_args.test_env))
        if os.path.exists(gradients_path):
            gradients = misc.read_pickle(gradients_path)
        else:
            gradients = get_dict_folder_to_score_topo(inf_args, dataset, data_splits, data_names, device)
            misc.write_pickle(gradients, gradients_path)

        # Error matrix
        error_path = os.path.join(inf_args.output_dir, "error_" + str(inf_args.test_env))
        if os.path.exists(error_path):
            error_matrix, N = misc.read_pickle(error_path)
        else:
            error_matrix, N = get_error_matrix(list(dict_folder_to_score.keys()), dataset, data_names, data_splits, device)
            misc.write_pickle([error_matrix, N], error_path)

        error_matrix = matrix_normalization(error_matrix, N)
        error_matrix = make_psd_eigen(error_matrix)

        # Distance matrix
        dis_matrix = cosine_similarity(gradients)
        dis_matrix = make_psd_eigen(dis_matrix)

        N = len(error_matrix)/2  # Number of models
        k = int(len(error_matrix)/2)  # Number of models to select
        lambda_param = 1  # Example lambda

        # Perform the brute force search
        best_subset, min_objective = brute_force_search(error_matrix, dis_matrix, k, lambda_param)

        models = list(dict_folder_to_score.keys())
        models_predictions = {}

        selected_models = [models[i] for i in best_subset]
        dict_results = get_wa_results(selected_models, dataset, data_names, data_splits, device)
        test_acc = dict_results["test_acc"]
        models_predictions[lambda_param] = test_acc
        print("test_acc", test_acc, "lambda_param", lambda_param)
        misc.log_write(train_log, "lambda: " + str(lambda_param) + ' test acc: ' + str(test_acc))

        acc_path = os.path.join(inf_args.exp_dir, "brutal_" + str(inf_args.test_env))
        misc.write_pickle(models_predictions, acc_path)

    elif inf_args.weight_selection == "optimization_pareto":  # Run twice: 1) Generate figs&accs_independent 2) Merge them

        train_log = os.path.join(inf_args.exp_dir, inf_args.dataset + "_" + str(inf_args.test_env) + "_pareto.log")
        if not os.path.exists(train_log):
            with open(train_log, "w") as f:
                f.write("log start!\n")

        gradients_path = os.path.join(inf_args.output_dir, "gradients_" + str(inf_args.test_env))
        if os.path.exists(gradients_path):
            gradients = misc.read_pickle(gradients_path)
        else:
            gradients = get_dict_folder_to_score_topo(inf_args, dataset, data_splits, data_names, device)
            misc.write_pickle(gradients, gradients_path)

        # Error matrix
        error_path = os.path.join(inf_args.output_dir, "error_" + str(inf_args.test_env))
        if os.path.exists(error_path):
            error_matrix, N = misc.read_pickle(error_path)
        else:
            error_matrix, N = get_error_matrix(list(dict_folder_to_score.keys()), dataset, data_names, data_splits,
                                               device)
            misc.write_pickle([error_matrix, N], error_path)

        error_matrix = matrix_normalization(error_matrix, N)
        error_matrix = make_psd_eigen(error_matrix)

        # Distance matrix
        dis_matrix = cosine_similarity(gradients)
        dis_matrix = make_psd_eigen(dis_matrix)

        k = int(len(error_matrix) / 2)

        # Define the problem
        problem = EnsemblePruningProblem(error_matrix, dis_matrix, k)

        # Define the algorithm
        # algorithm = NSGA2(pop_size=100,
        #                   sampling=get_sampling("real_random"),
        #                   crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        #                   mutation=get_mutation("real_pm", eta=20),
        #                   eliminate_duplicates=True)

        algorithm = NSGA2(pop_size=50, eliminate_duplicates=True)

        # Perform the Pareto optimization
        res = minimize(problem,
                       algorithm,
                       ("n_gen", 100),
                       verbose=True)

        # # Pareto optimal solutions
        # print("Pareto optimal solutions:")
        # print(res.X)
        #
        # # Corresponding objective values
        # print("Objective values:")
        # print(res.F)

        pareto_set = res.X
        # Select a solution from the Pareto set
        selected_index = 0  # Choose the index of the desired solution
        selected_lambda = pareto_set[selected_index][0]

        models = list(dict_folder_to_score.keys())
        models_predictions = {}

        z = linear_program_reg(error_matrix, dis_matrix, k, selected_lambda)
        sorted_indices = np.argsort(-z)[:k]
        selected_models = [models[i] for i in sorted_indices]
        dict_results = get_wa_results(selected_models, dataset, data_names, data_splits, device)
        test_acc = dict_results["test_acc"]
        models_predictions[selected_lambda] = test_acc
        print("test_acc", test_acc, "lambda_param", selected_lambda)
        misc.log_write(train_log, "lambda: " + str(selected_lambda) + ' test acc: ' + str(test_acc))

        acc_path = os.path.join(inf_args.exp_dir, "pareto_" + str(inf_args.test_env))
        misc.write_pickle(models_predictions, acc_path)

    elif inf_args.weight_selection == "optimization_MI":  # Run twice: 1) Generate figs&accs_independent 2) Merge them

        train_log = os.path.join(inf_args.exp_dir, inf_args.dataset + "_" + str(inf_args.test_env) +"_MI.log")
        if not os.path.exists(train_log):
            with open(train_log, "w") as f:
                f.write("log start!\n")

        pairwise_mis = get_MI_matrix(list(dict_folder_to_score.keys()), dataset, data_names, data_splits, device)

        # pairwise_mis = matrix_normalization(pairwise_mis, len(pairwise_mis))
        # pairwise_mis = make_psd_eigen(pairwise_mis)

        models = list(dict_folder_to_score.keys())
        models_predictions = {}

        print("pairwise_mis", pairwise_mis)

        pareto_solutions = linear_MI(pairwise_mis)

        acc_path = os.path.join(inf_args.exp_dir, "MI_" + str(inf_args.test_env))
        misc.write_pickle([pareto_solutions, pairwise_mis], acc_path)

        # for lambda_param in lambda_params:
        #     z = linear_program_reg(error_matrix, dis_matrix, k, lambda_param)
        #     print("z")
        #     print(z)
        #     sorted_indices = np.argsort(-z)[:k]
        #     selected_models = [models[i] for i in sorted_indices]
        #     dict_results = get_wa_results(selected_models, dataset, data_names, data_splits, device)
        #     test_acc = dict_results["test_acc"]
        #     models_predictions[lambda_param] = test_acc
        #     print("test_acc", test_acc, "lambda_param", lambda_param)
        #     misc.log_write(train_log, "lambda: " + str(lambda_param) + ' test acc: ' + str(test_acc))
        #
        # acc_path = os.path.join(inf_args.exp_dir, "opt22_" + str(inf_args.test_env))
        # misc.write_pickle(models_predictions, acc_path)

    else:
        raise ValueError(inf_args.weight_selection)

if __name__ == "__main__":
    main()