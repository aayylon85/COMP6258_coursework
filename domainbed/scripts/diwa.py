import torch.utils.data
from domainbed.scripts.utils import *
from itertools import combinations
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity, pairwise_distances
import warnings
import numpy as np
from itertools import combinations
# Ignore all warnings
warnings.filterwarnings('ignore')

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

    if inf_args.weight_selection == "TEP":

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

        # Distance matrix
        dis_matrix = cosine_similarity(gradients)
        dis_matrix = make_psd_eigen(dis_matrix)

        k = int(len(error_matrix)/2)

        lambda_param = 1

        models = list(dict_folder_to_score.keys())
        models_predictions = {}

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

    else:
        raise ValueError(inf_args.weight_selection)

if __name__ == "__main__":
    main()