# Ensemble Pruning for Out-of-distribution Generalization

This repository holds the Pytorch implementation of [Ensemble Pruning for Out-of-distribution Generalization](https://openreview.net/pdf?id=eP3vsbB5wW) by Fengchun Qiao and Xi Peng.
If you find our code useful in your research, please consider citing:

```
@inproceedings{qiao2024tep,
title={Ensemble Pruning for Out-of-distribution Generalization},
author={Fengchun Qiao and Xi Peng},
booktitle={International Conference on Machine Learning (ICML)},
year={2024}
}
```

## DomainBed

Our code is adapted from the open-source [DomainBed github](https://github.com/facebookresearch/DomainBed/) and [DiWA github](https://github.com/alexrame/diwa)

### Requirements

* python == 3.7.10
* torch == 1.8.1
* torchvision == 0.9.1
* numpy == 1.20.2

## DiWA Procedure Details

Please follow [DiWA github](https://github.com/alexrame/diwa) to obtain pre-trained individual models

### Average the diverse weights

We average the weights selected by our method

```sh
python -m domainbed.scripts.diwa\
       --data_dir=/my/data/dir/\
       --output_dir=/my/sweep/output/path\
       --dataset TerraIncognita\
       --test_env ${test_env}\
       --weight_selection TEP\
       --trial_seed ${trial_seed}
```

Please contact Fengchun Qiao (fengchun@udel.edu) for any question.
