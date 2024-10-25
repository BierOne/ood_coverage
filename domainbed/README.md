

## Usage
This is the PyTorch implementation of our NAC-ME: https://arxiv.org/abs/2306.02879. Our experimental settings carefully align with Domainbed. We adopt the improved implementation from [https://github.com/khanrc/swad/tree/main](https://github.com/khanrc/swad/tree/main). 

We provide the required packages in [environment.yml](https://github.com/BierOne/ood_coverage/tree/main/environment.yml), you can simply run the following command to create the environment:
```
pip install -r requirements.txt
```

 
## How to run

`train_all.py` script conducts multiple leave-one-out cross-validations for all target domain. Taking the ERM model, PACS dataset, and resent18 arch as an example, you can run the following command:

```sh
CUDA_VISIBLE_DEVICES=1 bash scripts/run_single_train.sh ERM PACS resnet18
```

Experiment results are reported as a table. In the table, the row coverage indicates out-of-domain accuracy from NAC-ME selection. The row coverage_rc indicates the rank correlation between NAC-ME scores and test accuracy.


Example results:
```
+----------------------------+--------------+----------+----------+----------+----------+
|         Selection          | art_painting | cartoon  |  photo   |  sketch  |   Avg.   |
+----------------------------+--------------+----------+----------+----------+----------+
|           oracle           |   71.979%    | 72.750%  | 44.539%  | 47.946%  | 59.303%  |
|            last            |   71.608%    | 70.401%  | 41.674%  | 47.478%  | 57.790%  |
| training-domain validation |   68.475%    | 71.588%  | 40.478%  | 43.381%  | 55.981%  |
|          coverage          |   71.608%    | 70.401%  | 42.776%  | 47.478%  | 58.066%  |
|        coverage_rc         |   69.853%    | 35.049%  | 54.167%  | 45.588%  | 51.164%  |
|           val_rc           |   65.441%    | 43.627%  | 37.500%  | 62.990%  | 52.390%  |
|         oracle_rc          |   94.608%    | 94.853%  | 92.892%  | 96.324%  | 94.669%  |
|            test            |   100.000%   | 100.000% | 100.000% | 100.000% | 100.000% |
+----------------------------+--------------+----------+----------+----------+----------+
```
In the above example, the best model selected by NAC-ME achieves 58.066% average accuracy, which is better than the validation-selected model (55.981%). Note that this is just a simple case of NAC-ME. To achieve the full comparison, it is necessary to sweep a large number of models and datasets in such an unstable DG training scenario. 

NAC-ME calculation is quite similar to the NAC-UE, please refer to the [source file](https://github.com/BierOne/ood_coverage/tree/master/domainbed/benchmark_notes/coverage.py) for more details.


## Credits
This codebase is developed based on [SWAD](https://github.com/khanrc/swad/tree/main). We extend our sincere gratitude for their generosity in providing this valuable resource.

