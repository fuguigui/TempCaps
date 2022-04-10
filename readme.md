# TempCaps: A Capsule Network-based Embedding Model for Temporal Knowledge Graph Completion


## How to Cite Our Work

```
@inproceedings{TempCaps,
    title = "TempCaps: A Capsule Network-based Embedding Model for Temporal Knowledge Graph Completion",
    author = "Fu, Guirong  and
      Meng, Zhao  and
      Han, Zhen  and
      Ding, Zifeng  and
      Ma, Yunpu  and
      Schubert, Matthias and
      Tresp, Volker and
      Wattenhofer, Roger",
    booktitle = "Proceedings of the 6th Workshop on Structured Prediction for NLP (SPNLP@ACL 2022)",
    year = "2022",
    publisher = "Association for Computational Linguistics"
}
```


## Installation

1. create a conda environment:

```bash
conda create -n tkg python=3.6 anaconda
```
2. run
```bash
source activate tkg
pip install -r requirements.txt
```



## How to use?

1. To train the model run the following command. 

```bash
source activate tkg
python main.py \
  --config_path "configs/dyrmlp.yml" \
  --data_dir "data/completion/" \
  --dataset "icews14" \
  --task "completion" \
  --max_time_range 3 \
  --do_test True \
  --batch_size 300 \
  --overwrite True \
  --resume_train False \
  --max_epochs 240 \
  --save_steps 30 \
  --eval_steps 60 \
  --neg_ratio 0 \
  --test_size 256 \
  --from_pretrained False \
  --fix_pretrained False \
  --verbose False \
  --save_eval True \
  --checkpoint "DyRMLP_2305-1921_240_icews14.pth"
```

important parameters:

- config_path: types and basic configurations of model, saved under the folder configs/. For DyRMLP model, use the value "configs/dyrmlp.yml"
- data_dir: the dataset folder for the experiment. Example: do the completion task, use "data/completion". Do the forecasting task, use "data/forecast"
- dataset: the name of the dataset for the experiment. Valid Values are the folders names under the 'data_dir' folder. For example, if want to do completion task on GDELT dataset, you can go to the "data/completion" and find a folder named "gdelt", then this value should be 'gdelt'.
- task: choose from "completion" or "forecasting"
- max_time_range. The default value is 3, corresponding to the Section 3.2.2 in the paper, the predefined time window size.
- do_test: True or False. Whether to do test on the testing dataset after training.
- batch_size: The default value is 16. Recommended value is 300
- overwrite: True or False. Whether to generate a new version of dataset object for the model. If you do some change on the dataset files under data/.../..., take the value as True. Otherwise, take it as False to save time.
- max_epochs: the default value is 30. How many epochs to train.
- neg_ratio: how many negative examples to generate for a positive example. Set it as 0 if you use dyrmlp model.\

- To do the ablation experiments, just change the certain parameter's value in the command line or in the config yaml file.



2. To directly evalute the model run the following command
```bash
source activate tkg
python main.py \
  --config_path "configs/dyrmlp.yml" \
  --data_dir "data/completion/" \
  --task "completion" \
  --dataset "gdelt" \
  --checkpoint "DyRMLP_0206-2144_14_gdelt.pth" \
  --do_train False \
  --do_test True \
  --max_time_range 3 \
  --overwrite False \
  --test_size 256 \
  --save_eval True
```


3. To reproduce the results for ICEWS14, ICEWS05-15, GDELT, run the following command:

```bash
source activate tkg
python main.py \
  --config_path "configs/dyrmlp.yml" \
  --data_dir "data/completion/" \
  --dataset "icews14" \
  --task "completion" \
  --max_time_range 3 \
  --do_test True \
  --batch_size 300 \
  --overwrite False \
  --resume_train False \
  --max_epochs 240 \
  --save_steps 240 \
  --eval_steps 240 \
  --neg_ratio 0 \
  --test_size 256
``` 

```bash
source activate tkg
python main.py \
  --config_path "configs/dyrmlp.yml" \
  --data_dir "data/completion/" \
  --dataset "icews05-15" \
  --task "completion" \
  --max_time_range 3 \
  --do_test True \
  --batch_size 300 \
  --overwrite False \
  --resume_train False \
  --max_epochs 200 \
  --save_steps 200 \
  --eval_steps 200 \
  --neg_ratio 0
``` 

```bash
source activate tkg
python main.py \
  --config_path "configs/dyrmlp.yml" \
  --data_dir "data/completion/" \
  --dataset "gdelt" \
  --task "completion" \
  --max_time_range 3 \
  --do_test True \
  --batch_size 300 \
  --overwrite False \
  --resume_train False \
  --max_epochs 5 \
  --save_steps 5 \
  --eval_steps 5 \
  --neg_ratio 0
```

4. if you want to do experiment using other datasets, save the corresponding dataset files under specific `data_dir` and change the parameter `dataset` to the folder name you have saved. Format requirments of dataset files are in the Section Structures.
## Structures

- cache/ (optional): save the trained models, etc caches
- configs/ (important): save the configurations of models. Each model to run must have a yaml file saved under this folder
- data/ (important): dataset to experiments. Dataset saved as a folder under specific tasks folder. In each dataset folder, there must be three files: stat.txt, test.txt and train.txt.  valid.txt is optional.
    - stat.txt has three elements: the number of enitites, the number of relations and the number of timepoints. Split by tab.
    - train/test/valid.txt follows the format: each line has four elements: subject entity id, relation id, object entity id and timepoint. They are split by tab.
