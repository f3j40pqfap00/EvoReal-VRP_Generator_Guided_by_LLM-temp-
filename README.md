# Bridging Synthetic and Real Routing Problems via LLM-Guided Instance Generation and Progressive Adaptation
This temporary repository contains code for an efficient progessive fine-tuning approach to enhance the generalization of Neural Combinatorial Solvers on the real-world benchmark dataset for TSP (Traveling Salesman Problem) and CVRP (Capacitated Vehicle Routing Problem), namely TSPLib (Reinelt, 1991) and CVRPLib (Uchoa et al. 2017).\


## Our Repository
Due to the upload limitation, we store our codes in a temporary Github repository: https://github.com/f3j40pqfap00/EvoReal-VRP_Generator_Guided_by_LLM-temp-/tree/main
We encourage the interested readers to view and test our codes. 

## Dependencies

### Dependent python packages for finetuning LEHD and POMO  
```bash
This code is ran on a machine with Python version 3.12.4 .
numpy==2.3.0
psutil==7.0.0             
torch==2.5.1
vrplib==1.0.0
tqdm==4.67.1
pytz==2025.2
matplotlib==3.10.3
joblib==1.5.1
```

If any package is missing, just install it following the prompts.

## Implementation

This project's structure is clear, the codes are based on .py files, and they should be easy to read, understand, and run.


## Basic Usage

### Fine-tuning Models

#### For reproducing our model with progressive fine-tuning in phase one
To fine-tune pre-trained models in phase one, i.e., LEHD-EvoReal and POMO-EvoReal, firstly download the official checkpoints from POMO/LEHD's github.
For POMO: https://github.com/yd-kwon/POMO/tree/master/NEW_py_ver  (checkpoint-2000.pt for TSP, and checkpoint-30500.pt for CVRP)
For LEHD: https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD/TSP (checkpoint-150.pt for TSP, and checkpoint-40.pt for CVRP)

Once downloaded, please run *train_p1.py* in each sub-folders for TSP and CVRP:
```bash
# For POMO
python /POMO-EvoReal/TSP/POMO_TSP_p1/train_p1.py
python /POMO-EvoReal/CVRP/POMO_CVRP_p1/train_p1.py

# For LEHD
python /LEHD-EvoReal/TSP/LEHD_TSP_p1/TSP/train_p1.py
python /LEHD-EvoReal/CVRP/LEHD_CVRP_p1/CVRP/train_p1.py
```
#### For reproducing our model with progressive fine-tuning in phase two (the final model used for inference)
No need to download checkpoints, saved checkpoints trained from phase one finetuning are in the folder ./result/p1-checkpoints of each problem.
Please run *train_p2.py* in each sub-folders for TSP and CVRP:
```bash
# For POMO
python POMO-EvoReal/TSP/POMO_TSP_p2/train_p2.py
python POMO-EvoReal/CVRP/POMO_CVRP_p2/train_p2.py

# For LEHD
python LEHD-EvoReal/TSP/LEHD_TSP_p2/TSP/train_p2.py
python LEHD-EvoReal/CVRP/LEHD_CVRP_p2/CVRP/train_p2.py
```
Modify parameters in the script as needed. Ablation studies in table 3 and table 4 can be reproduced by switching the corresponding training phase in train_xxx.py. No separate code is required beyond the provided training scripts. All checkpoints (trained from phase one+two) and training logs and statistics can be found be in the /EXP_checkpoints folder

### Inference


To evaluate trained models and reproduce main table results,  run *test_tsplib.py* and *test_cvrplib.py* in each sub-folders TSP and CVRP:
```bash
# For TSP
python POMO-EvoReal/TSP/test_tsplib.py
python POMO-EvoReal/CVRP/test_cvrplib.py

# For CVRP
python LEHD-EvoReal/TSP/test_tsplib.py
python LEHD-EvoReal/CVRP/test_cvrplib.py
```
All evaluation results and statistics will be saved in JSON format.

## Project Structure

```

LEHD-EvoReal/
    CVRP/
    TSP/

POMO-EvoReal/
    TSP/
    CVRP/
EXP_checkpoints/
    LEHD_CVRP_EvoReal_checkpoints/    # LEHD-CVRP model checkpoints and training logs
    LEHD_TSP_EvoReal_checkpoints/     # LEHD-TSP model checkpoints and training logs
    POMO_CVRP_EvoReal_checkpoints/    # POMO-CVRP model checkpoints and training logs
    POMO_TSP_EvoReal_checkpoints/     # POMO-TSP model checkpoints and training logs
    test_logs/                        # All inference logs and statistics
requirements.txt
README.md
```


## Dataset 
**Important** 

One of the training dataset cannot be uploaded to github due to the size limit. Please download the dataset from: https://drive.google.com/drive/folders/1xlkZ_EmkC8YLE8OqSRmvQXSR13qxTz0P?usp=sharing and place the file in this folder: /LEHD-EvoReal/TSP/LEHD_TSP_p1/TSP/data
