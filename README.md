# SemiFRL: Semi-Supervised Federated Reinforcement Learning for Clients Selection and Adaptive Pseudo labeling

## Create Environment
To set up your development environment, follow these steps:

1.  **Create the Conda environment:**
    ```bash
    conda env create -f env.yaml
    ```

2.  **Activate the environment:**
    ```bash
    cd src
    conda activate semifrl
    ```

## Examples
 - Train SemiFRL for CIFAR10 dataset (WResNet28x2, $N_\mathcal{S}=250$, fix ( $\tau=0.95$ ) and mix loss, $M=100$, $C=0.1$, Non IID(d=0.3), Epoch=5, global mometum $0.5$, server and client sBN statistics, finetune)
    ``` bash
    python train_classifier_ssfl_RL.py --data_name CIFAR10 --model_name wresnet28x2 --control_name 250_fix@0.95-mix_100_0.1_non-iid-d-0.3_5-5_0.5_1_1
    ```
