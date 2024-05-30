# Shared-Experience-Soft-Actor-Critic-for-Multi-Agent-Reinforcement-Learning

This repository implements baselines based on the paper [_Shared Experience Actor-Critic for Multi-Agent Reinforcement
Learning_](https://arxiv.org/abs/2006.07169) and provides a similar extension to Soft Actor-Critic that we call Shared
Experience Soft Actor-Critic (SESAC).

## Installation

1. Clone the repository

```bash
git clone https://github.com/aportekila/Shared-Experience-Soft-Actor-Critic-for-Multi-Agent-Reinforcement-Learning/tree/episodic
```

2. Install the mandatory dependencies

```bash
pip install -r requirements.txt
```

3. Install the dependencies for desired environments

**Level-based foraging**

```bash
    git clone https://github.com/Oakenmount/lb-foraging
    cd lb-foraging
    pip install -e .
  ```

**Robotic warehouse**

  ```bash
    git clone https://github.com/Oakenmount/robotic-warehouse
    cd lb-foraging
    pip install -e .
  ```

**Multiwalker and Waterworld**

  ```bash  
   pip install 'pettingzoo[sisl]'
   ```

## Running the code

### Training

To train on-policy model like SEAC, run the following command:

```bash
python train.py --env rware-tiny-4ag-easy-v1 --agent_type SEAC --seed 137
```

Off-policy models like SESAC can be trained by using train_off_policy.py.

### Evaluation

To generate learning curves, run the following command:

```bash
python plots.py --env rware-tiny-4ag-easy-v1
```

### Rendering

To view a policy in action, run render-policy.py with env, algo and seed arguments based on saved previous run.

```bash
python render-policy.py --env rware-tiny-4ag-easy-v1 --algo SEAC --seed 137
```