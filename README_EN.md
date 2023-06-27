# GroupIPS

## Environment
- python 3.7+
- pytorch 1.12.1+cu113
- numpy
- tqdm
- scikit-learn

## Dataset

Synthetic Dataset
- According to the merchanism described in the paper, generate the synthetic dataset of X, Y and T respectively. 

- Public dataset Open Pipeline by https://research.zozo.com/data.html

- Generate the synthetic datasets under different number of actions, sample size and outcome function using the bash file in the BashScript directory. 


## How to run
- bash ./bashScript/num_action_Syn.sh
- bash ./bashScript/num_sample_syn.sh
- bash ./bashScript/vary_beta.sh

## Experiments
- Results saved under GroupResult/.