# GroupIPS 

## 环境依赖
- python 3.7+
- pytorch 1.12.1+cu113
- numpy
- tqdm
- scikit-learn

## 数据集

- Open Bandit Pipeline - https://research.zozo.com/data.html 下载

- 根据产生机制，生成X, T, Y 数据集，在syntheticData.py文件里，具体生成机制可以参考论文里的介绍。

- 根据动作数量、样本数量、结果方程等可产生不同的模拟数据，Bash文件在bashScrip文件夹下

## 代码运行
- bash ./bashScript/num_action_Syn.sh
- bash ./bashScript/num_sample_syn.sh
- bash ./bashScript/vary_beta.sh

## 实验结果
- 实验数据结果将保存在GroupResult文件夹下
