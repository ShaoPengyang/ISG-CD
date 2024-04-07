## This repo is for KDD submission 2427. 

### 1. Reproducing experimental results in the original paper. 
To run all codes, Pytorch (gpu version), networkx, pandas, scikit-learn must be installed. 
Our environment:
```
Python 3.9.7 
torch 2.0.1
pandas 1.3.4
scikit-learn 0.24.2
networkx 2.6.3
```
#### Assist
```
cd assist-graph/RCD
python main_our.py
```
#### Junyi
```
cd junyi-graph/RCD
python main_our.py
```
#### Mooc-radar
The MOOC-Radar is too big to upload via Github. 
if you want to run codes for mooc-radar, please first download the dataset from [link](https://cloud.tsinghua.edu.cn/d/5443ee05152344c79419/), and put the files into "mooc-graph/data/coarse/" (Please mkdir first).
Then, you should run
```
cd mooc-graph/RCD
python divide_data.py
python main_our.py
```

### 2. Results based on k-fold cross-validation. 
Note that, we add some important baselines according to reviewers' suggestions, e.g., SCD, HAN, KSCD. 
Considering the time constraints of the rebuttal and the amount of additional experiments, we will release the most important part of the results till April 11 (AOE), with the remaining results gradually provided in the repo until April 18. 



<!--Finally, some codes are borrowed from [source1](https://github.com/HFUT-LEC/EduStudio/blob/68611db64e42bebf33be66fa0126de0269b07f74/edustudio/model/CD), and [source2](https://github.com/dmlc/dgl/blob/master/examples/pytorch/han/model_hetero.py] (https://github.com/bigdata-ustc/EduCDM). -->




