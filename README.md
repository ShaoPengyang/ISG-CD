## This repo is for KDD submission 2427. 

This repo includes (1) codes for results in the original paper; (2)  results based on five-fold cross-validation; (3) modifications about newly added baselines. 

### (1) codes for results in the original paper
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

### (2) results based on five-fold cross-validation 
Note that, we add some important baselines according to reviewers' suggestions, e.g., SCD, HAN, KSCD. 
Considering the time constraints of the rebuttal and the amount of additional experiments, we will release the most important part of the results till April 11 (AOE), with the remaining results gradually provided in the repo until April 18. 

### (3) modifications about newly added baselines
#### HAN
This model needs both graph structure and node features. In our paper, each node does not have features; we only consider heterogeneity in the graph structure (i.e., edge heterogeneity). Therefore, we do not consider these models in our initial submission. 

We agree with the reviewers that analyzing more models is beneficial for our task. To fit our task, we first define four meta paths, student —>(correctly) exercise <—(correctly) student, exercise —>(correctly) exercise <—(correctly) student, student —>(wrongly) exercise <—(wrongly) student, exercise —>(wrongly) exercise <—(wrongly) student. Then, we replace the initial node features&projection with free node embeddings. 
Based on node attention, we can obtain four embeddings for a node. HAN introduces a semantic-level attention to combine these embeddings. Finally, we adopt inner dot (node embedding can be any dimension, denoted as HAN) or NCDM-style interaction layer (the dimension of node embedding must be the number of concepts, denoted as HAN-CD) to build connections between combined embeddings to predicted response logs. 
#### KSCD 
KSCD and KaNCD both adopts the matrix factorization techniques. The only difference between KSCD and KaNCD is the diagnositic layer that maps student/exercise representations to predicted response logs. 
We adopt the original interaction layer in the original KSCD paper on ASSIST ... 
#### SCD
Although it is named after CD, however, SCD does not have the ability to provide students' proficiency levels on each concepts.
It can only be used to predict response logs. Therefore, we do not choose SCD as a baseline in our submission. 



<!--Finally, some codes are borrowed from [source1](https://github.com/HFUT-LEC/EduStudio/blob/68611db64e42bebf33be66fa0126de0269b07f74/edustudio/model/CD), and [source2](https://github.com/dmlc/dgl/blob/master/examples/pytorch/han/model_hetero.py] (https://github.com/bigdata-ustc/EduCDM). -->




