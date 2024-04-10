# This repo is for KDD submission 2427. 

This repo includes 

1. **Released Codes**

    **1.1 environments and how to run codes**

    **1.2 hyper-parameters and settings of baselines**


 

## 1. Released Codes

### 1.1 environments and how to run codes
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
cd assist-graph/CD
python main_our.py
```
#### Junyi
```
cd junyi-graph/CD
python main_our.py
```
#### Mooc-radar
The MOOC-Radar is too big to upload via Github. 
if you want to run codes for mooc-radar, please first download the dataset from [link](https://cloud.tsinghua.edu.cn/d/5443ee05152344c79419/), and put the files into "mooc-graph/data/coarse/" (Please mkdir first).
Then, you should run
```
cd mooc-graph/CD
python divide_data.py
python main_our.py
```

### 1.2 hyper-parameters and settings of baselines
We search the best learning rate in the range of {0.0001,0.0005, 0.001, 0.005, 0.01} **for all models**. 
For fair comparisons, we set the same embedding dimension to PMF, KaNCD, KSCD, ASG-CD, SCD (128 on ASSIST, 64 on Junyi dataset, 64 on MOOC-Radar dataset).  We set the batch size to 8192 **for all models**. 
We adopt Xavier to init trainable parameters **for all models**. We set a_range of IRT and MIRT to 1. We find that MIRT would achieve better results when the number of embedding dimension is smaller, therefore, the dimension for MIRT is searched in the range of {4,8,16,32} on three datasets. The hidden dimension of Poslinear layers are 256,128 respectively for neural network based CD models, e.g., NCDM, KaNCD, ASG-CD. Both our proposed ASG-CD and KaNCD adopt GMF as the basic matrix factorization technique. 

**RCD** We realize the graph aggregation process by torch sparse for our models and baselines. The original codes for RCD is too time-consuming, and sparse matrix multiplication can improve it and achieve the same operation. 


**KSCD** In a batch, KSCD has students' comprehension degrees (shape: batch size * concept number) and exercise difficulty (shape: batch size * concept number), concept embeddings (shape: concept number * embedding size). By repeating these matrices, KSCD obtains three matrices (batch size * concept number * concept number, batch size * concept number * concept number, batch size * concept number * embedding size).  


We find that matrix with the shape of batch size * concept number * concept number is too large to calculate on **Junyi and MOOC-Radar** dataset. To avoid this, we propose another solution, which can achieve similar performance on other datasets but saves memory. We choose to sum concept embeddings (shape: concept number * embedding size) to a vector (length: embedding size). Then, we repeat it to a matrix (shape: batch size * embedding size). After that, we concatenate two matrices (shape: batch size * embedding size, shape: batch size * concept number). Also, the dimension reduction process is changed to (concept number+embedding size -> concept number). Other operations are the same as KSCD. **We also release this KSCD-variant in junyi-graph/CD/models.py.**

**HAN** This model needs both graph structure and node features. In our paper, each node does not have features; we only consider heterogeneity in the graph structure (i.e., edge heterogeneity). Therefore, we do not consider these models in our initial submission. We agree with the reviewers that analyzing more models is beneficial for our task. To fit our task, we first replace the initial node features&projection with free node embeddings. 
Second, we define four meta paths, student —>(correctly) exercise <—(correctly) student, exercise —>(correctly) exercise <—(correctly) student, student —>(wrongly) exercise <—(wrongly) student, exercise —>(wrongly) exercise <—(wrongly) student. Two paths are used to update student embeddings, while the other two are for exercise embeddings. 
Based on node attention, we can obtain four embeddings for a node. HAN introduces a semantic-level attention to combine these embeddings. Finally, we adopt NCDM-style interaction layer (the dimension of node embedding must be the number of concepts, denoted as HAN-CD) to build connections between combined embeddings to predicted response logs. 

The difference between HAN-CD and our proposed ASG-CD lies in that HAN-CD introduces hierarchical attention and that HAN-CD does not apply adaptive learning. 



