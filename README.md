## This repo is for KDD submission 2427. 

This repo includes (1) codes for results in the original paper; (2)  results based on five-fold cross-validation; (3) discussions about baselines (including newly added ones, e.g., KSCD, SCD, HAN); (4) experiments about whether ASG-CD can detect randomly generated noises. 

### (1) Codes for results in the original paper
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

### (2) Results based on five-fold cross-validation 
Note that, we add some important baselines according to reviewers' suggestions, e.g., SCD, HAN, KSCD. 
Considering the time constraints of the rebuttal and the amount of additional experiments, we will release the most important part of the results till April 11 (AOE), with the remaining results gradually provided in the repo until April 18. 

During the rebuttal process, we randomly split response logs into training, validation, and testing sets with ratio of 7:1:2. Therefore, the results may be different from previous paper, but the tendency is similar. 

Five-fold cross-validation on ASSIST
|    | **ACC** |  **RMSE** |  **AUC** |  **DOA** |
|------------|---------|----------|----------|----------|
| **IRT**    | 0.7072  $\pm$ 0.0294   | 0.4421   $\pm$ 0.0212   | 0.7259  $\pm$ 0.0290   | -              |
| **MIRT**   | 0.7154  $\pm$ 0.0363   | 0.4409   $\pm$ 0.0211   | 0.7483  $\pm$ 0.0198   | -             |
| **PMF**    | 0.7084  $\pm$ 0.0794   | 0.4260   $\pm$ 0.0353   | 0.7472  $\pm$ 0.0703   | -             |
| **SCD**    | 0.7212  $\pm$ 0.0566   | 0.4288   $\pm$ 0.0308   | 0.7552  $\pm$ 0.0576   | -            |
|            |                      |                       |                      |                    |    
| **DINA**   | 0.6253  $\pm$ 0.0245   | 0.4960   $\pm$ 0.0128   | 0.6794  $\pm$ 0.0201   | 0.5579  $\pm$ 0.0316   |
| **NCDM**   | 0.7072  $\pm$ 0.0222   | 0.4460   $\pm$ 0.0173   | 0.7244  $\pm$ 0.0212   | 0.5543 $\pm$ 0.0293   |
| **RCD**    | 0.7103  $\pm$ 0.0164   | 0.4512   $\pm$ 0.0241   | 0.7301  $\pm$ 0.0226   | 0.6221  $\pm$ 0.0228   |
| **KSCD**   | 0.7209  $\pm$ 0.0241   | 0.4331   $\pm$ 0.0177   | 0.7503  $\pm$ 0.0252   | 0.5092  $\pm$ 0.0062   |
| **KaNCD**  | 0.7182  $\pm$ 0.0250   | 0.4423   $\pm$ 0.0195   | 0.7404  $\pm$ 0.0251   | 0.6057  $\pm$ 0.0231   |
| **HAN-CD** | 0.7257  $\pm$ 0.0229   | 0.4297   $\pm$ 0.0151   | 0.7524  $\pm$ 0.0240   | 0.6348  $\pm$ 0.0185   |
| **ASG-CD**    | 0.7283  $\pm$ 0.0222   | 0.4280   $\pm$ 0.0150   | 0.7555  $\pm$ 0.0244   | 0.6383  $\pm$ 0.0207   |

Five-fold cross-validation on Junyi
|    | **ACC** |  **RMSE** |  **AUC** |  **DOA** |
|------------|---------|----------|----------|----------|
| **IRT**    |    |    |    | -          |
| **MIRT**   |    |    |    | -          |
| **PMF**    |    |    |    | -          |
| **SCD**    |  0.7576 $\pm$ 0.0056  |  0.4084 $\pm$ 0.0036  |  0.7902 $\pm$ 0.0040  | -          |
|            |                       |                       |                       |            |    
| **DINA**   |    |    |    |   |
| **NCDM**   |  0.7482 $\pm$ 0.0013  |  0.4141 $\pm$ 0.0016  |  0.7816 $\pm$ 0.0013  | 0.4996 $\pm$ 0.0050 |
| **RCD**    |    |    |    |   |
| **KSCD-variant**   |  0.7561 $\pm$ 0.0029  |  0.4079 $\pm$ 0.0002   |  0.7909 $\pm$ 0.0053   | 0.5001 $\pm$ 0.0039   |
| **KaNCD**  |  0.7536 $\pm$ 0.0020 |  0.4096 $\pm$ 0.0012  |  0.7867 $\pm$ 0.0017  |  0.5529 $\pm$ 0.0212 |  
| **HAN-CD** |  0.7626 $\pm$ 0.0039 |  0.4031 $\pm$ 0.0031  |  0.7957 $\pm$ 0.0080  |  0.6469 $\pm$ 0.0132 |
| **ASG-CD** |  0.7647 $\pm$ 0.0047 |  0.4017 $\pm$ 0.0032  |  0.7998 $\pm$ 0.0067  |  0.6484 $\pm$ 0.0146 |

### (3) Discussions about baselines
First of all, results of these newly added baselines are recorded in ``(2) results based on five-fold cross-validation''. 

#### Newly added baselines during the rebuttal process
**1. HAN.**

This model needs both graph structure and node features. In our paper, each node does not have features; we only consider heterogeneity in the graph structure (i.e., edge heterogeneity). Therefore, we do not consider these models in our initial submission. 

We agree with the reviewers that analyzing more models is beneficial for our task. To fit our task, we first replace the initial node features&projection with free node embeddings. 
Second, we define four meta paths, student —>(correctly) exercise <—(correctly) student, exercise —>(correctly) exercise <—(correctly) student, student —>(wrongly) exercise <—(wrongly) student, exercise —>(wrongly) exercise <—(wrongly) student. Two paths are used to update student embeddings, while the other two are for exercise embeddings. 
Based on node attention, we can obtain four embeddings for a node. HAN introduces a semantic-level attention to combine these embeddings. Finally, we adopt NCDM-style interaction layer (the dimension of node embedding must be the number of concepts, denoted as HAN-CD) to build connections between combined embeddings to predicted response logs. 

**2. KSCD.**

KSCD and KaNCD both adopts the matrix factorization techniques. The only difference between KSCD and KaNCD is the diagnositic layer that maps student/exercise representations to predicted response logs. Therefore, we do not include them in our submission. 

We agree with reviewers that adding KSCD would be better. During the rebuttal process, we will add it as a baseline. 

The interaction layer of KSCD has several steps. First, in a batch, KSCD has students' comprehension degrees (shape: batch size * concept number) and exercise difficulty (shape: batch size * concept number), concept embeddings (shape: concept number * embedding size). 
Second, by repeating these matrices, KSCD obtains three matrices (batch size * concept number * concept number, batch size * concept number * concept number, batch size * concept number * embedding size). Third, KSCD uses torch.cat to concatenate these matrices (batch size * concept number * concept number+embedding size), and combine all these representations.  Finally, KSCD reduces the 2-th dimension from (concept number+embedding size) to 1. 

However, we find that matrix with the shape of batch size * concept number * concept number is too large to calculate on Junyi dataset. To avoid this, we propose another solution, which can achieve similar performance on other datasets but saves memory. We choose to sum concept embeddings (shape: concept number * embedding size) to a vector (length: embedding size). Then, we repeat it to a matrix (shape: batch size * embedding size). After that, we concatenate two matrices (shape: batch size * embedding size, shape: batch size * concept number). Also, the dimension reduction process is changed to (concept number+embedding size -> concept number). Other operations are the same as KSCD. 

We also release this KSCD-variant in junyi-graph/CD/models.py. 

**3. SCD.**

First, although SCD is named after CD, it does not have the ability to provide students' proficiency levels on each concepts. It can only be used to predict response logs. Second, SCD has similar shortcomings as RCD, as they both do not distinguish edges of correct/wrong response logs. As we have choosen RCD as a baseline, we do not add SCD in our submission. 

We agree with reviewers that adding SCD would be better. During the rebuttal process, we will add it as a baseline. The dimension of SCD's embeddings is the same as PMF, KaNCD, KSCD, ASG-CD ... （these models utilize latent embdeddings to represent students and exercises). In detail, the dimension is 128 on ASSIST, 64 on Junyi dataset, 64 on MOOC-Radar dataset. 

#### Hyper-parameter settings for baselines. 
During the rebuttal, we have conducted five-fold cross validation.
For fair comparisons, we set the same embedding dimension to PMF, KaNCD, KSCD, ASG-CD, SCD (128 on ASSIST, 64 on Junyi dataset, 64 on MOOC-Radar dataset).  
We set the batch size to 8192 for all models on all datasets. 
We find that MIRT would achieve better results when the number of embedding dimension is smaller, therefore, the dimension for MIRT is searched in the range of {4,8,16,32} on three datasets. 



### (4) Experiments about whether ASG-CD can detect randomly generated noises



<!--Finally, some codes are borrowed from [source1](https://github.com/HFUT-LEC/EduStudio/blob/68611db64e42bebf33be66fa0126de0269b07f74/edustudio/model/CD), and [source2](https://github.com/dmlc/dgl/blob/master/examples/pytorch/han/model_hetero.py] (https://github.com/bigdata-ustc/EduCDM). -->




