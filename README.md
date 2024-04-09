# This repo is for KDD submission 2427. 

This repo includes 

(1) Codes for results in the original paper; 

(2)  Experimental results based on five-fold cross-validation; 

(3) Experiments about whether ASG-CD can detect randomly generated noises & Experiments about whether removing W_1 and W_0;

(4) Discussions about baselines (including newly added ones, e.g., KSCD, SCD, HAN). 

(5) Hyper-parameter settings
 

## (1) Codes for results in the original paper
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

Note that, we realize the graph aggregation process by torch sparse for our models and baselines. The original codes for RCD is too time-consuming, and sparse matrix multiplication can improve it and achieve the same operation. 

## (2) Results based on five-fold cross-validation 
**Considering the time constraints of the rebuttal and the amount of additional experiments, we will release most results till April 11 (AOE), with the remaining results gradually provided in the repo until April 18.** As we re-split data and conduct five-fold cross-validation, the results may be different from previous paper, but the tendency is similar. 

We have categorized all the models into two groups. Models in the first group are unable to provide students' comprehension degrees on concepts and can only predict response logs. Models in the second group can simultaneously accomplish these two tasks. We have highlighted the optimal results in each group.


Five-fold cross-validation on ASSIST
|  **Model**  | **ACC** |  **RMSE** |  **AUC** |  **DOA** |
|------------|---------|----------|----------|----------|
| **IRT**    | 0.7072  $\pm$ 0.0294   | 0.4421   $\pm$ 0.0212   | 0.7259  $\pm$ 0.0290   | -             |
| **MIRT**   | 0.7154  $\pm$ 0.0363   | 0.4409   $\pm$ 0.0211   | 0.7483  $\pm$ 0.0198   | -             |
| **PMF**    | 0.7084  $\pm$ 0.0794   | 0.4310   $\pm$ 0.0353   | 0.7472  $\pm$ 0.0703   | -             |
| **SCD**    | **0.7212  $\pm$ 0.0566**   | **0.4288   $\pm$ 0.0308**   | **0.7552  $\pm$ 0.0576**   | -            |
|            |                      |                       |                      |                    |    
| **DINA**   | 0.6253  $\pm$ 0.0245   | 0.4960   $\pm$ 0.0128   | 0.6794  $\pm$ 0.0201   | 0.5579  $\pm$ 0.0316   |
| **NCDM**   | 0.7072  $\pm$ 0.0222   | 0.4460   $\pm$ 0.0173   | 0.7244  $\pm$ 0.0212   | 0.5543 $\pm$ 0.0293   |
| **RCD**    | 0.7103  $\pm$ 0.0164   | 0.4512   $\pm$ 0.0241   | 0.7301  $\pm$ 0.0226   | 0.6221  $\pm$ 0.0228   |
| **KSCD**   | 0.7209  $\pm$ 0.0241   | 0.4331   $\pm$ 0.0177   | 0.7503  $\pm$ 0.0252   | 0.5092  $\pm$ 0.0062   |
| **KaNCD**  | 0.7182  $\pm$ 0.0250   | 0.4423   $\pm$ 0.0195   | 0.7404  $\pm$ 0.0251   | 0.6057  $\pm$ 0.0231   |
| **HAN-CD** | 0.7257  $\pm$ 0.0229   | 0.4297   $\pm$ 0.0151   | 0.7524  $\pm$ 0.0240   | 0.6348  $\pm$ 0.0185   |
| **ASG-CD**    | **0.7283  $\pm$ 0.0222**   | **0.4280   $\pm$ 0.0150**   | **0.7555  $\pm$ 0.0244**   | **0.6383  $\pm$ 0.0207**   |

Five-fold cross-validation on Junyi
|  **Model**  | **ACC** |  **RMSE** |  **AUC** |  **DOA** |
|------------|---------|----------|----------|----------|
| **IRT**    |  0.7641 $\pm$ 0.0042  |  0.4020 $\pm$ 0.0027  |  0.7997 $\pm$ 0.0059  | -          |
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
| **ASG-CD** |  **0.7647 $\pm$ 0.0047** |  **0.4017 $\pm$ 0.0032**  |  **0.7998 $\pm$ 0.0067**  |  **0.6484 $\pm$ 0.0146** |

Five-fold cross-validation on MOOC-Radar
|  **Model**  | **ACC** |  **RMSE** |  **AUC** |  **DOA** |
|------------|---------|----------|----------|----------|
| **IRT**    |    |    |    | -          |
| **MIRT**   |    |    |    | -          |
| **PMF**    |    |    |    | -          |
| **SCD**    |    |    |    | -          |
|            |    |    |    |            |    
| **DINA**   |    |    |    |            |
| **NCDM**   |    |    |    |            |
| **RCD**    |    |    |    |            |
| **KSCD-variant**   |    |     |     |    |
| **KaNCD**  |    |    |    |            | 
| **HAN-CD** |    |    |    |            |
| **ASG-CD** |    |    |    |            |


## (3) Experiments about whether ASG-CD can detect randomly generated noises & Experiments about whether removing W_1 and W_0

#### Experiments about whether ASG-CD can detect randomly generated noises
We choose the Junyi dataset to conduct this experiment. 
We find that it is not easy to generate random noise on non-interacted student-exercise pairs. These non-interacted pairs consists of potential correct and incorrect response logs, and we do not know whether a correct/incorrect response log is noisy. 
Therefore, we randomly choose some existing student-exercise response logs (corresponding to edges in graph) and modify their labels. We conduct experiments on the modified Junyi dataset, and check out whether these modified logs can be detected by ASG-CD. 

|  **Model**  | **noisy logs/all logs** |  **detections/noisy logs** 
|------------|---------|----------|
| **ASG-CD** |  10%  |    |    
| **ASG-CD** |  20%  |    |    
| **ASG-CD** |  30%  |    |    
| **ASG-CD** |  40%  |    |   
| **ASG-CD** |  50%  |    |  

#### Experiments about whether removing W_1 and W_0
We choose the ASSIST and Junyi datasets to conduct these experiments.

Experiments about whether removing W_1 and W_0 on ASSIST dataset
|    | **ACC** |  **RMSE** |  **AUC** |  **DOA** |
|------------|---------|----------|----------|----------|
| **Not removing**    | **0.7283  $\pm$ 0.0222**   | **0.4280   $\pm$ 0.0150**   | **0.7555  $\pm$ 0.0244**   | **0.6383  $\pm$ 0.0207**   |
| **Removing**   |  0.7224 $\pm$ 0.0268 |  0.4316 $\pm$ 0.0166  |  0.7482 $\pm$ 0.274  |  0.6331 $\pm$ 0.0331 |

Experiments about whether removing W_1 and W_0 on Junyi dataset
|    | **ACC** |  **RMSE** |  **AUC** |  **DOA** |
|------------|---------|----------|----------|----------|
| **Not removing**    |  0.7647 $\pm$ 0.0047 | 0.4017 $\pm$ 0.0032  |  0.7998 $\pm$ 0.0067  |  0.6484 $\pm$ 0.0146 |
| **Removing**   |  **0.7668 $\pm$ 0.0033** | **0.4004 $\pm$ 0.0037**  |  **0.8026 $\pm$ 0.0056**  |  **0.6568 $\pm$ 0.0194** |

Apart from discussions in the rebuttal box, we provide more discussions here. 

First, the reviewers emphasized the removal of W_1 and W_0, which is essentially equivalent to setting both matrices as the identity matrix.
This writing approach has been used in previous literature, e.g., in the light graph-based recommendation model LR-GCCF earlier than lightgcn, the authors annotated W matrix but did not introduce it in the actual implementation as they treated it as an identity matrix. We argue that whether introducing W_1 and W_0 is a specific experimental setup, which are not directly related to the main focus of the paper. 

Second, we observe that the removal of W_1 and W_0 results in different trends on different datasets. 
On the ASSIST dataset, removing these transformation matrices leads to a small decrease in accuracy performance, while on the Junyi dataset, it results in performance improvement. All these changes were relatively small. 

Overall, since the effects of removal are small and these transformation matrices are not related to the main focus of the paper, we do not compare or discuss it in our submission. 
Due to reviewers' requests, we are willing to provide experimental results in above two tables. 


## (4) Discussions about baselines
First of all, results of these newly added baselines are recorded in ``(2) results based on five-fold cross-validation''. 
The following part includes introduction to newly-added baselines, and hyper-parameter settings. 

#### Newly added baselines during the rebuttal process
**1. HAN.**

This model needs both graph structure and node features. In our paper, each node does not have features; we only consider heterogeneity in the graph structure (i.e., edge heterogeneity). Therefore, we do not consider these models in our initial submission. 

We agree with the reviewers that analyzing more models is beneficial for our task. To fit our task, we first replace the initial node features&projection with free node embeddings. 
Second, we define four meta paths, student —>(correctly) exercise <—(correctly) student, exercise —>(correctly) exercise <—(correctly) student, student —>(wrongly) exercise <—(wrongly) student, exercise —>(wrongly) exercise <—(wrongly) student. Two paths are used to update student embeddings, while the other two are for exercise embeddings. 
Based on node attention, we can obtain four embeddings for a node. HAN introduces a semantic-level attention to combine these embeddings. Finally, we adopt NCDM-style interaction layer (the dimension of node embedding must be the number of concepts, denoted as HAN-CD) to build connections between combined embeddings to predicted response logs. 

In fact, HAN-CD is very similar to our ASG-CD. HAN can handle edge and feature heterogeneity at the same time, but there are no features. Consequently, the transfered solution is similar to GCMC used in our paper. 
For example, both them output the final aggregation layer rather than stacking all aggregation layers. The differences lie in that HAN-CD introduces hierarchical attention and that HAN-CD does not apply adaptive learning. 
Therefore, their performance are quite close. 

**2. KSCD.**

KSCD and KaNCD both adopts the matrix factorization techniques. The only difference between KSCD and KaNCD is the diagnositic layer that maps student/exercise representations to predicted response logs. As we already includes KaNCD, we do not include KSCD in our initial submission. 

We agree with reviewers that adding KSCD would be better. During the rebuttal process, we will add it as a baseline. Note that, the interaction layer of KSCD has several steps. First, in a batch, KSCD has students' comprehension degrees (shape: batch size * concept number) and exercise difficulty (shape: batch size * concept number), concept embeddings (shape: concept number * embedding size). 
Second, by repeating these matrices, KSCD obtains three matrices (batch size * concept number * concept number, batch size * concept number * concept number, batch size * concept number * embedding size). Third, KSCD uses torch.cat to concatenate these matrices (batch size * concept number * concept number+embedding size), and combine all these representations.  Finally, KSCD reduces the 2-th dimension from (concept number+embedding size) to 1. 

However, we find that matrix with the shape of batch size * concept number * concept number is too large to calculate on Junyi dataset. To avoid this, we propose another solution, which can achieve similar performance on other datasets but saves memory. We choose to sum concept embeddings (shape: concept number * embedding size) to a vector (length: embedding size). Then, we repeat it to a matrix (shape: batch size * embedding size). After that, we concatenate two matrices (shape: batch size * embedding size, shape: batch size * concept number). Also, the dimension reduction process is changed to (concept number+embedding size -> concept number). Other operations are the same as KSCD. We also release this KSCD-variant in junyi-graph/CD/models.py. 

**3. SCD.**

First, although SCD is named after CD, it does not have the ability to provide students' proficiency levels on each concepts. It can only be used to predict response logs. Second, SCD has similar shortcomings as RCD, as they both do not distinguish edges of correct/wrong response logs. As we have choosen RCD as a baseline, we do not add SCD in our submission. 

We agree with reviewers that adding SCD would be better. During the rebuttal process, we will add it as a baseline. The dimension of SCD's embeddings is the same as PMF, KaNCD, KSCD, ASG-CD ... （these models utilize latent embdeddings to represent students and exercises). In detail, the dimension is 128 on ASSIST, 64 on Junyi dataset, 64 on MOOC-Radar dataset. 

## (5) Hyper-parameter settings
During the rebuttal, we have conducted five-fold cross validation.
We search the best learning rate in the range of {0.0001,0.0005, 0.001, 0.005, 0.01} **for all models**. 
For fair comparisons, we set the same embedding dimension to PMF, KaNCD, KSCD, ASG-CD, SCD (128 on ASSIST, 64 on Junyi dataset, 64 on MOOC-Radar dataset).  We set the batch size to 8192 **for all models**. 
We adopt Xavier to init trainable parameters **for all models**. 

We set a_range of IRT and MIRT to 1. We find that MIRT would achieve better results when the number of embedding dimension is smaller, therefore, the dimension for MIRT is searched in the range of {4,8,16,32} on three datasets. The hidden dimension of Poslinear layers are 256,128 respectively for neural network based CD models, e.g., NCDM, KaNCD, ASG-CD. Both our proposed ASG-CD and KaNCD adopt GMF as the basic matrix factorization technique. 


<!--Finally, some codes are borrowed from [source1](https://github.com/HFUT-LEC/EduStudio/blob/68611db64e42bebf33be66fa0126de0269b07f74/edustudio/model/CD), and [source2](https://github.com/dmlc/dgl/blob/master/examples/pytorch/han/model_hetero.py] (https://github.com/bigdata-ustc/EduCDM). -->