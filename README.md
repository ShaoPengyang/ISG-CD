## This repo is for KDD submission 2427. 

This repo includes (1) codes for results in the original paper; (2)  results based on five-fold cross-validation; (3) discussions about newly added baselines; (4) experiments about whether ASG-CD can detect randomly generated noises. 

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
| **IRT**    | 0.7072  $\pm$ 0.0294   | 0.4421   $\pm$ 0.0212   | 0.7259  $\pm$ 0.0290   | -                     |
| **MIRT**   | 0.7154  $\pm$ 0.0363   | 0.4409   $\pm$ 0.0211   | 0.7483  $\pm$ 0.0198   | -                     |
| **PMF**    | 0.7084  $\pm$ 0.0794   | 0.4260   $\pm$ 0.0353   | 0.7472  $\pm$ 0.0703   | -                     |
| **SCD**    | 0.7212  $\pm$ 0.0566   | 0.4288   $\pm$ 0.0308   | 0.7552  $\pm$ 0.0576   | -                     |
|            |                        |                         |                        |                       |    
| **DINA**   | 0.6253  $\pm$ 0.0245   | 0.4960   $\pm$ 0.0128   | 0.6794  $\pm$ 0.0201   | 0.5579  $\pm$ 0.0316  |
| **NCDM**   | 0.7072  $\pm$ 0.0222   | 0.4460   $\pm$ 0.0173   | 0.7244  $\pm$ 0.0212   | 0.5543 $\pm$ 0.0293   |
| **RCD**    | 0.7103  $\pm$ 0.0164   | 0.4512   $\pm$ 0.0241   | 0.7301  $\pm$ 0.0226   | 0.6221  $\pm$ 0.0228  |
| **KSCD**   | 0.7209  $\pm$ 0.0241   | 0.4331   $\pm$ 0.0177   | 0.7503  $\pm$ 0.0252   | 0.5092  $\pm$ 0.0062  |
| **KaNCD**  | 0.7182  $\pm$ 0.0250   | 0.4423   $\pm$ 0.0195   | 0.7404  $\pm$ 0.0251   | 0.6057  $\pm$ 0.0231  |
| **HAN-CD** | 0.7257  $\pm$ 0.0229   | 0.4297   $\pm$ 0.0151   | 0.7524  $\pm$ 0.0240   | 0.6348  $\pm$ 0.0185  |
| **ASG-CD** | 0.7283  $\pm$ 0.0222   | 0.4280   $\pm$ 0.0150   | 0.7555  $\pm$ 0.0244   | 0.6383  $\pm$ 0.0207  |

### (3) discussions about newly added baselines
#### HAN
This model needs both graph structure and node features. In our paper, each node does not have features; we only consider heterogeneity in the graph structure (i.e., edge heterogeneity). Therefore, we do not consider these models in our initial submission. 

We agree with the reviewers that analyzing more models is beneficial for our task. To fit our task, we first replace the initial node features&projection with free node embeddings. 
Second, we define four meta paths, student —>(correctly) exercise <—(correctly) student, exercise —>(correctly) exercise <—(correctly) student, student —>(wrongly) exercise <—(wrongly) student, exercise —>(wrongly) exercise <—(wrongly) student. Two paths are used to update student embeddings, while the other two are for exercise embeddings. 
Based on node attention, we can obtain four embeddings for a node. HAN introduces a semantic-level attention to combine these embeddings. Finally, we adopt NCDM-style interaction layer (the dimension of node embedding must be the number of concepts, denoted as HAN-CD) to build connections between combined embeddings to predicted response logs. 
#### KSCD 
KSCD and KaNCD both adopts the matrix factorization techniques. The only difference between KSCD and KaNCD is the diagnositic layer that maps student/exercise representations to predicted response logs. 
We adopt the original interaction layer in the original KSCD paper on ASSIST ... 
#### SCD
Although it is named after CD, however, SCD does not have the ability to provide students' proficiency levels on each concepts.
It can only be used to predict response logs. Therefore, we do not choose SCD as a baseline in our submission. 



<!--Finally, some codes are borrowed from [source1](https://github.com/HFUT-LEC/EduStudio/blob/68611db64e42bebf33be66fa0126de0269b07f74/edustudio/model/CD), and [source2](https://github.com/dmlc/dgl/blob/master/examples/pytorch/han/model_hetero.py] (https://github.com/bigdata-ustc/EduCDM). -->




