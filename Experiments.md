# This repo is for KDD submission 2427. 

This markdown file includes 

1. **Experimental results**

    **1.1 five-fold cross-validation (including added baselines)**

    **1.2 removing W_1 and W_0**

    **1.3 detecting randomly generated noises**


## 1. Experimental results
**Considering the time constraints of the rebuttal and the amount of additional experiments, we will release most results till April 11 (AOE), with the remaining results gradually provided in the repo until April 18.**

### 1.1 five-fold cross-validation (including added baselines)

As we re-split data and conduct five-fold cross-validation, the results may be different from previous paper, but the tendency is similar. Specifically, we split the whole dataset into five folds. We take one fold as testing set in turn, and split the remaining four sets as training&validation sets with ratio of 7:1. 

We have categorized all the models into two groups. Models in the first group are unable to provide students' comprehension degrees on concepts and can only predict response logs. Models in the second group can simultaneously accomplish these two tasks. We have highlighted the optimal results in each group.


**Table 1 Five-fold cross-validation on ASSIST**
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

**Table 2 Five-fold cross-validation on Junyi**
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

**Table 3 Five-fold cross-validation on MOOC-Radar**
|  **Model**  | **ACC** |  **RMSE** |  **AUC** |  **DOA** |
|------------|---------|----------|----------|----------|
| **IRT**    |    |    |    | -          |
| **MIRT**   |    |    |    | -          |
| **PMF**    |    |    |    | -          |
| **SCD**    |  0.8597 $\pm$ 0.0002 |  0.3160 $\pm$ 0.0004  |  0.8769 $\pm$ 0.0006  |  - |
|            |    |    |    |            |    
| **DINA**   |    |    |    |            |
| **NCDM**   |  0.8537 $\pm$ 0.0003 |  0.3241 $\pm$ 0.0004  |  0.8663 $\pm$ 0.0006  |  0.6055 $\pm$ 0.0047 |
| **RCD**    |    |    |    |            |
| **KSCD-variant**   |  0.8627 $\pm$ 0.0010 |  0.3123 $\pm$ 0.0004  |  0.8800 $\pm$ 0.0008  |  0.5027 $\pm$ 0.0039 |
| **KaNCD**  |  0.8636 $\pm$ 0.0013 |  0.3112 $\pm$ 0.0004  |  0.8812 $\pm$ 0.0009  |  0.6980 $\pm$ 0.0065 |
| **HAN-CD** |  0.8650 $\pm$ 0.0007 |  0.3103 $\pm$ 0.0005  |  0.8824 $\pm$ 0.0006  |  0.7231 $\pm$ 0.0031 |
| **ASG-CD** |  **0.8666 $\pm$ 0.0002** |  **0.3087 $\pm$ 0.0004**  |  **0.8881 $\pm$ 0.0008**  |  **0.7401 $\pm$ 0.0037** |


### 1.2 removing W_1 and W_0
We choose the ASSIST and Junyi datasets to conduct these experiments.

**Table 4 Experiments about whether removing W_1 and W_0 on ASSIST dataset**
|    | **ACC** |  **RMSE** |  **AUC** |  **DOA** |
|------------|---------|----------|----------|----------|
| **Not removing**    | **0.7283  $\pm$ 0.0222**   | **0.4280   $\pm$ 0.0150**   | **0.7555  $\pm$ 0.0244**   | **0.6383  $\pm$ 0.0207**   |
| **Removing**   |  0.7224 $\pm$ 0.0268 |  0.4316 $\pm$ 0.0166  |  0.7482 $\pm$ 0.274  |  0.6331 $\pm$ 0.0331 |


**Table 5 Experiments about whether removing W_1 and W_0 on Junyi dataset**
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


### 1.3 detecting randomly generated noises

We choose the Junyi dataset to conduct this experiment. 
We find that it is not easy to generate random noise on non-interacted student-exercise pairs. These non-interacted pairs consists of potential correct and incorrect response logs, and we do not know whether a correct/incorrect response log is noisy. 
Therefore, we randomly choose some existing student-exercise response logs (corresponding to edges in graph) and modify their labels. We conduct experiments on the modified Junyi dataset, and check out whether these modified logs can be detected by ASG-CD. 

**Table 6 Detections about random generated noises on Junyi**
|  **Model**  | **noisy logs/all logs** |  **detections/noisy logs** 
|------------|---------|----------|
| **ASG-CD** |  5%  |  90.2%  | 
| **ASG-CD** |  10%  |  85.7%  |    
| **ASG-CD** |  20%  |  79.4%  |    
| **ASG-CD** |  30%  |  65.5%  |      

We have observed that when artificially adding noise edges at a rate of 5%, our model can relatively accurately distinguish the noise. However, as the number of noise edges increases, adaptive learning becomes less effective in accurately identifying the noise. 

The observed trend can be attributed to the approach we used to generate noise. In our design, we created noise by randomly selecting existing response logs and flipping their labels. As the quantity of noise increases, the data distribution of the main data gradually shifting towards the noise. In adaptive learning, the principle of identifying noise relies on detecting response logs that deviate significantly from the distribution of the main data, which means that we cannot effectively detect noise when the distribution of the main data has changed to noise.  


<!--Finally, some codes are borrowed from [source1](https://github.com/HFUT-LEC/EduStudio/blob/68611db64e42bebf33be66fa0126de0269b07f74/edustudio/model/CD), and [source2](https://github.com/dmlc/dgl/blob/master/examples/pytorch/han/model_hetero.py] (https://github.com/bigdata-ustc/EduCDM). -->