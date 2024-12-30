# This repo is for ``Exploring Heterogeneity and Uncertainty for Graph-based Cognitive Diagnosis Models in Intelligent Education''

Our environment:
```
Python 3.9.7 
torch 2.0.1
pandas 1.3.4
scikit-learn 0.24.2
networkx 2.6.3
```



Run the codes: 
#### ASSIST
```
cd assist-graph/CD
python main_our.py --gpu 0
```
#### Junyi
```
cd junyi-graph/CD
python main_our.py --gpu 0
```
#### Mooc-radar
The MOOC-Radar is too big to upload via Github. 
if you want to run codes for mooc-radar, please first download the dataset from [link](https://cloud.tsinghua.edu.cn/d/5443ee05152344c79419/), and put the files into "mooc-graph/data/coarse/" (Please mkdir first).
Then, you should run
```
cd mooc-graph/CD
python divide_data.py
python main_our.py --gpu 0
```

You can change the data path in data_loader.py
