### Exploring Heterogeneity and Uncertainty for Graph-based Cognitive Diagnosis Models in Intelligent Education

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
if you want to run codes for mooc-radar, please visit the [MOOC-Radar repo](https://github.com/THU-KEG/MOOC-Radar), and put the files into "mooc-graph/data/coarse/" (Please mkdir first).
Then, you should run
```
cd mooc-graph/CD
python divide_data.py
python main_our.py --gpu 0
```

You can change the data path in data_loader.py

```
@inproceedings{10.1145/3690624.3709264,
author = {Shao, Pengyang and Yang, Yonghui and Gao, Chen and Chen, Lei and Zhang, Kun and Zhuang, Chenyi and Wu, Le and Li, Yong and Wang, Meng},
title = {Exploring Heterogeneity and Uncertainty for Graph-based Cognitive Diagnosis Models in Intelligent Education},
year = {2025},
publisher = {Association for Computing Machinery},
booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1},
pages = {1233–1243},
numpages = {11}
}
```

