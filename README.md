### This repo is for KDD submission 2427. 
To run all codes, Pytorch (gpu version), networkx, pandas, scikit-learn must be installed. 

### Assist
```
cd assist-graph/RCD
python main_our.py
```

### Junyi
```
cd junyi-graph/RCD
python main_our.py
```

### Mooc-radar
The MOOC-Radar is too big to upload via Github. 
if you want to run codes for mooc-radar, please first download the dataset from [link](https://cloud.tsinghua.edu.cn/d/5443ee05152344c79419/), and put the files into "data/coarse/".
Then, you should run
```
cd mooc-graph/RCD
python divide_data.py
python main_our.py
```
