

# Recommendation Editing

The goal of recommendation editing is to quickly correct the erroneous recommendations of the recommendation system, thereby improving the user-friendliness of the recommendation system. For example, in the real-time re-ranking stage of industrial-level multi-stage recommendation systems, the recommendation system needs to perform real-time editing based on the negative feedback provided by users in real time, reducing the occurrence of negative feedback behavior in the recommended results of the next refresh.

<div align="center">
<img src="[img/editing.png](https://github.com/cycl2018/Recommendation-Editing/blob/master/img/editing.png)" border="0" width=400px/>
</div>

## 🚀Quick Start

### Clone
```
git clone git@github.com:cycl2018/Recommendation-Editing.git
```
### Set up the required environment
```
numba==0.58.1
numpy==1.26.3
scipy==1.12.0
torch==2.1.0+cu121
```
### Train the original recommendation model (if you directly use the checkpoint we provide, you can skip it)
- Example of training XSimGCL model by KuaiRand dataset.
- You can refer to the files in the/conf folder for configuration
```
python train.py --conf conf/XSimGCL/KuaiRand.conf
```
### Editing
- Example of editing XSimGCL model by FT method.
- --best_param indicates running with optimal parameters
```
python edit.py --model_conf conf/XSimGCL/KuaiRand.conf --edit_type FT --best_param --edit_num 10
```

## Acknowledgments
We are grateful to the authors of 
[SELFRec](https://github.com/Coder-Yu/SELFRec) 
for making their project codes publicly available.

## Citation
Our paper on this benchmark will be released soon!

<!-- If you use our benchmark in your works, we would appreciate citations to the paper: -->
