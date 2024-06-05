<img src="https://img.shields.io/badge/Python-3766AB?style=flat-square&logo=Python&logoColor=white"/></a>
<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=OpenCV&logoColor=white"/></a>

# Real-Time-Object-Synthesis-to-Face

**Machine Vision** Final Project: 06/2024

## Overview
<img src="./images/overall_architecture.png"></center>

## Prerequisites

- Python 3.7.16
- opencv-python 4.9.0.80
- numpy 1.21.6
- matplotlib 3.5.3

Please check *requirements.txt* for other packages.


## Training

```
python -m train_utils.train --data_root=<> --configs=<> --batch_size=2 --save_path=<> --epoch=<> --workers=4 --snapshot --rerank
```
### snapshot
**(notice) <u>Argument 'snapshot' will save the current folder. Thus, the save path must not include the current path. </u>**

### epoch
In each dataset, we used the below epoch values.
- Celeb-reID: 15
- Celeb-reID-light: 60
- LTCC: 100
- VC-Clothes: 80

### configs
Config files are provided in *configs* folder. 

### Dataset
Download each dataset before running code.
- [Celeb-reID & Celeb-reID-light](https://github.com/Huang-3/Celeb-reID)
- [LTCC](https://naiq.github.io/LTCC_Perosn_ReID.html)
- [VC-Clothes](https://wanfb.github.io/dataset.html)

If you use Celeb-reID dataset or Celeb-reID-light dataset, just set '--data_root' as dataset root, 
**however, if you want to use other dataset, you need to change dataset form as Celeb-reID dataset form.**

You can use provided files in *change_form*.

## Evaluating
```
python -m evaluate.evaluate --data_root=<> --configs=<> --batch_size=2 --save_path=<> --workers=4 --model=<> --rerank
``` 

### Dataset
If you use Celeb-reID dataset, Celeb-reID-light dataset or VC-Clothes dataset, just run the *evaluate.py* file in evaluate, 
**however if you want to use LTCC datasets, you need to modify [Clothes_Change_Person_ReID](https://github.com/xiangzhouzhang/Clothes_Change_Person_ReID) to get proper results.**

### Results (You can download pre-trained models [here](https://o365seoultech-my.sharepoint.com/:f:/g/personal/20512067_officestu_seoultech_ac_kr/Ene_Gnt3aktOumAr16_8ixABeo1rPUox98gs-fC7oLEHXA?e=yo73xd).)
<img src="./images/comparison.png"></center>


## License
Our code and the *models/AdaINGenerator.py* is under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (**Attribution-NonCommercial-ShareAlike 4.0 International**). You can check [here](https://github.com/NVlabs/DG-Net) for models/AdaINGenerator.py.
 
