# MutualReg-train
Source code for our ICASSP2024 paper [MutualReg: Mutual Learning for Unsupervised Medical Image Registration](https://ieeexplore.ieee.org/document/10445904)

# Data Preparation
1. Download the Abdomen CT-CT dataset of the [Learn2Reg challenge](https://learn2reg.grand-challenge.org/Datasets/).
2. Modify the variable path in line 8 of `data_utils.py` such that it points to the root directory of the data.

# Training
Original [reg-cyclical-self-train](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_64): Execute `python main.py --phase train --out_dir PATH/TO/OUTDIR --gpu GPU --num_warps 2 --ice true --reg_fac 1. --augment true --adam true --sampling true`.

MutualReg:
1. Execute `python main_multKD_teach`
2. Execute `python main_multKD_student`
3. Execute `python main_multKD_t2s_finetune`
4. Execute `python main_multKD_t2s2t_finetune`
5. Execute `python main_multKD_t2s2t2s_finetune`

# Testing
In l. 9 of `test.py`, set the path to the model weights you want to use for testing (for example [reg-cyclical-self-train](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_64) `final_model.pth`). Subsequently, execute `python main.py --phase test --gpu GPU`

## Citation
If you find our code useful for your work, please cite the following paper
```latex
@INPROCEEDINGS{10445904,
  author={Liu, Jun and Wang, Wenyi and Shen, Nuo and Wang, Wei and Wang, Kuanquan and Li, Qince and Yuan, Yongfeng and Zhang, Henggui and Luo, Gongning},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Mutualreg: Mutual Learning for Unsupervised Medical Image Registration}, 
  year={2024},
  pages={1561-1565},
  doi={10.1109/ICASSP48485.2024.10445904}
  }
```

## Acknowledgements
Our code heavily inherits from [reg-cyclical-self-train](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_64).
We appreciate them making their code open-source.