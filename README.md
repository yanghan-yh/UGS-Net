# Uncertainty-Guided Lung Nodule Segmentation with Feature-Aware Attention
This is the code related to "Uncertainty-Guided Lung Nodule Segmentation with Feature-Aware Attention"-MICCAI 2022.

<div align=center>
<img src="https://github.com/yanghan-yh/UGS-Net/blob/main/network.png" width="700" >
</div>

# 1. Paper information
Paper download: https://arxiv.org/abs/2110.12372

If you find it helpful to your research, please cite as follows:
```
@inproceedings
{yang2022uncertainty,
 title={Uncertainty-Guided Lung Nodule Segmentation with Feature-Aware Attention},
 author={Yang, Han and Shen, Lu and Zhang, Mengke and Wang, Qiuli},
 booktitle={Medical Image Computing and Computer Assisted Intervention--MICCAI 2022: 25th International Conference, Singapore, September 18--22, 2022,    Proceedings, Part V},
 pages={44--54},
 year={2022},
 organization={Springer}
}
```

# 2. Paper Abstract
Since radiologists have different training and clinical experience, they may provide various segmentation maps for a lung nodule. As a result, for a specific lung nodule, some regions have a higher chance of causing segmentation uncertainty, which brings difficulty for lung nodule segmentation with multiple annotations. To address this problem, this paper proposes an Uncertainty-Aware Segmentation Network (UAS-Net) based on multi-branch U-Net, which can learn the valuable visual features from the regions that may cause segmentation uncertainty and contribute to a better segmentation result. Meanwhile, **this network can provide a Multi-Confidence Mask (MCM) simultaneously, pointing out regions with different segmentation uncertainty levels**. We introduce a Feature-Aware Concatenation structure for different learning targets and let each branch have a specific learning preference. Moreover, a joint adversarial learning process is also adopted to help learn discriminative features of complex structures. Experimental results show that our method can predict the reasonable regions with higher uncertainty and improve lung nodule segmentation performance in LIDC-IDRI.

# 3. Environment Setup
* Python 3.7.13
* CUDA 11.1
* Pytorch 1.10.1
* torchvision 0.11.2

The source code of *gcn* is from <https://github.com/jxgu1016/Gabor_CNN_PyTorch>.

# 4. Data Preprocess
In this study, data in the LIDC-IDRI dataset needs to be preprocessed. Firstly, the number of annotations for each nodule was counted, and **the nodules with a number of annotations less than or equal to 2 were cleaned**. Then, each nodule's annotations were done with the intersection and union operation to obtain the intersection and union mask.  Specific operations can be referred to: https://github.com/qiuliwang/LIDC-IDRI-Toolbox-python.

<div align=center>
<img src="https://github.com/yanghan-yh/UGS-Net/blob/main/dif.png" width="800" >
</div>

# 5. Usage
Training UGS-Net：
```
python trainer.py
```
Training comparison method:
```
python trainer_baseline.py
```

# 6. Result

<div align=center>
<img src="https://github.com/yanghan-yh/UGS-Net/blob/main/result.png" width="600" >
</div>
