# PointAugment: an Auto-Augmentation Framework for Point Cloud Classification

This repository contains a PyTorch implementation of the paper:

[PointAugment: an Auto-Augmentation Framework for Point Cloud Classification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_PointAugment_An_Auto-Augmentation_Framework_for_Point_Cloud_Classification_CVPR_2020_paper.pdf). 
<br>
[Ruihui Li](https://liruihui.github.io/), 
[Xianzhi Li](https://nini-lxz.github.io/),
[Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/), 
[Chi-Wing Fu](http://www.cse.cuhk.edu.hk/~cwfu/).
<br>
CVPR 2020 (**Oral**)



## Dependencies
* Python 3.6
* CUDA 10.0.
* [PyTorch](http://pytorch.org/). Codes are tested with version 1.2.0
* (Optional) [TensorboardX](https://www.tensorflow.org/) for visualization of the training process. 

Following is the suggested way to install these dependencies: 
```bash
# Create a new conda environment
conda create -n PointAugment python=3.6
conda activate PointAugment

# Install pytorch (please refer to the commend in the official website)
conda install pytorch=1.2.0 torchvision cudatoolkit=10.0 -c pytorch -y
```

### Usage
Download the ModelNet40 dataset from <a href="https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip" target="_blank">here</a>.

To train a model to classify point clouds sampled from 3D shapes:

    python train_PA.py --data_dir ModelNet40_Folder

Log files and network parameters will be saved to `log` folder in default. 

Noted that the code may be not stable, if you can help please contact me.


### Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{li2020pointaugment,
  title={{PointAugment}: An Auto-Augmentation Framework for Point Cloud Classification},
  author={Li, Ruihui and Li, Xianzhi and Heng, Pheng-Ann and Fu, Chi-Wing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={6378--6387},
  year={2020}
}
```
