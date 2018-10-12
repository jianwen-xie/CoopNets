# Cooperative Training of Descriptor and Generator Networks

This repository contains a tensorflow implementation for the paper "[Cooperative Training of Descriptor and Generator Networks](http://www.stat.ucla.edu/~jxie/CoopNets/CoopNets_files/doc/CoopNets_PAMI.pdf)".
(http://www.stat.ucla.edu/~jxie/CoopNets/CoopNets.html)

## Requirements
- Python 2.7 or Python 3.3+
- [Tensorflow r1.0+](https://www.tensorflow.org/install/)
- [Scipy](https://www.scipy.org/install.html)
- [pillow](https://pillow.readthedocs.io/en/latest/installation.html)

## Usage

(1) For scene synthesis

First, put your scene data folder to `./data` directory:

    $ python download.py scene

To train a model with ***alp*** dataset:

    $ python main.py --category alp --data_dir ./data/scene --output_dir ./output 

synthesized results will be saved in `./output/alp/synthesis`. 

If you want to calculate inception score, use --calculate_inception=True. If category is mnist, data will be downloaded and parzen window-based log-likelihood is calculated automatically. 

To test generator by synthesizing interpolation results with trained model:

    $ python main.py --test --sample_size 144 --category alp --output_dir ./output --ckpt ./output/alp/checkpoints/model.ckpt
testing results will be saved in `./output/alp/test`

## Results
### Results of [MIT Place205](http://places.csail.mit.edu) dataset
**Descriptor result**
![descriptor](assets/descriptor.png)

**Generator result**
![generator](assets/generator.png)

**Interpolation result**
![interpolation](assets/interpolation.png)


## Reference
    @article{coopnets,
        author = {Xie, Jianwen and Lu, Yang and Gao, Ruiqi and Zhu, Song-Chun and Wu, Ying Nian},
        title = {Cooperative Training of Descriptor and Generator Networks},
        journal={IEEE transactions on pattern analysis and machine intelligence (PAMI)},
        year = {2018},
        publisher={IEEE}
    }
    
For any questions, please contact Jianwen Xie (jianwen@ucla.edu), Ruiqi Gao (ruiqigao@ucla.edu) and Zilong Zheng (zilongzheng0318@ucla.edu)
