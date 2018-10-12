# Cooperative Training of Descriptor and Generator Networks

This repository contains a tensorflow implementation for the paper "[Cooperative Training of Descriptor and Generator Networks](http://www.stat.ucla.edu/~jxie/CoopNets/CoopNets_files/doc/CoopNets_PAMI.pdf)".

Project Page: (http://www.stat.ucla.edu/~jxie/CoopNets/CoopNets.html)

## Requirements
- Python 2.7 or Python 3.3+
- [Tensorflow r1.0+](https://www.tensorflow.org/install/)
- [Scipy](https://www.scipy.org/install.html)
- [pillow](https://pillow.readthedocs.io/en/latest/installation.html)

## Usage

(1) For scene synthesis

(i) Training

First, put your data folder into `./data/scene/` directory, e.g., `./data/scene/rock/`
  
To train a model with ***rock*** dataset:

    $ python main.py --category rock --data_dir ./data/scene --output_dir ./output

synthesized results will be saved in `./output/rock/synthesis`. 

learned models will be saved in `./output/rock/checkpoints`. 

If you want to calculate inception score, use --calculate_inception=True. 

(ii) Testing for image synthesis

To test generator by synthesizing interpolation results with trained model:

    $ python main.py --test True --test_type syn --category rock --output_dir ./output --ckpt ./output/rock/checkpoints/model.ckpt-82000

testing results will be saved in `./output/alp/test`


If category is mnist, data will be downloaded and parzen window-based log-likelihood is calculated automatically. 

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
