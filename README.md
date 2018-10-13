# Cooperative Training of Descriptor and Generator Networks

This repository contains a tensorflow implementation for the paper "[Cooperative Training of Descriptor and Generator Networks](http://www.stat.ucla.edu/~jxie/CoopNets/CoopNets_files/doc/CoopNets_PAMI.pdf)".

Project Page: (http://www.stat.ucla.edu/~jxie/CoopNets/CoopNets.html)

## Requirements
- Python 2.7 or Python 3.3+
- [Tensorflow r1.0+](https://www.tensorflow.org/install/)
- [Scipy](https://www.scipy.org/install.html)
- [pillow](https://pillow.readthedocs.io/en/latest/installation.html)

## Usage

### (1) For scene synthesis

(i) Training

First, put your data folder into `./data/scene/` directory, e.g., `./data/scene/rock/`
  
To train a model with ***rock*** dataset:

    $ python main.py --category rock --data_dir ./data/scene --output_dir ./output --net_type scene --image_size 64

The synthesized results will be saved in `./output/rock/synthesis`. 

The learned models will be saved in `./output/rock/checkpoints`. 

If you want to calculate inception score, use --calculate_inception=True. 

(ii) Testing for image synthesis

    $ python main.py --test True --test_type syn --category rock --output_dir ./output --ckpt ./output/rock/checkpoints/model.ckpt-82000

testing results will be saved in `./output/rock/test/synthesis`

(iii) Testing for interpolation

To test generator by synthesizing interpolation results with trained model:

    $ python main.py --test True --test_type inter --category rock --output_dir ./output --ckpt ./output/rock/checkpoints/model.ckpt-82000
    
testing results will be saved in `./output/rock/test/interpolation`
    
### (2) For MNIST handwritten digits synthesis

If category is mnist, training data will be downloaded automatically 

    $ python main.py --category mnist --net_type mnist --image_size 28

If you want to calculate parzen window-based log-likelihood, use --calculate_parzen=True. 

The synthesized results will be saved in `./output/mnist/synthesis`. 

The learned models will be saved in `./output/mnist/checkpoints`. 


## Results
Synthesis
<p align="center">
    <img src="https://github.com/jianwen-xie/CoopNets/blob/master/demo/des_syn.png" width="350px"/>
    <img src="https://github.com/jianwen-xie/CoopNets/blob/master/demo/des_syn_mnist.png" width="350px"/>
</p>
    
Interpolation
<p align="center">
    <img src="https://github.com/jianwen-xie/CoopNets/blob/master/demo/des_syn_mnist.png" width="350px"/>
</p>

## Reference
    @article{coopnets,
        author = {Xie, Jianwen and Lu, Yang and Gao, Ruiqi and Zhu, Song-Chun and Wu, Ying Nian},
        title = {Cooperative Training of Descriptor and Generator Networks},
        journal={IEEE transactions on pattern analysis and machine intelligence (PAMI)},
        year = {2018},
        publisher={IEEE}
    }
    
For any questions, please contact Jianwen Xie (jianwen@ucla.edu), Ruiqi Gao (ruiqigao@ucla.edu) and Zilong Zheng (zilongzheng0318@ucla.edu)
