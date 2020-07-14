# TCSVT-LightWeight-CNNs
Code for our IEEE TCSVT Paper: Lightweight Modules for Efficient Deep Learning based Image Restoration

Authors: Avisek Lahiri*, Sourav Bairagya*, Sutanu Bera, Siddhant Haldar, Prabir Kumar Biswas <br/>
(* equal contribution) <br/>
1. Paper Link: https://arxiv.org/abs/2007.05835  <br/>
2. IEEE Early Access Link: https://ieeexplore.ieee.org/document/9134805 <br/>

### Key Points from Paper
* Paper provides re-usable modules to be plugged and played to compress a given CNN
* Select any favourite full-scale baseline for low-level vision applications
* Replace 3X3 conv by **LIST** layer
* Replace dilated conv layer **GSAT** layer
* Achieve efficient up/down-sample with **Bilinear SubSampling** followed by **LIST** layer

### TensorflowExamples
This contains the basic proposed modules in Tensorflow<br/>
TensorflowExamples/basicModules.py contains the proposed **LIST**, **GSAT** modules <br/>
It also contains the framework for **LIST** based up/down-sampling in a CNN  <br/>

### PytorchExamples
It contains the basic proposed modules in Pytorch <br/>

* block.py has the implementatin of **LIST** module based DnCNN denoising framework
* Denoising_demo.ipynb is a notebook to reflect our training/inference setup for DnCNN experiments

![Cover Picture](/combined_cover.png)
