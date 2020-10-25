

# pytorch-deepglr
A pytorch implementation of Deep Graph Laplacian Regularization for image denoising. Original work: [Zeng et al.](http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Zeng_Deep_Graph_Laplacian_Regularization_for_Robust_Denoising_of_Real_Images_CVPRW_2019_paper.pdf)
<p align="center">
  <img width="817" height="300" src="img/sample2.JPG" class="img-responsive">
</p>

Table of Contents
=================

   * [Installation](#installation)
   * [Basic usage](#basic-usage)
      * [Evaluate a trained model](#evaluate-a-trained-model)
         * [Evaluate DeepGLR](#evaluate-deepglr)
         * [Evaluate GLR](#evaluate-glr)
      * [Train a model](#train-a-model)
         * [Train a DeepGLR](#train-a-deepglr)
         * [Train a GLR](#train-a-glr)
      * [Remove noise of a single image using a trained DeepGLR](#remove-noise-of-a-single-image-using-a-trained-deepglr)
   * [Acknowledgments](#acknowledgments)
   
# Installation
1. Clone this [repository](https://github.com/huyvd7/pytorch-deepglr)

    ```git
    git clone https://github.com/huyvd7/pytorch-deepglr
    ```
2. Install required packages listed in requirements.txt (with Python 3.7)   
    
# Basic usage

We provide implementations for both DeepGLR and GLR since DeepGLR is a stacking version of GLRs. GLR is a smaller network and can be test more quickly, but it has poorer performance than DeepGLR.

## Evaluate a trained model

### Evaluate DeepGLR

    python validate_DGLR.py dataset/sample/ -m model/YOUR_MODEL -w
    
### Evaluate GLR

    python validate_GLR.py dataset/sample/ -m model/YOUR_MODEL -w 
    
The above commands resize the input images located at ```dataset/sample/``` to square images with size ```width x width```, then performs evaluation for the given trained model ```model/YOUR_MODEL``` and saves outputs to current directory.

## Train a model
### Train a DeepGLR
    
    python train_DGLR.py dataset/sample/ -n MODEL_NAME -d ./ -w width -e epoch -b batch_size -l learning_rate
    
The above command will train a new DeepGLR. If you want to continue training an existing DeepGLR instead of training from scratch, you can add ```-m PATH_TO_EXISTED_MODEL```:

    python train_DGLR.py dataset/train/ -m model/deepglr.pretrained -n MODEL_NAME -d ./ -w width -e epoch -b batch_size -l learning_rate

### Train a GLR

    python train_GLR.py dataset/train/ -n MODEL_NAME -d ./ -w width -e epoch -b batch_size -l learning_rate
    
Same parameters as DeepGLR.


## Remove noise of a single image using a trained DeepGLR (LEGACY, MIGHT NOT WORKING)

    python denoise.py dataset/test/noisy/2_n.bmp -m model/deepglr.pretrained -w 324 -o OUTPUT_IMAGE.PNG

The above command will resize the ```dataset/test/noisy/2_n.bmp``` to ```324x324```, then denoise it using a trained DeepGLR ```model/deepglr.pretrained``` and save the result at ```OUTPUT_IMAGE.PNG```.

