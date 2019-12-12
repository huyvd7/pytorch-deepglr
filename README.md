# pytorch-deepglr
A pytorch implementation of Deep Graph Laplacian Regularization for image denoising. Original work: [Zeng et al.](http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Zeng_Deep_Graph_Laplacian_Regularization_for_Robust_Denoising_of_Real_Images_CVPRW_2019_paper.pdf)

# Installation
1. Clone this repo
2. Install required packages listed in requirements.txt

      ``` pip install -r requirements.txt ```

# Basic usage
## Evaluate a trained model
We provide two sample scripts to evaluate the trained models. For both DeepGLR and GLR. The GLR is a smaller network and can be tested quickly

### Evaluate DeepGLR
Using validate_DGLR.py 

      ```python validate_DGLR.py dataset/test/ -m model/deepglr.pretrained -w 324```

### Evaluate GLR
Using validate_GLR.py: 

      ```python validate_DGLR.py dataset/test/ -m model/glr.pretrained -w 324```

### Parameters
These two scripts have similar CLI parameters:

      
## Train DeepGLR from scratch
