# pytorch-deepglr
A pytorch implementation of Deep Graph Laplacian Regularization for image denoising. Original work: [Zeng et al.](http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Zeng_Deep_Graph_Laplacian_Regularization_for_Robust_Denoising_of_Real_Images_CVPRW_2019_paper.pdf)

# Installation
1. Clone this repo
2. Install required packages listed in requirements.txt

      ``` pip install -r requirements.txt ```
3. If you have issues when installing PyTorch. Please follow their official installation guide []()
# Basic usage
## Evaluate a trained model
We provide two sample scripts to evaluate the trained models. For both DeepGLR and GLR. The GLR is a smaller network and can be tested quickly

### Evaluate DeepGLR
Using validate_DGLR.py. The following command will resize the input images located at ```dataset/test/``` to square images with size ```324 x 324```, then perform evaluation for the given trained DeepGLR ```model/deepglr.pretrained``` and write outputs to ```./``` (current directory)

    python validate_DGLR.py dataset/test/ -m model/deepglr.pretrained -w 324 -o ./

### Evaluate GLR
Using validate_GLR.py. This runs much faster but since it's a single GLR layer, it has poor results. The following command will resize the input images located at ```dataset/test/``` to square images with size ```324 x 324```, then perform evaluation for the given trained GLR ```model/glr.pretrained``` and write outputs to ```./``` (current directory)

    python validate_GLR.py dataset/test/ -m model/glr.pretrained -w 324
      
### NOTE
The provided sample dataset in this directory is a resized version of RENOIR dataset (resized to 720x720). The original dataset is located at [Adrian Barbu's site](http://adrianbarburesearch.blogspot.com/p/renoir-dataset.html). 
Because this is a resize version, the evaluation results are different from what were reported. To reproduce the same results as written in the report, please replace the sample dataset in this repository with the original one. For your convinient, you can use this [Google Drive mirror](http://adrianbarburesearch.blogspot.com/p/renoir-dataset.html). This mirror will be deleted at the end of Dec 2019
      
## Train DeepGLR from scratch
