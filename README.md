# ViTA: Video Transformer Adaptor for Robust Video Depth Estimation
This repository contains a pytorch implementation of our T-MM paper:
> ViTA: Video Transformer Adaptor for Robust Video Depth Estimation
> 
> Ke Xian, Juewen Peng, Zhiguo Cao, Jianming Zhang, Guosheng Lin

[Project Page](https://kexianhust.github.io/ViTA/)
![image](https://github.com/KexianHust/ViTA/blob/main/st-slice-sota.png)

[Video](https://youtu.be/3NgVnLWGQTU)

## Changelog
* [Aug. 2023] Initial release of inference code and models

## Setup 

1) Download the weights and place them in the `weights` and `checkpoints` folder, respectively:
- [dpt_hybrid-midas-501f0c75.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt), [Mirror](https://drive.google.com/file/d/1dgcJEYYw1F8qirXhZxgNK8dWWz_8gZBD/view?usp=sharing)
- [vita-hybrid-intval=3.pth](https://drive.google.com/file/d/1z4vKbGaZRUDMMftRKmTGv48Ih_B3a_Y5/view?usp=share_link)

  
2) Set up dependencies: 

    ```shell
    pip install torch torchvision opencv-python timm
    ```

   The code was tested with Python 3.9, PyTorch 1.11.0

## Usage 

1) Place one or more input videos in the folder `input_video`.

2) Run our model:

    ```shell
    python demo.py
    ```


3) The results are written to the folder `output_monodepth`.

Please contact Ke Xian (ke.xian@ntu.edu.sg or xianke1991@gmail.com) if you have any questions.


If you find our work useful in your research, please consider citing the paper.

```
@article{Xian_2023_TMM,
author = {Xian, Ke and Peng, Juewen and Cao, Zhiguo and Zhang, Jianming and Lin Guosheng},
title = {ViTA: Video Transformer Adaptor for Robust Video Depth Estimation},
journal = {IEEE Transactions on Multimedia},
year = {2023}
}
```
