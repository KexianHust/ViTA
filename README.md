<div align="center">

  <h1>ViTA: Video Transformer Adaptor for Robust Video Depth Estimation</h1>
  
  <div>
      <a href="https://sites.google.com/site/kexian1991/" target="_blank">Ke Xian<sup>1</sup></a>,
      <a href="https://juewenpeng.github.io/" target="_blank">Juewen Peng<sup>2</sup></a>,
      <a href="http://english.aia.hust.edu.cn/info/1085/1528.htm" target="_blank">Zhiguo Cao<sup>2</sup></a>,
      <a href="https://jimmie33.github.io/" target="_blank">Jianming Zhang<sup>3</sup></a>,
      <a href="https://guosheng.github.io/" target="_blank">Guosheng Lin<sup>1*</sup></a>
  </div>
  <div>
      <br/><sup>1</sup>S-Lab, Nanyang Technological Univerisity<br/><sup>2</sup>Huazhong University of Science and Technology<br/><sup>3</sup>Adobe Research<br/>
  </div>
  <div>
  IEEE T-MM.
</div>

### [Project Page](https://kexianhust.github.io/ViTA/) | [Arxiv]() | [Video]([https://youtu.be/SNV9F-60xrE](https://youtu.be/3NgVnLWGQTU))

<!-- | ![visitors](https://visitor-badge.laobi.icu/badge?page_id=Yuxinn-J/Scenimefy) -->

  </br>
  
  <div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="st-slice-sota.png">
  </div>

</div>

## Updates
* [08/2023] Initial release of inference code and models.
* [08/2023] The paper is accepted by T-MM.

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
