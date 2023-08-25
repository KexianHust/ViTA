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

### [Project Page](https://kexianhust.github.io/ViTA/) | [Arxiv]() | [Video](https://youtu.be/3NgVnLWGQTU)

## :speech_balloon: Abstract
<b>TL; DR: **ViTA** is a robust and fast video depth estimation model that estimates spatially accurate and temporally consistent depth maps from any monocular video.

> Depth information plays a pivotal role in numerous computer vision applications, including autonomous driving, 3D reconstruction, and 3D content generation. When deploying depth estimation models in practical applications, it is essential to ensure that the models have strong generalization capabilities. However, existing depth estimation methods primarily concentrate on robust single-image depth estimation, leading to the occurrence of flickering artifacts when applied to video inputs. On the other hand, video depth estimation methods either consume excessive computational resources or lack robustness. To address the above issues, we propose ViTA, a video transformer adaptor, to estimate temporally consistent video depth in the wild. In particular, we leverage a pre-trained image transformer (\ie DPT) and introduce additional temporal embeddings in the transformer blocks. Such designs enable our ViTA to output reliable results given an unconstrained video. Besides, we present a spatio-temporal consistency loss for supervision. The spatial loss computes the per-pixel discrepancy between the prediction and the ground truth in space, while the temporal loss regularizes the inconsistent outputs of the same point in consecutive frames. To find the correspondences between consecutive frames, we design a bi-directional warping strategy based on the forward and backward optical flow. During inference, our ViTA no longer requires optical flow estimation, which enables it to estimate spatially accurate and temporally consistent video depth maps with fine-grained details in real time. We conduct a detailed ablation study to verify the effectiveness of the proposed components. Extensive experiments on the zero-shot cross-dataset evaluation demonstrate that the proposed method is superior to previous methods. Code can be available at \url{https://kexianhust.github.io/ViTA/}.

<!-- | ![visitors](https://visitor-badge.laobi.icu/badge?page_id=KexianHust/ViTA) -->

  </br>
  
  <div style="width: 100%; text-align: center; margin:auto;">
      <img style="width:100%" src="st-slice-sota.png">
  </div>

</div>

## :eyes: Updates
* [08/2023] Initial release of inference code and models.
* [08/2023] The paper is accepted by T-MM.

## :wrench: Setup 

1) Download the weights and place them in the `weights` and `checkpoints` folder, respectively:
- [dpt_hybrid-midas-501f0c75.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt), [Mirror](https://drive.google.com/file/d/1dgcJEYYw1F8qirXhZxgNK8dWWz_8gZBD/view?usp=sharing)
- [vita-hybrid-intval=3.pth](https://drive.google.com/file/d/1z4vKbGaZRUDMMftRKmTGv48Ih_B3a_Y5/view?usp=share_link)

  
2) Set up dependencies: 

    ```shell
    pip install torch torchvision opencv-python timm
    ```

   The code was tested with Python 3.9, PyTorch 1.11.0

## :zap: Inference 

1) Place one or more input videos in the folder `input_video`.

2) Run our model:

    ```shell
    python demo.py
    ```

3) The results are written to the folder `output_monodepth`.


## :blush: Citation
If you find our work useful in your research, please consider citing the paper.
```
@article{Xian_2023_TMM,
author = {Xian, Ke and Peng, Juewen and Cao, Zhiguo and Zhang, Jianming and Lin Guosheng},
title = {ViTA: Video Transformer Adaptor for Robust Video Depth Estimation},
journal = {IEEE Transactions on Multimedia},
year = {2023}
}
```

## :key: License
Distributed under the S-Lab License. Please refer to [LICENSE](./LICENSE) for more details.

## :email: Contact
Please contact Ke Xian (ke.xian@ntu.edu.sg or xianke1991@gmail.com) if you have any questions.
