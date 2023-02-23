# Pavement Distress Classification
With the rapid growth of transport infrastructures such as airports, bridges, and roads, pavement maintenance is deemed as a crucial element of sustainable pavement today. It is a trend to automate this process via machine learning and pattern recognition techniques, which can significantly reduce the cost of labor and resources. One of the core tasks in pavement maintenance is pavement distress classification (PDC), which aims to detect the damaged pavement and recognize its specific distress category. These two steps are also referred to as pavement distress detection and pavement distress recognition, respectively.

**This repo collects related datasets and some papers. Moreover, all code will be integrated into this repo.**

## Table of Contents

- [Datasets](#dataset)
  - [CQU-BPDD](#cqu-bpdd)
  - [CFD-PDD、CrackTree200-PDD、Crack500-PDD](#cfd-pddcracktree200-pddcrack500-pdd)
- [Methods](#method)
  - [IOPLIN (T-ITS)](#ioplin-t-its)
  - [PicT (MM 2022)](#pict-acmmm-2022)
  - [WSPLIN (T-ITS)](#wsplin-t-its)
  - [WSPLIN-IP (ICASSP 2021)](#wsplin-ip-icassp-2021)
  - [DPSSL (Electronics Letters)](#dpssl-electronics-letters)
- [Citations](#citations)

## Dataset

### CQU-BPDD

<img src=".\doc\cqu_bpdd.png" alt="cqu_bpdd" width="500px" />

This [dataset](https://github.com/DearCaat/CQU-BPDD) consists of 60,056 bituminous pavement images, which were automatically captured by the in-vehicle cameras of the professional pavement inspection vehicle at different times from different areas in southern China . Each pavement image is corresponding to a 2 × 3 meters pavement patch of highways and its resolution is 1200×900. The CQU-BPDD involves seven different distresses, namely **transverse crack, massive crack, alligator crack, crack pouring, longitudinal crack, ravelling, repair, and the normal ones.**

[Downloading the dataset](https://pan.baidu.com/s/1KFtu5ZGb3lqqwxRoKUEzMw), _Password:mtf7. (Please note: CQU-BPDD can be only used in the uncommercial case.)_

### CFD-PDD、CrackTree200-PDD、Crack500-PDD

<img src=".\doc\cfd.png" alt="cfd" width="500px" />

Three public pavement crack segmentation (pixel-level pavement crack detection) datasets, namely **Crack Forest Dataset (CFD), CrackTree200, and Crack500**, are adopted for validation. We automatically produce the normal version of each diseased images via replacing the disease pixels with their neighbor normal pixels. However, such an automatic normal image fashion does not always work well for all samples. We manually filter out some low-quality generated normal images and only retain the high-quality ones. Finally, we have **155 diseased images and 114 recovered normal images on CFD dataset, 206 diseased images and 191 recovered normal images on CrackTree200 dataset, while 494 diseased images and 286 recovered normal images**.

**Download CFD-PDD**: [Baidu Cloud](https://pan.baidu.com/s/1FCnwC3UUf9v35_p50eOpog?pwd=6fur)

**Download CrackTree200-PDD**: [Baidu Cloud](https://pan.baidu.com/s/1KYusgYlb4VkPphjqBlIFmw?pwd=oziw)

**Download Crack500-PDD**: [Baidu Cloud](https://pan.baidu.com/s/1QbSTx3z9m5L26DTZAUqFqg?pwd=qhhn)

## Method

### IOPLIN (IEEE T-ITS)

<img src=".\doc\ioplin.png" alt="ioplin" width="500px" />

This [paper](https://ieeexplore.ieee.org/abstract/document/9447759) **first proposes the main PDC datasets**: CQU-BPDD、CFD-PDD、CrackTree200-PDD, and achieves a good performance based on the **E-M optimization strategy**.

**Code: [Github](https://github.com/DearCaat/IOPLIN)**

### PicT (ACMMM 2022)

<img src=".\doc\pict.png" alt="image-20221020105610479" width="500px" />

This [paper](https://dl.acm.org/doi/abs/10.1145/3503161.3548176) first introduces the **Vision Transformer** into the PDC, and achieves 2.4% more detection performance gains in P@R, 3.9% recognition performance gains in F1, **1.8x higher throughput**, and **7x faster training speed**.

**arXiv version: [PicT](https://arxiv.org/abs/2209.10074)**

**Code: [Github](https://github.com/DearCaat/PicT)**

### WSPLIN-IP (ICASSP 2021)

<img src=".\doc\wsplin_ip.png" alt="wsplin_ip" width="500px" />

This [paper](https://ieeexplore.ieee.org/abstract/document/9413517) proposes an **end-to-end training framework** based on the weakly supervised patch label inference network.

**Code: [Github](https://github.com/DearCaat/WSPLIN)**

### WSPLIN (IEEE T-ITS)

<img src=".\doc\wsplin.png" alt="wsplin" width="500px" />

This [paper](https://ieeexplore.ieee.org/document/10050387) is an extension of [WSPLIN-IP](###WSPLIN-IP (ICASSP 2021)). It rethinks the patch collection strategy, and **finds a good trade-off between performance and efficiency**.

**arXiv version: [WSPLIN](https://arxiv.org/abs/2203.16782)**

**Code: [Github](https://github.com/DearCaat/WSPLIN)**
### DPSSL (Electronics Letters)

<img src=".\doc\dpssl.png" alt="dpssl" width="500px" />

This [paper](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ell2.12570) try to rethink PDC from the perspective of **Multi-Instance Learning**, and leverages **attention mechanism and Knowledge Distillation** to improve the performance and efficiency.

## Citations

### Citing IOPLIN、CQU-BPDD、CFD-PDD、CrackTree200-PDD

```
@article{tang2021iteratively,
  title={An iteratively optimized patch label inference network for automatic pavement distress detection},
  author={Tang, Wenhao and Huang, Sheng and Zhao, Qiming and Li, Ren and Huangfu, Luwen},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2021},
  publisher={IEEE}
}
```
### Citing WSPLIN、Crack500-PDD
```
@article{huang2022weakly,
  title={Weakly Supervised Patch Label Inference Networks for Efficient Pavement Distress Detection and Recognition in the Wild},
  author={Huang, Sheng and Tang, Wenhao and Huang, Guixin and Huangfu, Luwen and Yang, Dan},
  journal={arXiv preprint arXiv:2203.16782},
  year={2022}
}
```
### Citing PicT
```
@inproceedings{tang2022pict,
  title={PicT: A Slim Weakly Supervised Vision Transformer for Pavement Distress Classification},
  author={Tang, Wenhao and Huang, Sheng and Zhang, Xiaoxian and Huangfu, Luwen},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={3076--3084},
  year={2022}
}
```

### Citing WSPLIN-IP

```
@inproceedings{huang2021weakly,
  title={Weakly supervised patch label inference network with image pyramid for pavement diseases recognition in the wild},
  author={Huang, Guixin and Huang, Sheng and Huangfu, Luwen and Yang, Dan},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7978--7982},
  year={2021},
  organization={IEEE}
}
```
### Citing DPSSL

```
@article{zhang2022efficient,
  title={Efficient pavement distress classification via deep patch soft selective learning and knowledge distillation},
  author={Zhang, Shizheng and Tang, Wenhao and Wang, Jing and Huang, Sheng},
  journal={Electronics Letters},
  year={2022},
  publisher={Wiley Online Library}
}
```

### Citing CFD

```
@article{shi2016automatic,
  title={Automatic road crack detection using random structured forests},
  author={Shi, Yong and Cui, Limeng and Qi, Zhiquan and Meng, Fan and Chen, Zhensong},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={17},
  number={12},
  pages={3434--3445},
  year={2016},
  publisher={IEEE}
}
```

### Citing CrackTree200

```
@article{zou2012cracktree,
  title={CrackTree: Automatic crack detection from pavement images},
  author={Zou, Qin and Cao, Yu and Li, Qingquan and Mao, Qingzhou and Wang, Song},
  journal={Pattern Recognition Letters},
  volume={33},
  number={3},
  pages={227--238},
  year={2012},
  publisher={Elsevier}
}
```

### Citing Crack500

```
@article{yang2019crack500,
  title={Feature pyramid and hierarchical boosting network for pavement crack detection},
  author={Yang, Fan and Zhang, Lei and Yu, Sijia and Prokhorov, Danil and Mei, Xue and Ling, Haibin},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={21},
  number={4},
  pages={1525--1535},
  year={2019},
  publisher={IEEE}
}
```
