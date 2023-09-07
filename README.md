## 3D-VisTA: Pre-trained Transformer for 3D Vision and Text Alignment

<p align="left">
    <a href='https://arxiv.org/pdf/2308.04352.pdf'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://arxiv.org/abs/2308.04352'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
    <a href='https://3d-vista.github.io/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
    <a href='https://huggingface.co/spaces/SceneDiffuser/SceneDiffuserDemo'>
      <img src='https://img.shields.io/badge/Demo-HuggingFace-yellow?style=plastic&logo=AirPlay%20Video&logoColor=yellow' alt='HuggingFace'>
    </a>
    <a href='https://drive.google.com/drive/folders/1UZ5V9VbPCU-ikiyj6NI4LyMssblwr1LC?usp=share_link'>
      <img src='https://img.shields.io/badge/Model-Checkpoints-orange?style=plastic&logo=Google%20Drive&logoColor=orange' alt='Checkpoints'>
    </a>
</p>

[Ziyu Zhu](https://zhuziyu-edward.github.io/), 
[Xiaojian Ma](https://jeasinema.github.io/), 
[Yixin Chen](https://yixchen.github.io/),
[Zhidong Deng](https://www.cs.tsinghua.edu.cn/csen/info/1165/4052.htm)📧,
[Siyuan Huang](https://siyuanhuang.com/)📧,
[Qing Li](https://liqing-ustc.github.io/)📧

This repository is the official implementation of the ICCV 2023 paper "3D-VisTA: Pre-trained Transformer for 3D Vision and Text Alignment".

[Paper](https://arxiv.org/pdf/2308.04352.pdf) |
[arXiv](https://arxiv.org/abs/2308.04352) |
[Project](https://3d-vista.github.io/) |
[HuggingFace Demo](https://huggingface.co/spaces/SceneDiffuser/SceneDiffuserDemo) |
[Checkpoints](https://drive.google.com/drive/folders/1UZ5V9VbPCU-ikiyj6NI4LyMssblwr1LC?usp=share_link)

<div align=center>
<img src='https://3d-vista.github.io/file/overall.png' width=60%>
</div>

### Abstract

3D vision-language grounding (3D-VL) is an emerging field that aims to connect the 3D physical world with natural language, which is crucial for achieving embodied intelligence. Current 3D-VL models rely heavily on sophisticated modules, auxiliary losses, and optimization tricks, which calls for a simple and unified model. In this paper, we propose 3D-VisTA, a pre-trained Transformer for 3D Vision and Text Alignment that can be easily adapted to various downstream tasks. 3D-VisTA simply utilizes self-attention layers for both single-modal modeling and multi-modal fusion without any sophisticated task-specific design. To further enhance its performance on 3D-VL tasks, we construct ScanScribe, the first large-scale 3D scene-text pairs dataset for 3D-VL pre-training. ScanScribe contains 2,995 RGB-D scans for 1,185 unique indoor scenes originating from ScanNet and 3R-Scan datasets, along with paired 278K scene descriptions generated from existing 3D-VL tasks, templates, and GPT-3. 3D-VisTA is pre-trained on ScanScribe via masked language/object modeling and scene-text matching. It achieves state-of-the-art results on various 3D-VL tasks, ranging from visual grounding and dense captioning to question answering and situated reasoning. Moreover, 3D-VisTA demonstrates superior data efficiency, obtaining strong performance even with limited annotations during downstream task fine-tuning.

### Install
1. Install conda package
```
conda env create --name envname --file=environments.yml
```

2. install pointnet2
```
cd vision/pointnet2
python3 setup.py install
```
### Prepare dataset
1. Follow [Vil3dref](https://github.com/cshizhe/vil3dref) and download scannet data under `data/scanfamily/scan_data`, this folder should look like
```
./data/scanfamily/scan_data/
├── instance_id_to_gmm_color
├── instance_id_to_loc
├── instance_id_to_name
└── pcd_with_global_alignment
```
2. Download [scanrefer+referit3d](https://github.com/cshizhe/vil3dref), [scanqa](https://github.com/ATR-DBI/ScanQA), and [sqa3d](https://github.com/SilongYong/SQA3D), and put them under `/data/scanfamily/annotations`

```
data/scanfamily/annotations/
├── meta_data
│   ├── cat2glove42b.json
│   ├── scannetv2-labels.combined.tsv
│   ├── scannetv2_raw_categories.json
│   ├── scanrefer_corpus.pth
│   └── scanrefer_vocab.pth
├── qa
│   ├── ScanQA_v1.0_test_w_obj.json
│   ├── ScanQA_v1.0_test_wo_obj.json
│   ├── ScanQA_v1.0_train.json
│   └── ScanQA_v1.0_val.json
├── refer
│   ├── nr3d.jsonl
│   ├── scanrefer.jsonl
│   ├── sr3d+.jsonl
│   └── sr3d.jsonl
├── splits
│   ├── scannetv2_test.txt
│   ├── scannetv2_train.txt
│   └── scannetv2_val.txt
└── sqa_task
    ├── answer_dict.json
    └── balanced
        ├── v1_balanced_questions_test_scannetv2.json
        ├── v1_balanced_questions_train_scannetv2.json
        ├── v1_balanced_questions_val_scannetv2.json
        ├── v1_balanced_sqa_annotations_test_scannetv2.json
        ├── v1_balanced_sqa_annotations_train_scannetv2.json
        └── v1_balanced_sqa_annotations_val_scannetv2.json
```
3. Download all checkpoints and put them under `project/pretrain_weights`

| Checkpoint           | Link                                                         | Note                                              |
| :------------------- | ------------------------------------------------------------ | ------------------------------------------------- |
| Pre-trained  | [link](https://drive.google.com/file/d/1x6xNDGjOZHAIj2GsxGzH3YUY4adZzal0/view?usp=sharing) | 3D-VisTA Pre-trained checkpoint.                  |
| ScanRefer            | [link](https://drive.google.com/file/d/1jsAPjI4UoP0DZqnsVWrO3SQXXM9gxzz1/view?usp=share_link) | Fine-tuned ScanRefer from pre-trained checkpoint. |
| ScanQA               | [link](https://drive.google.com/file/d/1AVfLISS8soTvaLmPdUHdWmirzIunLSi5/view?usp=share_link) | Fine-tined ScanQA from pre-trained checkpoint.    |
| Sr3D                 | [link](https://drive.google.com/file/d/1KXcxr1YkAlrFQS8vEQMtuVruBgmjlOPn/view?usp=share_link) | Fine-tuned Sr3D from pre-trained checkpoint.      |
| Nr3D                 | [link](https://drive.google.com/file/d/1lGg2o_r7cHJaTWa71-RTk-7141LAUCYj/view?usp=share_link) | Fine-tuned Nr3D from pre-trained checkpoint.      |
| SQA                  | [link](https://drive.google.com/file/d/1gCnR6XRj8TwuqbEaHk8FwfLClj6qFDh7/view?usp=share_link) | Fine-tuned SQA from pre-trained checkpoint.       |
| Scan2Cap             | [link](https://drive.google.com/file/d/1UF3zb7yIRsmJhM07A78eFxEptyNfF3k2/view?usp=share_link) | Fine-tuned Scan2Cap from pre-trained checkpoint.  |

### Run 3D-VisTA
To run 3D-VisTA, use the following command, task includes scanrefer, scanqa, sr3d, nr3d, sqa, and scan2cap.
```
python3 run.py --config project/vista/{task}_config.yml
```

### Acknowledgement
We would like to thank the authors of [Vil3dref](https://github.com/cshizhe/vil3dref) and for their open-source release.

### News

- [ 2023.08 ] First version!
- [ 2023.09 ] We release codes for all downstream tasks.

### Citation:
```
@article{zhu2023vista,
  title={3D-VisTA: Pre-trained Transformer for 3D Vision and Text Alignment},
  author={Zhu, Ziyu and Ma, Xiaojian and Chen, Yixin and Deng, Zhidong and Huang, Siyuan and Li, Qing},
  journal={ICCV},
  year={2023}
}
```
