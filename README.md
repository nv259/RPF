# From Region to Patch: Attribute-Aware Foreground-Background Contrastive Learning for Fine-Grained Fashion Retrieval
This is a repository contains the implementation of API based on SIGIR'23 full paper [From Region to Patch: Attribute-Aware Foreground-Background Contrastive Learning for Fine-Grained Fashion Retrieval](https://doi.org/10.48550/arXiv.2305.10260).
![network structure](imgs/myframework.jpg)

## Abstract
Attribute-specific fashion retrieval (ASFR), a nuanced task in content-based image retrieval, focuses on identifying specific attributes and details in fashion items from images. Addressing this complex challenge, in this report, we introduce a novel Fine-grained Fashion Retrieval system, leveraging the Region-to-Patch Framework (RPF) enhanced with large-scale search techniques including K-D tree, LSH and FAISS. Our system uniquely addresses the challenge of attribute-specific fashion retrieval by integrating dual modal inputs: an image query and a text-based attribute description. Furthermore, we present a comprehensive re-evaluation of our system against the original RPF and other state-of-the-art methods on a benchmark dataset of 180k fashion images.

## Team Members
| **Full Name**      | **MSSV** | **Role**   |
| :----------------: | :------: | :--------: |
| *Nguyen Viet Nhat*   | *21520378* | *Leader* |
| Ha Van Hoang       | 21520033 | Member     |
| Nguyen Quoc Truong | 21521604 | Member     |
| Vo Thi Phuong Anh  | 21522883 | Member     | 

## Table of Contents

* [Environments](#environments)
* [System](#system)
* [Datasets](#datasets)
* [Configuration](#configuration)
* [Training](#training)
* [Evaluation](#evaluation)
* [Performance](#performance)

## Environments
- **Ubuntu** 22.04
- **CUDA** 11.7
- **Python** 3.7

Create virtual environment for python3.7
```sh
pip install virtualenv
virtualenv --python="path/to/python3.7" py37
source py37/bin/activate
```

Install other required packages by
```sh
pip install -r requirements.txt
```

## System
- **Input**: 
  - Fashion image (*visual data*)
  - Attributes (*textual data*) 

- **Output**:
  - Ranked list of fashion items (expressed by image_id in database) that relevant to attributes in query image.

- **Example**:
```python
Input:
{
  "img_path": "path/to/query/image",
  "attrs": "Độ dài tay; Thiết kế cổ áo; Độ dài váy"
}

Output:
{
  "ids": [ 124, 768, 1243, 8940, ... ],
  "similarities": [ 0.94295, 0.87365, 0.7645, 0.652, ... ],
  "retrieval_times": 0.566
}
```


## Datasets
In this API, we use only two fashion related datasets, i.e., FashionAI and DeepFashion. The former is used for training (train, dev) and as collection (test 14400 images), while the latter is used for evaluating system. Please download and put them in the corresponding folders.
### Download Data
#### FashionAI

As the full FashionAI has not been publicly released, we utilize its early version for the [FashionAI Global Challenge 2018](https://tianchi.aliyun.com/competition/entrance/231671/introduction?spm=5176.12281949.1003.9.493e3eafCXLQGm). You can first sign up and download two training subsets:

- **fashionAI_attributes_train1.zip(6G)**
- **fashionAI_attributes_train2.zip(7G)**. 

Once done, you should uncompress and link them into the `data/FashionAI` directory.

#### DeepFashion

[DeepFashion](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.pdf) is a large dataset which consists of four benchmarks for various tasks in the field of clothing including [category and attribute prediction](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) which we use for our experiments, in-shop clothes retrieval, fashion landmark  detection and consumer-to-shop clothes retrieval. Download the images into a `img` directory that should be created in `data/DeepFashion` directory.

### Configuration

The behavior of our codes is controlled by configuration files under the `config` directory. 

```sh
config
│── FashionAI
│   ├── FashionAI.yaml
│   ├── s1.yaml
│   └── s2.yaml
├── DARN
│   ├── DARN.yaml
│   ├── s1.yaml
│   └── s2.yaml
└── DeepFashion
    ├── DeepFashion.yaml
    ├── s1.yaml
    └── s2.yaml
```

Each dataset is configured by two types of configuration files. One is `<Dataset>.yaml` that specifies basic dataset information such as path to the training data and annotation files. The other two set some training options as needed.

If the above `data` directory is placed at the same level with `main.py`, no changes are needed to the configuration files. Otherwise, be sure to correctly configure relevant path to the data according to your working environment.

## Startup

Download Google pre-trained ViT models for Patch-aware Branch and put into `pretrained` dir:
```bash
wget https://drive.google.com/file/d/1N2rdQcbhegIOB4fHpifi92w1Lp86umN1/view?usp=sharing
```

Pre-trained ResNet model for Region-aware Branch is auto downloaded and put into `pretrained` directory, but if error occurs, copy it manually into `pretrained` directory.

Download checkpoint for FashionAI models and place in `runs/FashionAI_s2`
- RPF on FashionAI: [released_model](https://drive.google.com/file/d/1pIJ2REblm2eXNq81vyhAj9bs8y1EzNvR/view?usp=sharing)

In this API, we implemented the multi-attributes version for input query. Therefore, we need to download the [collections](https://onedrive.live.com/?authkey=%21ALkPPaOECy0QoUM&id=406CC5EDDBB3A996%2119523&cid=406CC5EDDBB3A996&parId=root&parQt=sharedby&o=OneUp), extract it and put into `collections/multi_attrs/` directory. Single-attribute version is smaller and quicker but lack of practical, so we omit it, but if you concern it, feel free to contact 21520378@gm.uit.edu.vn

API is hosted at `0.0.0.0:8000` and receive HTTP POST at `0.0.0.0:8000/submit`
```sh
uvicorn my_api:app *or* python my_api.py
```

### Performance 
Expected mAP for top 50 on FashionAI Dataset
|Methods|skirt length| sleeve length| coat length |pant length |collar design| lapel design| neckline design| neck design| overall|
| :---------: | :--: | :--: | :--: | :---: | :---: |:--: | :--: | :---: | :---: |
|ASEN++|74.4878|64.9343|64.9360|74.1380|77.1294|71.1829|74.2218|73.1567|71.2090|
|RPF (default)|71.7552|62.4961|62.5449|71.9812|75.3120|72.7903|70.1916|66.9725|69.2555|
|RPF + K-D Tree|71.7552|62.4961|62.5434|71.9815|75.3120|72.7903|70.1916|66.9725|69.2553|
|RPF + LSH|69.9015|61.0136|63.5678|70.6794|74.1572|71.4406|67.7121|67.4673|68.2424|
|RPF + FAISS|71.7552|62.4961|62.5442|71.9812|75.3120|72.7903|70.1916|66.9725|69.2554|

Expected time (s) for top 50 on FashionAI Dataset
|Methods|skirt length| sleeve length| coat length |pant length |collar design| lapel design| neckline design| neck design| overall|
| :---------: | :--: | :--: | :--: | :---: | :---: |:--: | :--: | :---: | :---: |
|ASEN++|5.0354|8.5509|6.7652|4.9764|3.7669|3.1144|9.9180|2.7550|44.8822|
|RPF (default)|15.3965|23.6755|19.9781|15.3857|12.1933|10.2527|27.3010|9.2787|133.4616|
|RPF + K-D Tree|15.4125|23.3209|19.7294|15.2851|12.1535|10.2375|26.8065|9.2894|132.2348|
|RPF + LSH|14.8336|22.2084|19.4082|15.4584|12.0020|11.6832|25.0456|10.1672|130.8067|
|RPF + FAISS|14.1729|21.1337|18.0834|14.1678|11.4107|9.6600|23.9230|8.8454|121.3969|

Expected mAP for top 50 on DeepFashion Dataset
|Methods|texture |fabric |shape |part| style |overall|
| :---------: | :--: | :--: | :--: | :---: | :---: | :---: |
|ASEN++|24.1983|15.3561|24.2988|14.4966|9.8564|17.6751|
|RPF (default)|23.4559|15.3517|23.4140|14.7252|10.6022|17.5098|
|RPF + K-D Tree|23.4559|15.3517|23.4140|14.7252|10.6022|17.5098|
|RPF + LSH|24.8087|14.2795|22.2055|14.4342|9.1695|16.9795|
|RPF + FAISS|23.4559|15.3518|23.4140|14.7252|10.6022|17.5098|

Expected time (s) for top 50 on DeepFashion Dataset
|Methods|texture |fabric |shape |part| style |overall|
| :---------: | :--: | :--: | :--: | :---: | :---: | :---: |
|ASEN++|25.3470|51.6129|28.9521|23.6551|20.1089|149.6759|
|RPF (default)|49.5464|82.9808|54.6277|45.7379|40.0285|272.9214|
|RPF + K-D Tree|43.9967|69.5555|47.9082|43.9777|38.4290|243.8673|
|RPF + LSH|37.1015|54.9096|42.1345|39.0283|33.9313|207.1052|
|RPF + FAISS|37.0928|55.7777|40.4665|37.5468|33.4578|204.3416|

## Citation
If you find this repository useful, please consider citing our paper:
```
@inproceedings{RPF2023,
  title={From Region to Patch: Attribute-Aware Foreground-Background Contrastive Learning for Fine-Grained Fashion Retrieval},
  author={Jianfeng Dong and Xiaoman Peng and Zhe Ma and Daizong Liu and Xiaoye Qu and Xun Yang and Jixiang Zhu and Baolong Liu},
  booktitle={Proceedings of the 46rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2023}
}

```
