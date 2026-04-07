# ELIP: Enhanced Visual-Language Foundation Models for Image Retrieval
This is the implementation of the IEEE CBMI 2025 paper "ELIP: Enhanced Visual-Language Foundation Models for Image Retrieval" created by [Guanqi Zhan*](https://www.robots.ox.ac.uk/~guanqi/), [Yuanpei Liu*](https://scholar.google.com/citations?user=GHTB15QAAAAJ&hl=zh-CN), [Kai Han](https://www.kaihan.org/), [Weidi Xie](https://weidixie.github.io/) and [Andrew Zisserman](https://scholar.google.com/citations?user=UZ5wscMAAAAJ&hl=en).

<p align="center">
    <a href="https://arxiv.org/abs/2502.15682"><img src="https://img.shields.io/badge/arXiv-2502.15682-b31b1b"></a>
    <a href="https://www.robots.ox.ac.uk/~vgg/research/elip/"><img src="https://img.shields.io/badge/Project-Website-blue"></a>
    <a href="#jump"><img src="https://img.shields.io/badge/Citation-8A2BE2"></a>
</p>


## Evaluation Dataset 
The datasets we use in this work include: [COCO](https://drive.google.com/file/d/1dbTYF778NZ9gkaZpAjmu-Q-VvAfPAR9x/view?usp=sharing), [Flickr](https://drive.google.com/file/d/1ACRUyIPYlDnaPKXBgfvUTDAEGnh5jhW4/view?usp=sharing), [Occluded COCO](https://drive.google.com/drive/folders/1duYPsnoyslUkQ9MJpWMOi4Q13ml-5IwS?usp=drive_link) and [ImageNet-R](https://drive.google.com/file/d/1i-m5Fbmyp84kXc1SMKwi5SukGawETdgM/view?usp=sharing)

## Running
For ELIP-C, ELIP-S, and ELIP-S2, change into the `ELIP-C` directory and follow the corresponding [README](./ELIP-C/README.md). For ELIP-B, change into the `ELIP-B` directory and consult its [README](./ELIP-B/README.md).

## Acknowledgement
This repository is built upon [OpenCLIP](https://github.com/mlfoundations/open_clip) and [LAVIS](https://github.com/salesforce/LAVIS). Thanks for those well-organized codebases.

## Citing this work
<span id="jump"></span>
If you find this repo useful for your research, please consider citing our paper:

```
@inproceedings{Zhan2025ELIP,
    author = {Zhan, Guanqi and Liu, Yuanpei and Han, Kai and Xie, Weidi and Zisserman, Andrew},
    title = {ELIP: Enhanced Visual-Language Foundation Models for Image Retrieval},
    booktitle = {International Conference on Content-Based Multimedia Indexing (CBMI)},
    year = {2025}
}
```
