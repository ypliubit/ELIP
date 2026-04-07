## Dependencies
Please run
    
        python -m venv .venv2
        source .venv2/bin/activate
        pip install -r requirements.txt

to install the dependencies.

Replace `./.venv2/lib/python3.11/site-packages/timm/models/vision_transformer.py` and `./.venv2/lib/python3.11/site-packages/timm/models/vision_transformer_hybrid.py` with the files as in the folder `ELIPC_env_code`.

### Training data
The csv file containing training data caption and urls can be found in [this link](https://drive.google.com/file/d/1LnuAM4AX1ZfSYm-ysMDomOqa-UIgkSe0/view?usp=drive_link).


## Evaluation

### Step1. Download dataset annotations
&nbsp;&nbsp;&nbsp;&nbsp; COCO:
[coco_test.csv](https://drive.google.com/file/d/1FG-HK2mX0XhHTU8bUysLy-cKzovGmV3P/view?usp=sharing),
[coco_test_txtid2imgid.json](https://drive.google.com/file/d/14kgWYc2HcwzRf82bzuN_1DaIMys0WJaU/view?usp=drive_link)

&nbsp;&nbsp;&nbsp;&nbsp; Flickr:
[flickr_test.csv](https://drive.google.com/file/d/1R6iyG96GL3bHo84NTX5g8vVrUJgVAsPX/view?usp=sharing),
[flickr_test_txtid2imgid.json](https://drive.google.com/file/d/193u8zxKhbL8LnDtJ-qKmPv8xp9FnFnc9/view?usp=drive_link)

&nbsp;&nbsp;&nbsp;&nbsp; Occluded COCO:
[karpathy_test_cat_id_2_occ_img_negative_img.json](https://drive.google.com/file/d/1E4e-5dv41Vt6fKUDEzXQjLpWLdabqj2G/view?usp=drive_link)

&nbsp;&nbsp;&nbsp;&nbsp; ImageNet-R:
[imagenet-r-annfile.json](https://drive.google.com/file/d/1c90b0Fn3Lioa0WGbY3neUJd8MM6VAWYq/view?usp=drive_link),
[imagenet-r-cat2name.json](https://drive.google.com/file/d/10eeYPXohxf_qli3-fNPwzIo4EF6AOd-y/view?usp=drive_link)

### Step2. Download pretrained models
&nbsp;&nbsp;&nbsp;&nbsp; ELIP-C models:
[ELIP-C](https://drive.google.com/file/d/1BavChS8PaekpYawJoG25w2BCmkmgwSoF/view?usp=drive_link),
[ELIP-C(fine-tuned on COCO)](https://drive.google.com/file/d/129u5oCiG-w2SA1BGeZuD1DW3-85NW4H3/view?usp=drive_link),
[ELIP-C(fine-tuned on ImageNet)](https://drive.google.com/file/d/1Gg-EvqjrW-hx2j8eorB2-C1t63QLyADn/view?usp=drive_link)

&nbsp;&nbsp;&nbsp;&nbsp; ELIP-S models:
[ELIP-S](https://drive.google.com/file/d/1s5ciaAwOmxBEiVDJ-9wh2huMu5vOrIib/view?usp=drive_link),
[ELIP-S(fine-tuned on COCO)](https://drive.google.com/file/d/1gD60L4WtbwbcEgo7ICWte2QohvZmaYYQ/view?usp=drive_link),
[ELIP-S(fine-tuned on ImageNet)](https://drive.google.com/file/d/1YVPu5fA6PUM1f0zWeng_wXJIl4nwpny9/view?usp=drive_link)

&nbsp;&nbsp;&nbsp;&nbsp; ELIP-S2 models:
[ELIP-S-2](https://drive.google.com/file/d/1hDgYlMwRvC1DZTa689ILjjgCB1THpQcl/view?usp=drive_link),
[ELIP-S-2(fine-tuned on COCO)](https://drive.google.com/file/d/1lga5DHrakHP2ip04RKAXa7_0lsZ8z3w_/view?usp=drive_link),
[ELIP-S-2(fine-tuned on ImageNet)](https://drive.google.com/file/d/1QKyaz8I-7J_7vrmJYF6Y1oNh6kbf9Hlx/view?usp=drive_link)


### Step3. Download pre-computed baseline model features
&nbsp;&nbsp;&nbsp;&nbsp; CLIP features:
[COCO](https://drive.google.com/file/d/1dWrSnYOsLSbLUv4pjyxylRPcLwxuJ1zs/view?usp=drive_link), 
[Flickr](https://drive.google.com/file/d/1YvHT5wuXlGsuHyRSELME6RwDEwjzMHUb/view?usp=drive_link), 
[Occluded COCO](https://drive.google.com/file/d/1BXLZdI8xFR32hSilAdE6bytFlzfZdrCr/view?usp=drive_link), 
[ImageNet-R](https://drive.google.com/file/d/1ROHdlCnIL4aZ_HXOo7z89PyTjfQdZd9p/view?usp=drive_link)

&nbsp;&nbsp;&nbsp;&nbsp; SigLIP features:
[COCO](https://drive.google.com/file/d/15007D8F4i1_ZzpZbUrbFMBI9liOCf2am/view?usp=drive_link),
[Flickr](https://drive.google.com/file/d/1VX76_2csqh3stQQhuBlrNcKQ2uahIrLM/view?usp=drive_link),
[Occluded COCO](https://drive.google.com/file/d/1cSaVCYRyacrQRRO7GQF_DLYV3jhGuggF/view?usp=drive_link),
[ImageNet-R](https://drive.google.com/file/d/1Z1Q36n1mLtDRG3i7m0kKlkn-CtBd6Khj/view?usp=drive_link)

&nbsp;&nbsp;&nbsp;&nbsp; SigLIP2 features:
[COCO](https://drive.google.com/file/d/1wJuT0KhQ9t5vvQ2TU1VFcO91YAAj_sTO/view?usp=drive_link),
[Flickr](https://drive.google.com/file/d/1F709fG22MX58aF1GpE-NYsvMF16umQI5/view?usp=drive_link),
[Occluded COCO](https://drive.google.com/file/d/1-v9wCWvqcrq8DThn8aFVr_t2f09j14mX/view?usp=drive_link),
[ImageNet-R](https://drive.google.com/file/d/1WVaoo_FwnsX_MkOY3wtClTJ8K2VG6Isd/view?usp=drive_link)

### Step4. Organize the files

&nbsp;&nbsp;&nbsp;&nbsp; Save the downloaded annotation files, pre-computed feature files and model files in an organized manner:
```
ELIP-C/
│── standard_benchmarks/
│    │
│    ├── coco/                  
│    │   ├── clip_vitb16_standard_coco_infer_img_txt_features_10.24.pkl  
│    │   ├── siglip2G_standard_coco_infer_img_txt_features_2.26.pkl    
│    │   ├── siglipSO400M_standard_coco_infer_img_txt_features_2.26.pkl
│    │   ├── coco_test.csv
│    │   └── coco_test_txtid2imgid.json
│    │
│    ├── flickr/                    
│    │   └── ...
│    │
│    │── occ_coco/                   
│    │   └── ...
│    │
│    └── imagenet_r/                   
│        └── ...
│
└── elip_models/
     │
     ├── elip-c/
     │   ├── original/12.15_v2_2024_12_15-07_14_55-model_ViT-B-16-lr_0.001-b_20-j_8-p_amp-epoch_1.pt
     │   ├── coco_finetuned/12.15_v2_2025_01_13-13_31_24-model_ViT-B-16-lr_1e-05-b_20-j_8-p_amp-epoch_2.pt
     │   └── imagenet_finetuned/12.15_v2_2025_01_14-07_04_20-model_ViT-B-16-lr_1e-05-b_20-j_8-p_amp-epoch_2.pt
     │
     ├── elip-s/                    
     │   └── ...
     │
     └── elip-s2/                   
         └── ... 
```


### Step5. Run the corresponding scripts

&nbsp;&nbsp;&nbsp;&nbsp; Run the corresponding script according to different model and dataset. For example, to run the COCO evaluation of ELIP-C:

        sh eval_scripts/eval_elipc_coco.sh 0
