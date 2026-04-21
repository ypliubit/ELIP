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
[coco_test.csv](https://thor.robots.ox.ac.uk/elip/elip_c/step1/coco_test.csv),
[coco_test_txtid2imgid.json](https://thor.robots.ox.ac.uk/elip/elip_c/step1/coco_test_txtid2imgid.json)

&nbsp;&nbsp;&nbsp;&nbsp; Flickr:
[flickr_test.csv](https://thor.robots.ox.ac.uk/elip/elip_c/step1/flickr_test.csv),
[flickr_test_txtid2imgid.json](https://thor.robots.ox.ac.uk/elip/elip_c/step1/flickr_test_txtid2imgid.json)

&nbsp;&nbsp;&nbsp;&nbsp; Occluded COCO:
[karpathy_test_cat_id_2_occ_img_negative_img.json](https://thor.robots.ox.ac.uk/elip/elip_c/step1/karpathy_test_cat_id_2_occ_img_negative_img.json)

&nbsp;&nbsp;&nbsp;&nbsp; ImageNet-R:
[imagenet-r-annfile.json](https://thor.robots.ox.ac.uk/elip/elip_c/step1/imagenet-r-annfile.json),
[imagenet-r-cat2name.json](https://thor.robots.ox.ac.uk/elip/elip_c/step1/imagenet-r-cat2name.json)

### Step2. Download pretrained models
&nbsp;&nbsp;&nbsp;&nbsp; ELIP-C models:
[ELIP-C](https://thor.robots.ox.ac.uk/elip/elip_c/step2/12.15_v2_2024_12_15-07_14_55-model_ViT-B-16-lr_0.001-b_20-j_8-p_amp-epoch_1.pt),
[ELIP-C(fine-tuned on COCO)](https://thor.robots.ox.ac.uk/elip/elip_c/step2/12.15_v2_2025_01_13-13_31_24-model_ViT-B-16-lr_1e-05-b_20-j_8-p_amp-epoch_2.pt),
[ELIP-C(fine-tuned on ImageNet)](https://thor.robots.ox.ac.uk/elip/elip_c/step2/12.15_v2_2025_01_14-07_04_20-model_ViT-B-16-lr_1e-05-b_20-j_8-p_amp-epoch_2.pt)

&nbsp;&nbsp;&nbsp;&nbsp; ELIP-S models:
[ELIP-S](https://thor.robots.ox.ac.uk/elip/elip_c/step2/3.2_v5_2025_02_27-23_40_38-model_ViT-SO400M-14-SigLIP-384-lr_0.001-b_5-j_8-p_amp-epoch_1.pt),
[ELIP-S(fine-tuned on COCO)](https://thor.robots.ox.ac.uk/elip/elip_c/step2/3.2_v5_2025_03_04-20_44_20-model_ViT-SO400M-14-SigLIP-384-lr_1e-05-b_5-j_8-p_amp-epoch_2.pt),
[ELIP-S(fine-tuned on ImageNet)](https://thor.robots.ox.ac.uk/elip/elip_c/step2/3.2_v5_2025_03_04-17_40_12-model_ViT-SO400M-14-SigLIP-384-lr_1e-05-b_5-j_8-p_amp-epoch_2.pt)

&nbsp;&nbsp;&nbsp;&nbsp; ELIP-S2 models:
[ELIP-S-2](https://thor.robots.ox.ac.uk/elip/elip_c/step2/3.2_v6_2025_02_28-12_15_47-model_ViT-gopt-16-SigLIP2-384-lr_0.001-b_5-j_8-p_amp-epoch_1.pt),
[ELIP-S-2(fine-tuned on COCO)](https://thor.robots.ox.ac.uk/elip/elip_c/step2/3.2_v6_2025_03_04-20_44_51-model_ViT-gopt-16-SigLIP2-384-lr_1e-05-b_5-j_8-p_amp-epoch_2.pt),
[ELIP-S-2(fine-tuned on ImageNet)](https://thor.robots.ox.ac.uk/elip/elip_c/step2/3.2_v6_2025_03_04-17_40_13-model_ViT-gopt-16-SigLIP2-384-lr_1e-05-b_5-j_8-p_amp-epoch_2.pt)


### Step3. Download pre-computed baseline model features
&nbsp;&nbsp;&nbsp;&nbsp; CLIP features:
[COCO](https://thor.robots.ox.ac.uk/elip/elip_c/step3/clip_vitb16_standard_coco_infer_img_txt_features_10.24.pkl), 
[Flickr](https://thor.robots.ox.ac.uk/elip/elip_c/step3/clip_vitb16_standard_flickr_infer_img_txt_features_10.24.pkl), 
[Occluded COCO](https://thor.robots.ox.ac.uk/elip/elip_c/step3/occluded_coco_clip_feat_revised_11.7.pkl), 
[ImageNet-R](https://thor.robots.ox.ac.uk/elip/elip_c/step3/imagenet-r_clip_feat.pkl)

&nbsp;&nbsp;&nbsp;&nbsp; SigLIP features:
[COCO](https://thor.robots.ox.ac.uk/elip/elip_c/step3/siglipSO400M_standard_coco_infer_img_txt_features_2.26.pkl),
[Flickr](https://thor.robots.ox.ac.uk/elip/elip_c/step3/siglipSO400M_standard_flickr_infer_img_txt_features_2.26.pkl),
[Occluded COCO](https://thor.robots.ox.ac.uk/elip/elip_c/step3/occluded_coco_revised_siglipSO_feat.pkl),
[ImageNet-R](https://thor.robots.ox.ac.uk/elip/elip_c/step3/imagenet-r_siglipSO_feat.pkl)

&nbsp;&nbsp;&nbsp;&nbsp; SigLIP2 features:
[COCO](https://thor.robots.ox.ac.uk/elip/elip_c/step3/siglip2G_standard_coco_infer_img_txt_features_2.26.pkl),
[Flickr](https://thor.robots.ox.ac.uk/elip/elip_c/step3/siglip2G_standard_flickr_infer_img_txt_features_2.26.pkl),
[Occluded COCO](https://thor.robots.ox.ac.uk/elip/elip_c/step3/occluded_coco_revised_siglip2G_feat.pkl),
[ImageNet-R](https://thor.robots.ox.ac.uk/elip/elip_c/step3/imagenet-r_siglip2G_feat.pkl)

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
