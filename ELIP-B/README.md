## Dependencies
Please run
    
        conda env create -f environment.yml
		conda activate lavis

to install the dependencies.

Replace `~/anaconda3/envs/lavis/lib/python3.8/site-packages/torch/utils/data/distributed.py` and `~/anaconda3/envs/lavis/lib/python3.8/site-packages/torch/utils/data/__init__.py` with the files as in the folder `ELIPB_env_code`.

### Training data
The csv file containing training data caption and urls can be found in [this link](https://drive.google.com/file/d/1eyaRpy82v-7MvFDfRkRtW50oe-N5crOL/view?usp=drive_link).

## Evaluation

### Step1. Download dataset annotations

&nbsp;&nbsp;&nbsp;&nbsp; Occluded COCO:
[karpathy_test_cat_id_2_occ_img_negative_img.json](https://drive.google.com/file/d/1E4e-5dv41Vt6fKUDEzXQjLpWLdabqj2G/view?usp=drive_link)

&nbsp;&nbsp;&nbsp;&nbsp; ImageNet-R:
[imagenet-r-annfile.json](https://drive.google.com/file/d/1c90b0Fn3Lioa0WGbY3neUJd8MM6VAWYq/view?usp=drive_link),
[imagenet-r-cat2name.json](https://drive.google.com/file/d/10eeYPXohxf_qli3-fNPwzIo4EF6AOd-y/view?usp=drive_link)

### Step2. Download pretrained models
&nbsp;&nbsp;&nbsp;&nbsp; ELIP-B models:
[ELIP-B](https://drive.google.com/file/d/1XGTAlsZzneeudCCFSlEXmZWMNNzg-hgI/view?usp=sharing)


### Step3. Organize the files

&nbsp;&nbsp;&nbsp;&nbsp; Save the downloaded annotation files, pre-computed feature files and model files in an organized manner:
```
ELIP-B/
│── standard_benchmarks/
│    │
│    ├── occ_coco/                  
│    │   └── karpathy_test_cat_id_2_occ_img_negative_img.json
│    │
│    └── imagenet_r/       
│        ├── imagenet-r-cat2name.json             
│        └── imagenet-r-annfile.json
│
└── elip_models/
     │
     └── elip-b/
         ├── original/full_model_iccv_v27-20241229044-checkpoint_0.pth
         ├── coco_finetuned/
         └── imagenet_finetuned/
```


### Step4. Run the corresponding scripts

&nbsp;&nbsp;&nbsp;&nbsp; Run the corresponding script according to different model and dataset. For example, to run the COCO evaluation of ELIP-B:

        sh eval_scripts/eval_ret_coco_multi_prompt3.sh 0 elip_models/elip-b/original/full_model_iccv_v27-20241229044-checkpoint_0.pth
