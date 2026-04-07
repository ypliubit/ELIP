gpuid=$1

cp -rf model_configs/ViT-B-16_tgvpt.json src/open_clip/model_configs/ViT-B-16.json


CUDA_VISIBLE_DEVICES=${gpuid} python -m training.main_eval_occluded_coco_retrieval_tgvpt \
    --val-data standard_benchmarks/coco/coco_test_new.csv \
    --resume elip_models/elip-c/coco_finetuned/12.15_v2_2025_01_13-13_31_24-model_ViT-B-16-lr_1e-05-b_20-j_8-p_amp-epoch_2.pt \
    --model ViT-B-16 \
    --pretrained datacomp_xl_s13b_b90k
