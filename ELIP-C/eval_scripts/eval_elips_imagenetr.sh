gpuid=$1

# additional occluded coco
CUDA_VISIBLE_DEVICES=${gpuid} python -m training.main_eval_imagenet_r_retrieval_tgvpt \
    --val-data standard_benchmarks/coco/coco_test_new.csv \
    --resume elip_models/elip-s/original/3.2_v5_2025_02_27-23_40_38-model_ViT-SO400M-14-SigLIP-384-lr_0.001-b_5-j_8-p_amp-epoch_1.pt \
    --model ViT-SO400M-14-SigLIP-384 \
    --pretrained 'webli' \
    --simple-sigliploss True