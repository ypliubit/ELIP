gpuid=$1

CUDA_VISIBLE_DEVICES=${gpuid} python -m training.main_eval_occluded_coco_retrieval_tgvpt \
    --val-data standard_benchmarks/coco/coco_test_new.csv \
    --resume elip_models/elip-s/coco_finetuned/3.2_v5_2025_03_04-20_44_20-model_ViT-SO400M-14-SigLIP-384-lr_1e-05-b_5-j_8-p_amp-epoch_2.pt \
    --model ViT-SO400M-14-SigLIP-384 \
    --pretrained 'webli' \
    --simple-sigliploss True
