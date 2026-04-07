gpuid=$1

CUDA_VISIBLE_DEVICES=${gpuid} python -m training.main_eval_occluded_coco_retrieval_tgvpt \
    --val-data standard_benchmarks/flickr/flickr_test_new.csv \
    --resume elip_models/elip-s2/coco_finetuned/3.2_v6_2025_03_04-20_44_51-model_ViT-gopt-16-SigLIP2-384-lr_1e-05-b_5-j_8-p_amp-epoch_2.pt \
    --model ViT-gopt-16-SigLIP2-384 \
    --pretrained 'webli' \
    --simple-sigliploss True
