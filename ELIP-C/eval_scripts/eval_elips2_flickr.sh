gpuid=$1

CUDA_VISIBLE_DEVICES=${gpuid} python -m training.main_eval_tgvpt_reranking_standard_flickr \
    --val-data standard_benchmarks/flickr/flickr_test_new.csv \
    --resume elip_models/elip-s2/original/3.2_v6_2025_02_28-12_15_47-model_ViT-gopt-16-SigLIP2-384-lr_0.001-b_5-j_8-p_amp-epoch_1.pt \
    --model ViT-gopt-16-SigLIP2-384 \
    --pretrained 'webli' \
    --simple-sigliploss True