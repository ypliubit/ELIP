gpuid=$1

cp -rf model_configs/ViT-B-16_tgvpt.json src/open_clip/model_configs/ViT-B-16.json

CUDA_VISIBLE_DEVICES=${gpuid} python -m training.main_eval_tgvpt_reranking_standard_flickr \
    --val-data /home/ypliu/projects/OccludedCLIP/open_clip-main/attention_score/flickr_test_new.csv \
    --resume elip_models/elip-c/original/12.15_v2_2024_12_15-07_14_55-model_ViT-B-16-lr_0.001-b_20-j_8-p_amp-epoch_1.pt  \
    --model ViT-B-16 \
    --pretrained datacomp_xl_s13b_b90k