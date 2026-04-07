gpu=$1

CUDA_VISIBLE_DEVICES=$gpu python -m torch.distributed.run --nproc_per_node=1 --master_port=2958$gpu evaluate.py --cfg-path lavis/projects/blip2/eval/ret_flickr_eval_multi_prompt3.yaml --ckpt $2
# python -m torch.distributed.run --nproc_per_node=16 evaluate.py --cfg-path lavis/projects/blip2/eval/ret_coco_eval.yaml
