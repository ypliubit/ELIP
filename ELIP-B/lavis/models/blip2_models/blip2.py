"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib
import logging
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

import lavis.common.dist_utils as dist_utils
from lavis.common.dist_utils import download_cached_file
from lavis.common.utils import is_url
from lavis.common.logger import MetricLogger
from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel, BertLMHeadModel2, BertLMHeadModel2org
from lavis.models.eva_vit import create_deepvpt_eva_vit_g, create_eva_vit_g, create_turbo_eva_vit_g
from lavis.models.clip_vit import create_clip_vit_L
from transformers import BertTokenizer

import ipdb
from tqdm import tqdm
import numpy as np

# # Register hooks to monitor gradients
# def hook_fn(grad):
#     print("Gradient shape:", grad.shape)




class Blip2Base(BaseModel):
    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @classmethod
    def init_Qformer2_org(cls, num_query_token, vision_width, cross_attention_freq=2):        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers += 2
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel2org.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        # for name, param in Qformer.named_parameters():
        #     # if name.split('.')[0] in ['12', '13']: 
        #     param.register_hook(lambda grad, name=name: print(f"Gradient for {name}: {grad.shape} {grad[0]}"))

        return Qformer, query_tokens


    def init_vision_encoder(
        self, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision
    ):
        assert model_name in [
            "eva_clip_g",
            "eva2_clip_L",
            "clip_L",
            "deepvpt_eva_clip_g",
            "turbo_eva_clip_g",
        ], "vit model must be eva_clip_g, eva2_clip_L or clip_L"
        if model_name == "eva_clip_g":
            visual_encoder = create_eva_vit_g(
                img_size, drop_path_rate, use_grad_checkpoint, precision
            )
        elif model_name == "deepvpt_eva_clip_g":
            visual_encoder = create_deepvpt_eva_vit_g(
                img_size, drop_path_rate, use_grad_checkpoint, precision
            )
        elif model_name == "turbo_eva_clip_g":
            visual_encoder = create_turbo_eva_vit_g(
                img_size, drop_path_rate, use_grad_checkpoint, precision
            )
#         elif model_name == "eva2_clip_L":
#             visual_encoder = create_eva2_vit_L(
#                 img_size, drop_path_rate, use_grad_checkpoint, precision
#             )
        elif model_name == "clip_L":
            visual_encoder = create_clip_vit_L(img_size, use_grad_checkpoint, precision)
        ln_vision = LayerNorm(visual_encoder.num_features)
        self.vit_name = model_name
        return visual_encoder, ln_vision

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    def get_optimizer_params(self, weight_decay, lr_scale=1):

        vit_num_layers = self.visual_encoder.get_num_layer()
        lr_scales = list(lr_scale ** (vit_num_layers + 1 - i) for i in range(vit_num_layers + 2))

        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias"):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay
            if 'visual_encoder' in name:
                layer_id = self.visual_encoder.get_num_layer(name.replace('visual_encoder.',''))
                group_name = "vit_layer_%d_%s" % (layer_id, group_name)
            else:
                layer_id = None

            if group_name not in parameter_group_names:
                if layer_id is not None:
                    scale = lr_scales[layer_id]
                else:
                    scale = 1
                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)
        import json
        print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
        # ipdb.set_trace()
        optim_params = list(parameter_group_vars.values())
        return optim_params

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


def compute_sim_matrix(model, data_loader, **kwargs):


    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        print(i)
        text = texts[i : min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)
        text_feat = model.forward_text(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    topk_text = 25
    vit_feats = []
    image_embeds = []
    iter_image_i = 0
    for samples in data_loader:
        print(iter_image_i)
        iter_image_i += 1
        image = samples["image"]

        image = image.to(model.device)
        image_feat, vit_feat = model.forward_image(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)

        vit_feats.append(vit_feat.cpu())
        image_embeds.append(image_embed)

    vit_feats = torch.cat(vit_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)
    # cur_image_embeds_list = []    for image_embed in image_embeds:
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)




    # score_matrix_i2t = torch.full(
    #     (len(data_loader.dataset.image), len(texts)), -100.0
    # ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    # step = sims_matrix.size(0) // num_tasks + 1
    # start = rank * step
    # end = min(sims_matrix.size(0), start + step)

    # for i, sims in enumerate(
    #     metric_logger.log_every(sims_matrix[start:end], 50, header)
    # ):
    #     topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
    #     image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device)
    #     score = model.compute_itm(
    #         image_inputs=image_inputs,
    #         text_ids=text_ids[topk_idx],
    #         text_atts=text_atts[topk_idx],
    #     ).float()
    #     score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (topk_text, len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[start + i].repeat(k_test, 1),
            text_atts=text_atts[start + i].repeat(k_test, 1),
        ).float()
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        # torch.distributed.all_reduce(
        #     score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        # )
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    # return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
    return score_matrix_t2i.cpu().numpy(), score_matrix_t2i.cpu().numpy()

def compute_sim_matrix_itm_rerank(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")
    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()
    
    k_test = 20
    # -------------------------------------- get text features --------------------------------------
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    # text_embeds_org = []
    text_atts = []
    logging.info("Get text features.")
    for i in tqdm(range(0, num_text, text_bs)):
        # print(i)
        text = texts[i: min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)
        # text_feat, text_feat_org = model.forward_text_new(text_input)
        text_feat = model.forward_text(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
        # text_embed_org = F.normalize(model.text_proj(text_feat_org))
        text_embeds.append(text_embed)
        # text_embeds_org.append(text_embed_org)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    # text_embeds_org = torch.cat(text_embeds_org, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    # -------------------------------------- get image features --------------------------------------
    vit_feats = []
    image_embeds = []
    iter_image_i = 0
    images = []
    logging.info("Get image features.")
    for samples in tqdm(data_loader):
        iter_image_i += 1
        image = samples["image"]

        image = image.to(model.device)
        image_feat, vit_feat = model.forward_image(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)

        vit_feats.append(vit_feat.cpu())
        image_embeds.append(image_embed)
        images.append(image.cpu())

    vit_feats = torch.cat(vit_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)
    images = torch.cat(images, dim=0)

    # -------------------------------------- get similarity matrix --------------------------------------
    sims_matrix = []
    for image_embed in image_embeds:
        # sim_q2t = image_embed @ text_embeds_org.t()
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    # -------------------------------------- reranking --------------------------------------
    for i, sims in enumerate(
            metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        # original
        # topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        # image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        # score = model.compute_itm(
        #     image_inputs=image_inputs,
        #     text_ids=text_ids[start + i].repeat(k_test, 1),
        #     text_atts=text_atts[start + i].repeat(k_test, 1),
        # ).float()
        # score_matrix_t2i[start + i, topk_idx] = score + topk_sim

        # new
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        # image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        image_inputs = images[topk_idx.cpu()].to(model.device)
        score = model.compute_itm_tgvpt(
            image_inputs=image_inputs,
            text_ids=text_ids[start + i].repeat(k_test, 1),
            text_atts=text_atts[start + i].repeat(k_test, 1),
            text_embeds=text_embeds[start + i].repeat(k_test, 1),
        ).float()
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim
        # if i == 0:
        #     print(score_matrix_t2i[start + i])
        #     break

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    # return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
    return score_matrix_t2i.cpu().numpy(), score_matrix_t2i.cpu().numpy()


def compute_sim_matrix_itm_rerank_multi_prompt(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")
    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()
    
    k_test = 20
    # -------------------------------------- get text features --------------------------------------
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    logging.info("Get text features.")
    for i in tqdm(range(0, num_text, text_bs)):
        # print(i)
        text = texts[i: min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)
        text_feat = model.forward_text(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    # -------------------------------------- get image features --------------------------------------
    vit_feats = []
    image_embeds = []
    iter_image_i = 0
    images = []
    logging.info("Get image features.")
    for samples in tqdm(data_loader):
        iter_image_i += 1
        image = samples["image"]

        image = image.to(model.device)
        image_feat, vit_feat = model.forward_image(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)

        vit_feats.append(vit_feat.cpu())
        image_embeds.append(image_embed)
        images.append(image.cpu())

    vit_feats = torch.cat(vit_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)
    images = torch.cat(images, dim=0)

    # -------------------------------------- get similarity matrix --------------------------------------
    sims_matrix = []
    for image_embed in image_embeds:
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    # -------------------------------------- reranking --------------------------------------
    for i, sims in enumerate(
            metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        # original
        # topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        # image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        # score = model.compute_itm(
        #     image_inputs=image_inputs,
        #     text_ids=text_ids[start + i].repeat(k_test, 1),
        #     text_atts=text_atts[start + i].repeat(k_test, 1),
        # ).float()
        # score_matrix_t2i[start + i, topk_idx] = score + topk_sim

        # new
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        # image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        image_inputs = images[topk_idx.cpu()].to(model.device)
        score = model.compute_itm_tgvpt(
            image_inputs=image_inputs,
            text_ids=text_ids[start + i].repeat(k_test, 1),
            text_atts=text_atts[start + i].repeat(k_test, 1),
            text_embeds=text_embeds[start + i].repeat(k_test, 1),
        ).float()
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim
        # if i == 0:
        #     print(score_matrix_t2i[start + i])
        #     break

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    # return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
    return score_matrix_t2i.cpu().numpy(), score_matrix_t2i.cpu().numpy()


def compute_sim_matrix_itm_rerank_multi_prompt3(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")
    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()
    
    k_test = 20
    # -------------------------------------- get text features --------------------------------------
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_embeds_new = []
    text_atts = []
    logging.info("Get text features.")
    for i in tqdm(range(0, num_text, text_bs)):
        # print(i)
        text = texts[i: min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)
        text_feat = model.forward_text_org(text_input)
        text_feat_new = model.forward_text(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
        text_embed_new = F.normalize(model.text_proj(text_feat_new))
        text_embeds.append(text_embed)
        text_embeds_new.append(text_embed_new)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_embeds_new = torch.cat(text_embeds_new, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    # -------------------------------------- get image features --------------------------------------
    vit_feats = []
    image_embeds = []
    iter_image_i = 0
    images = []
    logging.info("Get image features.")
    for samples in tqdm(data_loader):
        iter_image_i += 1
        image = samples["image"]

        image = image.to(model.device)
        image_feat, vit_feat = model.forward_image(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)

        vit_feats.append(vit_feat.cpu())
        image_embeds.append(image_embed)
        images.append(image.cpu())

    vit_feats = torch.cat(vit_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)
    images = torch.cat(images, dim=0)

    # -------------------------------------- get similarity matrix --------------------------------------
    sims_matrix = []
    for image_embed in image_embeds:
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    # -------------------------------------- reranking --------------------------------------
    for i, sims in enumerate(
            metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        # original
        # topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        # image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        # score = model.compute_itm(
        #     image_inputs=image_inputs,
        #     text_ids=text_ids[start + i].repeat(k_test, 1),
        #     text_atts=text_atts[start + i].repeat(k_test, 1),
        # ).float()
        # score_matrix_t2i[start + i, topk_idx] = score + topk_sim

        # new
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        # image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        image_inputs = images[topk_idx.cpu()].to(model.device)
        score = model.compute_itm_tgvpt(
            image_inputs=image_inputs,
            text_ids=text_ids[start + i].repeat(k_test, 1),
            text_atts=text_atts[start + i].repeat(k_test, 1),
            text_embeds=text_embeds_new[start + i].repeat(k_test, 1),
        ).float()
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim
        # if i == 0:
        #     print(score_matrix_t2i[start + i])
        #     break

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    # return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
    return score_matrix_t2i.cpu().numpy(), score_matrix_t2i.cpu().numpy()




def compute_sim_matrix_old(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")
    k_test = 10

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    logging.info("Get text features.")
    for i in tqdm(range(0, num_text, text_bs)):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)
        text_feat = model.forward_text(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    vit_feats = []
    image_embeds = []
    logging.info("Get image features.")
    for samples in tqdm(data_loader):
        image = samples["image"]

        image = image.to(model.device)
        image_feat, vit_feat = model.forward_image(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)

        vit_feats.append(vit_feat.cpu())
        image_embeds.append(image_embed)

    vit_feats = torch.cat(vit_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = []
    for image_embed in image_embeds:
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)

    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    logging.info("Reranking for image to text.")
    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[topk_idx],
            text_atts=text_atts[topk_idx],
        ).float()
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    logging.info("Reranking for text to image.")
    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[start + i].repeat(k_test, 1),
            text_atts=text_atts[start + i].repeat(k_test, 1),
        ).float()
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim
        # if i == 0:
        #     print(score_matrix_t2i[start + i])
        #     break

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


def compute_sim_matrix_no_rerank(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")
    k_test = 10

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    logging.info("Get text features.")
    for i in tqdm(range(0, num_text, text_bs)):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)
        text_feat = model.forward_text(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    vit_feats = []
    image_embeds = []
    logging.info("Get image features.")
    for samples in tqdm(data_loader):
        image = samples["image"]

        image = image.to(model.device)
        image_feat, vit_feat = model.forward_image(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)

        vit_feats.append(vit_feat.cpu())
        image_embeds.append(image_embed)

    vit_feats = torch.cat(vit_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = []
    for image_embed in image_embeds:
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)

    # score_matrix_i2t = sims_matrix
    # score_matrix_i2t = torch.full(
    #     (len(data_loader.dataset.image), len(texts)), -100.0
    # ).to(model.device)

    # num_tasks = dist_utils.get_world_size()
    # rank = dist_utils.get_rank()
    # step = sims_matrix.size(0) // num_tasks + 1
    # start = rank * step
    # end = min(sims_matrix.size(0), start + step)
    #
    # logging.info("Reranking for image to text.")
    # for i, sims in enumerate(
    #     metric_logger.log_every(sims_matrix[start:end], 50, header)
    # ):
    #     topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
    #     image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device)
    #     score = model.compute_itm(
    #         image_inputs=image_inputs,
    #         text_ids=text_ids[topk_idx],
    #         text_atts=text_atts[topk_idx],
    #     ).float()
    #     score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    # sims_matrix = sims_matrix.t()
    # score_matrix_t2i = torch.full(
    #     (len(texts), len(data_loader.dataset.image)), -100.0
    # ).to(model.device)
    #
    # step = sims_matrix.size(0) // num_tasks + 1
    # start = rank * step
    # end = min(sims_matrix.size(0), start + step)
    #
    # logging.info("Reranking for text to image.")
    # for i, sims in enumerate(
    #     metric_logger.log_every(sims_matrix[start:end], 50, header)
    # ):
    #     topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
    #     image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
    #     score = model.compute_itm(
    #         image_inputs=image_inputs,
    #         text_ids=text_ids[start + i].repeat(k_test, 1),
    #         text_atts=text_atts[start + i].repeat(k_test, 1),
    #     ).float()
    #     score_matrix_t2i[start + i, topk_idx] = score + topk_sim
    #
    # if dist_utils.is_dist_avail_and_initialized():
    #     dist.barrier()
    #     torch.distributed.all_reduce(
    #         score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
    #     )
    #     torch.distributed.all_reduce(
    #         score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
    #     )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    # return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
    return sims_matrix.cpu().numpy(), sims_matrix.t().cpu().numpy()



def compute_sim_matrix_itm_rerank_withloss2(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")
    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    k_test = 20
    # -------------------------------------- get text features --------------------------------------
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    logging.info("Get text features.")
    for i in tqdm(range(0, num_text, text_bs)):
        # print(i)
        text = texts[i: min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)
        text_feat = model.forward_text(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    # -------------------------------------- get image features --------------------------------------
    vit_feats = []
    image_embeds = []
    iter_image_i = 0
    images = []
    logging.info("Get image features.")
    for samples in tqdm(data_loader):
        iter_image_i += 1
        image = samples["image"]

        image = image.to(model.device)
        image_feat, vit_feat = model.forward_image(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)

        vit_feats.append(vit_feat.cpu())
        image_embeds.append(image_embed)
        images.append(image.cpu())

    vit_feats = torch.cat(vit_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)
    images = torch.cat(images, dim=0)

    # -------------------------------------- get similarity matrix --------------------------------------
    sims_matrix = []
    for image_embed in image_embeds:
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    # -------------------------------------- reranking --------------------------------------
    txt2img = data_loader.dataset.txt2img
    for i, sims in enumerate(
            metric_logger.log_every(sims_matrix[start:end], 1, header)
    ):
        # new
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        img_idx = txt2img[i]
        itm_labels = (topk_idx==img_idx).long()
        image_inputs = images[topk_idx.cpu()].to(model.device)
        score = model.compute_itm_tgvpt_withloss(
            image_inputs=image_inputs,
            text_ids=text_ids[start + i].repeat(k_test, 1),
            text_atts=text_atts[start + i].repeat(k_test, 1),
            text_embeds=text_embeds[start + i].repeat(k_test, 1),
            itm_labels=itm_labels
        ).float()
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim
        # if i == 0:
        #     print(score_matrix_t2i[start + i])
        #     break

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    # return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
    return score_matrix_t2i.cpu().numpy(), score_matrix_t2i.cpu().numpy()

    
def compute_sim_matrix_occluded_coco_tgvpt2(model, data_loader, **kwargs):
    import json
    import pickle

    # load annotation
    occluded_coco_retrieval_ann_file = json.load(
        open('DATASET_PATH/blip2_ypliu_coco2017val_cat_id_2_occ_img_negative_img.json'))
    cat_id_2_occ_img = occluded_coco_retrieval_ann_file['cat_id_2_occ_img']
    cat_id_2_negative_img = occluded_coco_retrieval_ann_file['cat_id_2_negative_img']
    img_id_2_all_ins_id = occluded_coco_retrieval_ann_file['img_id_2_all_ins_id']

    logging.info("Computing features for evaluation...")
    # ----------------------------- text features for evaluation -----------------------------
    coco_91 = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush']
    #
    # logging.info("Get text features.")
    # cat_id_2_txt_feat = {}
    # cat_id_2_txt_id = {}
    # cat_id_2_txt_atts = {}
    # iter_txt = 0
    # for cat_id in cat_id_2_occ_img.keys():
    #     iter_txt += 1
    #     cur_txt = coco_91[int(cat_id)]
    #     text_input = model.tokenizer(
    #         cur_txt,
    #         padding="max_length",
    #         truncation=True,
    #         max_length=35,
    #         return_tensors="pt",
    #     ).to(model.device)
    #     text_feat = model.forward_text(text_input)
    #     text_embed = F.normalize(model.text_proj(text_feat))
    #     cat_id_2_txt_feat[cat_id] = text_embed.detach().cpu().numpy()
    #     cat_id_2_txt_id[cat_id] = text_input.input_ids.detach().cpu().numpy()
    #     cat_id_2_txt_atts[cat_id] = text_input.attention_mask.detach().cpu().numpy()

    # ----------------------------- image features for evaluation -----------------------------
    from torchvision import transforms as pth_transforms
    from PIL import Image
    img_preprocess = pth_transforms.Compose([
        pth_transforms.Resize([364, 364], pth_transforms.InterpolationMode.BICUBIC),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    # logging.info("Get image features.")
    #
    # img_id_2_vit_feat = {}
    # img_id_2_img_feat = {}
    img_folder = 'DATASET_PATH/imagenet-a'
    # iter_img = 0
    # for img_id in tqdm(img_id_2_all_ins_id.keys()):
    #     iter_img += 1
    #     cur_image = img_preprocess(Image.open(os.path.join(img_folder, str(img_id).rjust(12,'0')+'.jpg')).convert('RGB')).unsqueeze(0)
    #     cur_image = cur_image.to(device=model.device, non_blocking=True)
    #     image_feat, vit_feat = model.forward_image(cur_image)
    #     image_embed = model.vision_proj(image_feat)
    #     image_embed = F.normalize(image_embed, dim=-1)
    #     img_id_2_vit_feat[img_id] = vit_feat.detach().cpu().numpy()
    #     img_id_2_img_feat[img_id] = image_embed.detach().cpu().numpy()
    #
    #     pickle.dump({'cat_id_2_txt_feat': cat_id_2_txt_feat
    #                     , 'cat_id_2_txt_id': cat_id_2_txt_id
    #                     , 'cat_id_2_txt_atts': cat_id_2_txt_atts
    #                     , 'img_id_2_vit_feat': img_id_2_vit_feat
    #                     , 'img_id_2_img_feat': img_id_2_img_feat}, dump_f)
    cur_pkl = pickle.load(open('DATASET_PATH/blip2_ypliu_occluded_coco_clip_feat.pkl', 'rb'))
    cat_id_2_txt_feat = cur_pkl['cat_id_2_txt_feat']
    cat_id_2_txt_id = cur_pkl['cat_id_2_txt_id']
    cat_id_2_txt_atts = cur_pkl['cat_id_2_txt_atts']
    img_id_2_vit_feat = cur_pkl['img_id_2_vit_feat']
    img_id_2_img_feat = cur_pkl['img_id_2_img_feat']

    # ----------------------------- similarity matrix for evaluation -----------------------------
    recall_cnt = 0
    ap_sum = 0
    topk = 100  # topK in reranking
    for cat_id in tqdm(cat_id_2_txt_feat.keys()):
        if cat_id == '1':
            continue
        print(coco_91[int(cat_id)])
        # if coco_91[int(cat_id)] in ['bicycle', 'toilet', 'motorcycle', 'potted plant', 'bench', 'sink']:
        #     continue
        # if coco_91[int(cat_id)] not in ['toilet']:
        #     continue
        cur_txt_feat = torch.from_numpy(cat_id_2_txt_feat[cat_id]).to(model.device)
        logit_list = []
        occ_img_id_list = cat_id_2_occ_img[cat_id]
        neg_img_id_list = cat_id_2_negative_img[cat_id]
        for img_id in occ_img_id_list:
            cur_img_feat = torch.from_numpy(img_id_2_img_feat[str(img_id)]).to(model.device)
            cur_logit = torch.max(cur_img_feat @ cur_txt_feat.T).cpu().item()
            logit_list.append([cur_logit, 1, img_id])
        for img_id in neg_img_id_list:
            cur_img_feat = torch.from_numpy(img_id_2_img_feat[str(img_id)]).to(model.device)
            cur_logit = torch.max(cur_img_feat @ cur_txt_feat.T).cpu().item()
            logit_list.append([cur_logit, 0, img_id])
        logit_list.sort(key=lambda x: -x[0])

        # reranking
        num_img = min(topk, len(logit_list))


        # # stat num_positive, if lower than 20, skip
        # pre_pos_cnt = 0
        # for top_i in range(num_img):
        #     if logit_list[top_i][1] == 1:
        #         pre_pos_cnt += 1
        # if pre_pos_cnt > 20:

        # num_img = new_input_list.shape[0]
        all_score = []
        bsz = 20
        for i in tqdm(range(0, num_img, bsz)):
            # print(i)
            new_input_list = []  # expected BS=100
            for top_i in range(i, min(num_img, i + bsz)):
                img_id = logit_list[top_i][-1]
                cur_image = img_preprocess(Image.open(os.path.join(img_folder, str(img_id).rjust(12, '0') + '.jpg')).convert('RGB')).unsqueeze(0)
                cur_image = cur_image.to(device=model.device, non_blocking=True)
                new_input_list.append(cur_image)

            new_input_list = torch.cat(new_input_list, dim=0)  # 100 x 3 x 224 x 224
            # ipdb.set_trace()
            # print(new_input_list.shape)

            num_input = new_input_list.shape[0]
            score = model.compute_itm_tgvpt(
                image_inputs=new_input_list,
                text_ids=torch.from_numpy(cat_id_2_txt_id[cat_id]).to(model.device).repeat(num_input, 1),
                text_atts=torch.from_numpy(cat_id_2_txt_atts[cat_id]).to(model.device).repeat(num_input, 1),
                text_embeds=cur_txt_feat[0].repeat(num_input, 1),
            ).float()
            all_score.append(score)
            # ipdb.set_trace()
            # print(score.shape)

        all_score = torch.cat(all_score, dim=0)
        org_ap = calculate_ap(logit_list)
        print('org_ap: ', org_ap)
        pos_ave_score = 0
        neg_ave_score = 0
        pos_cnt = 0
        neg_cnt = 0
        # ipdb.set_trace()
        for top_i in range(num_img):
            logit_list[top_i][0] += all_score[top_i].item() * 0.05
            if logit_list[top_i][1] == 1:
                pos_ave_score += all_score[top_i].item()
                pos_cnt += 1
            else:
                neg_ave_score += all_score[top_i].item()
                neg_cnt += 1
        print('pos_ave: ', pos_ave_score / pos_cnt)
        print('neg_ave: ', neg_ave_score / neg_cnt)
        print('pos_cnt: ', pos_cnt)
        print('neg_cnt: ', neg_cnt)
        for top_i in range(num_img, len(logit_list)):
            logit_list[top_i][0] = logit_list[top_i][0] - 1000

        cur_ap = calculate_ap(logit_list)
        print('ap_after_rerank: ', cur_ap)
        ap_sum += cur_ap
        recall_cnt += 1

        # if cur_ap < org_ap:
        #     ipdb.set_trace()

    print('mAP: ', ap_sum / recall_cnt)

    return None



def compute_sim_matrix_imagenet_a_old(model, data_loader, **kwargs):
    import json
    import pickle

    cat_id_2_cat_name = json.load(open('DATASET_PATH/imagenet-a-cat2name.json', 'r'))
    occluded_coco_retrieval_ann_file = json.load(open('DATASET_PATH/imagenet-a-annfile.json'))

    # ----------------------------- similarity matrix for evaluation -----------------------------
    cal_recall = True
    rec_sum = 0
    rec_cnt = 0
    recall_cnt = 0
    ap_sum = 0
    topk = 200  # topK in reranking
    all_logit_list = {}
    for cat_id in tqdm(cat_id_2_txt_feat.keys()):
        if cat_id == '1':
            continue
        # print(coco_91[int(cat_id)])
        # cur_txt_feat = torch.from_numpy(cat_id_2_txt_feat[cat_id]).to(model.device)
        cur_txt_feat = cat_id_2_txt_feat[cat_id]
        logit_list = []
        occ_img_id_list = cat_id_2_occ_img[cat_id]
        neg_img_id_list = cat_id_2_negative_img[cat_id]
        for img_id in occ_img_id_list:
            # cur_img_feat = torch.from_numpy(img_id_2_img_feat[str(img_id)]).to(model.device)
            # cur_logit = torch.max(cur_img_feat @ cur_txt_feat.T).cpu().item()
            cur_img_feat = img_id_2_img_feat[str(img_id)]
            cur_logit = np.max(np.dot(cur_img_feat, cur_txt_feat.T))
            logit_list.append([cur_logit, 1, img_id])
        for img_id in neg_img_id_list:
            # cur_img_feat = torch.from_numpy(img_id_2_img_feat[str(img_id)]).to(model.device)
            # cur_logit = torch.max(cur_img_feat @ cur_txt_feat.T).cpu().item()
            cur_img_feat = img_id_2_img_feat[str(img_id)]
            cur_logit = np.max(np.dot(cur_img_feat, cur_txt_feat.T))
            logit_list.append([cur_logit, 0, img_id])
        logit_list.sort(key=lambda x: -x[0])

        # reranking
        num_img = min(topk, len(logit_list))
        # num_img = new_input_list.shape[0]
        all_score = []

        if cal_recall:
            if len(occ_img_id_list):
                rec_n = 0
                for i in range(num_img):
                    if logit_list[i][1] == 1:
                        rec_n += 1
                rec = rec_n/len(occ_img_id_list)
                print(cat_id, f' top{topk} recall is ', rec)
                rec_sum += rec
                rec_cnt += 1
            else:
                print(cat_id, ' no images')
        bsz = 20
        for i in tqdm(range(0, num_img, bsz)):
            new_input_list = []  # expected BS=100
            for top_i in range(i, min(num_img, i + bsz)):
                img_id = logit_list[top_i][-1]
                new_input_list.append(torch.from_numpy(img_id_2_vit_feat[str(img_id)]).to(model.device))
            new_input_list = torch.cat(new_input_list, dim=0)  # 100 x 3 x 224 x 224

            num_input = new_input_list.shape[0]
            score = model.compute_itm(
                image_inputs=new_input_list,
                text_ids=torch.from_numpy(cat_id_2_txt_id[cat_id]).to(model.device).repeat(num_input, 1),
                text_atts=torch.from_numpy(cat_id_2_txt_atts[cat_id]).to(model.device).repeat(num_input, 1),
            ).float()
            all_score.append(score)

        all_score = torch.cat(all_score, dim=0)

        # print('logit_list', logit_list[0][0], logit_list[-1][0])
        # print('all_score', torch.max(all_score), torch.min(all_score))

        for top_i in range(num_img):
            logit_list[top_i][0] += 0.03*all_score[top_i].item()
        for top_i in range(num_img, len(logit_list)):
            logit_list[top_i][0] = logit_list[top_i][0] - 1000

        cur_ap = calculate_ap(logit_list)
        ap_sum += cur_ap
        recall_cnt += 1

        all_logit_list[cat_id] = logit_list

    pickle.dump(all_logit_list, open('sim_matrix/imagenet_a_all_logit_list_blip2_org.pkl', 'wb'))

    print('mAP: ', ap_sum / recall_cnt)
    print('rec: ', rec_sum / rec_cnt)
    return None




def compute_sim_matrix_imagenet_a_tgvpt(model, data_loader, **kwargs):
    import json
    import pickle

    # load annotation
    cat_id_2_cat_name = json.load(open('DATASET_PATH/imagenet-a-cat2name.json', 'r'))
    occluded_coco_retrieval_ann_file = json.load(open('DATASET_PATH/imagenet-a-annfile.json'))
    cat_id_2_occ_img = occluded_coco_retrieval_ann_file['cat_id_2_occ_img']
    cat_id_2_negative_img = occluded_coco_retrieval_ann_file['cat_id_2_negative_img']
    img_id_2_all_ins_id = occluded_coco_retrieval_ann_file['img_id_2_all_ins_id']

    logging.info("Computing features for evaluation...")
    # ----------------------------- text features for evaluation -----------------------------
    logging.info("Get text features.")
    cat_id_2_txt_feat = {}
    cat_id_2_txt_id = {}
    cat_id_2_txt_atts = {}
    iter_txt = 0
    for cat_id in cat_id_2_occ_img.keys():
        iter_txt += 1
        cur_txt = cat_id_2_cat_name[cat_id]
        text_input = model.tokenizer(
            cur_txt,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)
        text_feat = model.forward_text(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
        cat_id_2_txt_feat[cat_id] = text_embed.detach().cpu().numpy()
        cat_id_2_txt_id[cat_id] = text_input.input_ids.detach().cpu().numpy()
        cat_id_2_txt_atts[cat_id] = text_input.attention_mask.detach().cpu().numpy()

    # ----------------------------- image features for evaluation -----------------------------
    from torchvision import transforms as pth_transforms
    from PIL import Image
    img_preprocess = pth_transforms.Compose([
        pth_transforms.Resize([364, 364], pth_transforms.InterpolationMode.BICUBIC),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    logging.info("Get image features.")
    
    img_id_2_vit_feat = {}
    img_id_2_img_feat = {}
    img_folder = 'DATASET_PATH/imagenet-a'
    iter_img = 0
    for img_id in tqdm(img_id_2_all_ins_id):
        iter_img += 1
        cur_image = img_preprocess(Image.open(os.path.join(img_folder, img_id)).convert('RGB')).unsqueeze(0)
        cur_image = cur_image.to(device=model.device, non_blocking=True)
        image_feat, vit_feat = model.forward_image(cur_image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)
        img_id_2_vit_feat[img_id] = vit_feat.detach().cpu().numpy()
        img_id_2_img_feat[img_id] = image_embed.detach().cpu().numpy()
        with open('DATASET_PATH/imagenet_a_blip_feat.pkl', 'wb') as dump_f:    cur_pkl = pickle.load(open('DATASET_PATH/imagenet_a_blip_feat.pkl', 'rb'))

    # ----------------------------- similarity matrix for evaluation -----------------------------
    recall_cnt = 0
    ap_sum = 0
    topk = 200  # topK in reranking
    for cat_id in tqdm(cat_id_2_txt_feat.keys()):
        # if cat_id == '1':
        #     continue
        # print(coco_91[int(cat_id)])
        cur_txt_feat = torch.from_numpy(cat_id_2_txt_feat[cat_id]).to(model.device)
        logit_list = []
        occ_img_id_list = cat_id_2_occ_img[cat_id]
        neg_img_id_list = cat_id_2_negative_img[cat_id]
        for img_id in occ_img_id_list:
            cur_img_feat = torch.from_numpy(img_id_2_img_feat[str(img_id)]).to(model.device)
            cur_logit = torch.max(cur_img_feat @ cur_txt_feat.T).cpu().item()
            logit_list.append([cur_logit, 1, img_id])
        for img_id in neg_img_id_list:
            cur_img_feat = torch.from_numpy(img_id_2_img_feat[str(img_id)]).to(model.device)
            cur_logit = torch.max(cur_img_feat @ cur_txt_feat.T).cpu().item()
            logit_list.append([cur_logit, 0, img_id])
        logit_list.sort(key=lambda x: -x[0])

        # reranking
        num_img = min(topk, len(logit_list))
        # num_img = new_input_list.shape[0]
        all_score = []
        bsz = 20
        for i in tqdm(range(0, num_img, bsz)):
            new_input_list = []  # expected BS=100
            for top_i in range(i, min(num_img, i + bsz)):
                img_id = logit_list[top_i][-1]
                cur_image = img_preprocess(Image.open(os.path.join(img_folder, img_id)).convert('RGB')).unsqueeze(0)
                cur_image = cur_image.to(device=model.device, non_blocking=True)
                new_input_list.append(cur_image)

            new_input_list = torch.cat(new_input_list, dim=0)  # 100 x 3 x 224 x 224

            num_input = new_input_list.shape[0]
            score = model.compute_itm_tgvpt(
                image_inputs=new_input_list,
                text_ids=torch.from_numpy(cat_id_2_txt_id[cat_id]).to(model.device).repeat(num_input, 1),
                text_atts=torch.from_numpy(cat_id_2_txt_atts[cat_id]).to(model.device).repeat(num_input, 1),
                text_embeds=cur_txt_feat[0].repeat(num_input, 1),
            ).float()
            all_score.append(score)

        all_score = torch.cat(all_score, dim=0)
        org_ap = calculate_ap(logit_list)
        print('org_ap: ', org_ap)
        pos_ave_score = 0
        neg_ave_score = 0
        pos_cnt = 0
        neg_cnt = 0
        # ipdb.set_trace()
        for top_i in range(num_img):
            logit_list[top_i][0] += all_score[top_i].item() * 0.03
            if logit_list[top_i][1] == 1:
                pos_ave_score += all_score[top_i].item()
                pos_cnt += 1
            else:
                neg_ave_score += all_score[top_i].item()
                neg_cnt += 1
        print('pos_ave: ', pos_ave_score / pos_cnt)
        print('neg_ave: ', neg_ave_score / neg_cnt)
        print('pos_cnt: ', pos_cnt)
        print('neg_cnt: ', neg_cnt)
        for top_i in range(num_img, len(logit_list)):
            logit_list[top_i][0] = logit_list[top_i][0] - 1000

        cur_ap = calculate_ap(logit_list)
        print('ap_after_rerank: ', cur_ap)
        ap_sum += cur_ap
        recall_cnt += 1

    print('mAP: ', ap_sum / recall_cnt)

    return None


def compute_sim_matrix_occluded_coco_tgvpt(model, data_loader, **kwargs):
    import json
    import pickle

    # load annotation
    occluded_coco_retrieval_ann_file = json.load(
        open('DATASET_PATH/blip2_ypliu_coco2017val_cat_id_2_occ_img_negative_img.json'))
    cat_id_2_occ_img = occluded_coco_retrieval_ann_file['cat_id_2_occ_img']
    cat_id_2_negative_img = occluded_coco_retrieval_ann_file['cat_id_2_negative_img']
    img_id_2_all_ins_id = occluded_coco_retrieval_ann_file['img_id_2_all_ins_id']

    logging.info("Computing features for evaluation...")
    # ----------------------------- text features for evaluation -----------------------------
    coco_91 = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush']
    #
    # logging.info("Get text features.")
    # cat_id_2_txt_feat = {}
    # cat_id_2_txt_id = {}
    # cat_id_2_txt_atts = {}
    # iter_txt = 0
    # for cat_id in cat_id_2_occ_img.keys():
    #     iter_txt += 1
    #     cur_txt = coco_91[int(cat_id)]
    #     text_input = model.tokenizer(
    #         cur_txt,
    #         padding="max_length",
    #         truncation=True,
    #         max_length=35,
    #         return_tensors="pt",
    #     ).to(model.device)
    #     text_feat = model.forward_text(text_input)
    #     text_embed = F.normalize(model.text_proj(text_feat))
    #     cat_id_2_txt_feat[cat_id] = text_embed.detach().cpu().numpy()
    #     cat_id_2_txt_id[cat_id] = text_input.input_ids.detach().cpu().numpy()
    #     cat_id_2_txt_atts[cat_id] = text_input.attention_mask.detach().cpu().numpy()

    # ----------------------------- image features for evaluation -----------------------------
    from torchvision import transforms as pth_transforms
    from PIL import Image
    img_preprocess = pth_transforms.Compose([
        pth_transforms.Resize([364, 364], pth_transforms.InterpolationMode.BICUBIC),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    # logging.info("Get image features.")
    #
    # img_id_2_vit_feat = {}
    # img_id_2_img_feat = {}
    img_folder = 'DATASET_PATH/imagenet-a'
    # iter_img = 0
    # for img_id in tqdm(img_id_2_all_ins_id.keys()):
    #     iter_img += 1
    #     cur_image = img_preprocess(Image.open(os.path.join(img_folder, str(img_id).rjust(12,'0')+'.jpg')).convert('RGB')).unsqueeze(0)
    #     cur_image = cur_image.to(device=model.device, non_blocking=True)
    #     image_feat, vit_feat = model.forward_image(cur_image)
    #     image_embed = model.vision_proj(image_feat)
    #     image_embed = F.normalize(image_embed, dim=-1)
    #     img_id_2_vit_feat[img_id] = vit_feat.detach().cpu().numpy()
    #     img_id_2_img_feat[img_id] = image_embed.detach().cpu().numpy()
    #
    #     pickle.dump({'cat_id_2_txt_feat': cat_id_2_txt_feat
    #                     , 'cat_id_2_txt_id': cat_id_2_txt_id
    #                     , 'cat_id_2_txt_atts': cat_id_2_txt_atts
    #                     , 'img_id_2_vit_feat': img_id_2_vit_feat
    #                     , 'img_id_2_img_feat': img_id_2_img_feat}, dump_f)
    cur_pkl = pickle.load(open('DATASET_PATH/blip2_ypliu_occluded_coco_clip_feat.pkl', 'rb'))
    cat_id_2_txt_feat = cur_pkl['cat_id_2_txt_feat']
    cat_id_2_txt_id = cur_pkl['cat_id_2_txt_id']
    cat_id_2_txt_atts = cur_pkl['cat_id_2_txt_atts']
    img_id_2_vit_feat = cur_pkl['img_id_2_vit_feat']
    img_id_2_img_feat = cur_pkl['img_id_2_img_feat']

    # ----------------------------- similarity matrix for evaluation -----------------------------
    recall_cnt = 0
    ap_sum = 0
    topk = 100  # topK in reranking
    for cat_id in tqdm(cat_id_2_txt_feat.keys()):
        if cat_id == '1':
            continue
        print(coco_91[int(cat_id)])
        # if coco_91[int(cat_id)] in ['bicycle', 'toilet', 'motorcycle', 'potted plant', 'bench', 'sink']:
        #     continue
        # if coco_91[int(cat_id)] not in ['toilet']:
        #     continue
        cur_txt_feat = torch.from_numpy(cat_id_2_txt_feat[cat_id]).to(model.device)
        logit_list = []
        occ_img_id_list = cat_id_2_occ_img[cat_id]
        neg_img_id_list = cat_id_2_negative_img[cat_id]
        for img_id in occ_img_id_list:
            cur_img_feat = torch.from_numpy(img_id_2_img_feat[str(img_id)]).to(model.device)
            cur_logit = torch.max(cur_img_feat @ cur_txt_feat.T).cpu().item()
            logit_list.append([cur_logit, 1, img_id])
        for img_id in neg_img_id_list:
            cur_img_feat = torch.from_numpy(img_id_2_img_feat[str(img_id)]).to(model.device)
            cur_logit = torch.max(cur_img_feat @ cur_txt_feat.T).cpu().item()
            logit_list.append([cur_logit, 0, img_id])
        logit_list.sort(key=lambda x: -x[0])

        # reranking
        num_img = min(topk, len(logit_list))


        # stat num_positive, if lower than 20, skip
        pre_pos_cnt = 0
        for top_i in range(num_img):
            if logit_list[top_i][1] == 1:
                pre_pos_cnt += 1
        if pre_pos_cnt > 20:

            # num_img = new_input_list.shape[0]
            all_score = []
            bsz = 20
            for i in tqdm(range(0, num_img, bsz)):
                # print(i)
                new_input_list = []  # expected BS=100
                for top_i in range(i, min(num_img, i + bsz)):
                    img_id = logit_list[top_i][-1]
                    cur_image = img_preprocess(Image.open(os.path.join(img_folder, str(img_id).rjust(12, '0') + '.jpg')).convert('RGB')).unsqueeze(0)
                    cur_image = cur_image.to(device=model.device, non_blocking=True)
                    new_input_list.append(cur_image)

                new_input_list = torch.cat(new_input_list, dim=0)  # 100 x 3 x 224 x 224
                # ipdb.set_trace()
                # print(new_input_list.shape)

                num_input = new_input_list.shape[0]
                score = model.compute_itm_tgvpt(
                    image_inputs=new_input_list,
                    text_ids=torch.from_numpy(cat_id_2_txt_id[cat_id]).to(model.device).repeat(num_input, 1),
                    text_atts=torch.from_numpy(cat_id_2_txt_atts[cat_id]).to(model.device).repeat(num_input, 1),
                    text_embeds=cur_txt_feat[0].repeat(num_input, 1),
                ).float()
                all_score.append(score)
                # ipdb.set_trace()
                # print(score.shape)

            all_score = torch.cat(all_score, dim=0)
            org_ap = calculate_ap(logit_list)
            print('org_ap: ', org_ap)
            pos_ave_score = 0
            neg_ave_score = 0
            pos_cnt = 0
            neg_cnt = 0
            # ipdb.set_trace()
            for top_i in range(num_img):
                logit_list[top_i][0] += all_score[top_i].item() * 0.05
                if logit_list[top_i][1] == 1:
                    pos_ave_score += all_score[top_i].item()
                    pos_cnt += 1
                else:
                    neg_ave_score += all_score[top_i].item()
                    neg_cnt += 1
            print('pos_ave: ', pos_ave_score / pos_cnt)
            print('neg_ave: ', neg_ave_score / neg_cnt)
            print('pos_cnt: ', pos_cnt)
            print('neg_cnt: ', neg_cnt)
            for top_i in range(num_img, len(logit_list)):
                logit_list[top_i][0] = logit_list[top_i][0] - 1000

        cur_ap = calculate_ap(logit_list)
        print('ap_after_rerank: ', cur_ap)
        ap_sum += cur_ap
        recall_cnt += 1

        # if cur_ap < org_ap:
        #     ipdb.set_trace()

    print('mAP: ', ap_sum / recall_cnt)

    return None


def compute_sim_matrix_occluded_coco_tgvpt_multi_prompt3(model, data_loader, **kwargs):
    import json
    import pickle

    # load annotation
    occluded_coco_retrieval_ann_file = json.load(open('DATASET_PATH/karpathy_test_cat_id_2_occ_img_negative_img.json'))
    cat_id_2_occ_img = occluded_coco_retrieval_ann_file['cat_id_2_occ_img']
    cat_id_2_negative_img = occluded_coco_retrieval_ann_file['cat_id_2_negative_img']

    logging.info("Computing features for evaluation...")
    # ----------------------------- text features for evaluation -----------------------------
    coco_91 = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush']
    #
    logging.info("Get text features.")
    cat_id_2_txt_feat = {}
    cat_id_2_txt_feat_new = {}
    cat_id_2_txt_id = {}
    cat_id_2_txt_atts = {}
    iter_txt = 0
    for cat_id in cat_id_2_occ_img.keys():
        iter_txt += 1
        cur_txt = coco_91[int(cat_id)]
        text_input = model.tokenizer(
            cur_txt,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)
        text_feat = model.forward_text_org(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
        text_feat_new = model.forward_text(text_input)
        text_embed_new = F.normalize(model.text_proj(text_feat_new))

        cat_id_2_txt_feat[cat_id] = text_embed.detach().cpu().numpy()
        cat_id_2_txt_feat_new[cat_id] = text_embed_new.detach().cpu().numpy()
        cat_id_2_txt_id[cat_id] = text_input.input_ids.detach().cpu().numpy()
        cat_id_2_txt_atts[cat_id] = text_input.attention_mask.detach().cpu().numpy()
    # ----------------------------- image features for evaluation -----------------------------
    from torchvision import transforms as pth_transforms
    from PIL import Image
    img_preprocess = pth_transforms.Compose([
        pth_transforms.Resize([364, 364], pth_transforms.InterpolationMode.BICUBIC),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    cur_pkl = pickle.load(open('DATASET_PATH/imagenet_r_clip_feat.pkl', 'rb'))
    # cat_id_2_txt_feat = cur_pkl['cat_id_2_txt_feat']
    # cat_id_2_txt_id = cur_pkl['cat_id_2_txt_id']
    # cat_id_2_txt_atts = cur_pkl['cat_id_2_txt_atts']
    img_id_2_vit_feat = cur_pkl['img_id_2_vit_feat']
    img_id_2_img_feat = cur_pkl['img_id_2_img_feat']

    # ----------------------------- similarity matrix for evaluation -----------------------------
    img_folder = '/home/ypliu/datasets/coco/val2017/'
    img_folder2 = '/home/ypliu/datasets/coco/train2017/'
    recall_cnt = 0
    ap_sum = 0
    topk = 100  # topK in reranking
    for cat_id in tqdm(cat_id_2_txt_feat.keys()):
        if cat_id == '1':
            continue
        # print(coco_91[int(cat_id)])
        cur_txt_feat = torch.from_numpy(cat_id_2_txt_feat[cat_id]).to(model.device)
        logit_list = []
        occ_img_id_list = cat_id_2_occ_img[cat_id]
        neg_img_id_list = cat_id_2_negative_img[cat_id]
        for img_id in occ_img_id_list:
            cur_img_feat = torch.from_numpy(img_id_2_img_feat[img_id]).to(model.device)
            cur_logit = torch.max(cur_img_feat @ cur_txt_feat.T).cpu().item()
            logit_list.append([cur_logit, 1, img_id])
        for img_id in neg_img_id_list:
            cur_img_feat = torch.from_numpy(img_id_2_img_feat[img_id]).to(model.device)
            cur_logit = torch.max(cur_img_feat @ cur_txt_feat.T).cpu().item()
            logit_list.append([cur_logit, 0, img_id])
        logit_list.sort(key=lambda x: -x[0])

        # reranking
        num_img = min(topk, len(logit_list))
        # num_img = new_input_list.shape[0]
        all_score = []
        bsz = 20
        for i in tqdm(range(0, num_img, bsz)):
            new_input_list = []  # expected BS=100
            for top_i in range(i, min(num_img, i + bsz)):
                img_id = logit_list[top_i][-1]
                # cur_image = img_preprocess(Image.open(os.path.join(img_folder, str(img_id).rjust(12, '0') + '.jpg')).convert('RGB')).unsqueeze(0)
                # cur_image = img_preprocess(Image.open(os.path.join(img_folder, str(img_id).rjust(12, '0') + '.jpg')).convert('RGB')).unsqueeze(0)
                # cur_image = img_preprocess(Image.open(img_id_2_path[img_id]).convert('RGB')).unsqueeze(0)
                if os.path.exists(os.path.join(img_folder, str(logit_list[top_i][-1]).rjust(12, '0') + '.jpg')):
                    cur_image = img_preprocess(Image.open(os.path.join(img_folder, str(logit_list[top_i][-1]).rjust(12, '0') + '.jpg')).convert('RGB')).unsqueeze(0)
                else:
                    cur_image = img_preprocess(Image.open(os.path.join(img_folder2, str(logit_list[top_i][-1]).rjust(12, '0') + '.jpg')).convert('RGB')).unsqueeze(0)

                cur_image = cur_image.to(device=model.device, non_blocking=True)
                new_input_list.append(cur_image)

            new_input_list = torch.cat(new_input_list, dim=0)  # 100 x 3 x 224 x 224

            num_input = new_input_list.shape[0]
            score = model.compute_itm_tgvpt(
                image_inputs=new_input_list,
                text_ids=torch.from_numpy(cat_id_2_txt_id[cat_id]).to(model.device).repeat(num_input, 1),
                text_atts=torch.from_numpy(cat_id_2_txt_atts[cat_id]).to(model.device).repeat(num_input, 1),
                text_embeds=torch.from_numpy(cat_id_2_txt_feat_new[cat_id]).to(model.device)[0].repeat(num_input, 1),
            ).float()
            all_score.append(score)

            # image_inputs = images[topk_idx.cpu()].to(model.device)
            # score = model.compute_itm_tgvpt(
            #     image_inputs=image_inputs,
            #     text_ids=text_ids[start + i].repeat(k_test, 1),
            #     text_atts=text_atts[start + i].repeat(k_test, 1),
            #     text_embeds=text_embeds_new[start + i].repeat(k_test, 1),
            # ).float()
            # score_matrix_t2i[start + i, topk_idx] = score + topk_sim

        all_score = torch.cat(all_score, dim=0)
        for top_i in range(num_img):
            logit_list[top_i][0] += 0.05 * all_score[top_i].item()
        for top_i in range(num_img, len(logit_list)):
            logit_list[top_i][0] = logit_list[top_i][0] - 1000

        cur_ap = calculate_ap(logit_list)
        ap_sum += cur_ap
        recall_cnt += 1

    print('mAP: ', ap_sum / recall_cnt)

    return None


def compute_sim_matrix_imagenet_r_tgvpt_multi_prompt3(model, data_loader, **kwargs):
    import json
    import pickle

    cat_id_2_cat_name = json.load(open('DATASET_PATH/imagenet-r-cat2name.json', 'r'))
    occluded_coco_retrieval_ann_file = json.load(
        open('DATASET_PATH/imagenet-r-annfile.json'))

    logging.info("Computing features for evaluation...")
    # ----------------------------- text features for evaluation -----------------------------
    logging.info("Get text features.")
    cat_id_2_txt_feat = {}
    cat_id_2_txt_feat_new = {}
    cat_id_2_txt_id = {}
    cat_id_2_txt_atts = {}
    iter_txt = 0
    for cat_id in cat_id_2_occ_img.keys():
        iter_txt += 1
        cur_txt = cat_id_2_cat_name[cat_id]
        # print(cur_txt)
        text_input = model.tokenizer(
            cur_txt,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)
        text_feat = model.forward_text_org(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
        text_feat_new = model.forward_text(text_input)
        text_embed_new = F.normalize(model.text_proj(text_feat_new))

        cat_id_2_txt_feat[cat_id] = text_embed.detach().cpu().numpy()
        cat_id_2_txt_feat_new[cat_id] = text_embed_new.detach().cpu().numpy()
        cat_id_2_txt_id[cat_id] = text_input.input_ids.detach().cpu().numpy()
        cat_id_2_txt_atts[cat_id] = text_input.attention_mask.detach().cpu().numpy()

    # ----------------------------- image features for evaluation -----------------------------
    from torchvision import transforms as pth_transforms
    from PIL import Image
    img_preprocess = pth_transforms.Compose([
        pth_transforms.Resize([364, 364], pth_transforms.InterpolationMode.BICUBIC),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    # logging.info("Get image features.")
    #
    # img_id_2_vit_feat = {}
    # img_id_2_img_feat = {}
    img_folder = '/disk1/work/ypliu/imagenet-r'
    # iter_img = 0
    # for img_id in tqdm(img_id_2_all_ins_id.keys()):
    #     iter_img += 1
    #     cur_image = img_preprocess(Image.open(os.path.join(img_folder, str(img_id).rjust(12,'0')+'.jpg')).convert('RGB')).unsqueeze(0)
    #     cur_image = cur_image.to(device=model.device, non_blocking=True)
    #     image_feat, vit_feat = model.forward_image(cur_image)
    #     image_embed = model.vision_proj(image_feat)
    #     image_embed = F.normalize(image_embed, dim=-1)
    #     img_id_2_vit_feat[img_id] = vit_feat.detach().cpu().numpy()
    #     img_id_2_img_feat[img_id] = image_embed.detach().cpu().numpy()
    #
    #     pickle.dump({'cat_id_2_txt_feat': cat_id_2_txt_feat
    #                     , 'cat_id_2_txt_id': cat_id_2_txt_id
    #                     , 'cat_id_2_txt_atts': cat_id_2_txt_atts
    #                     , 'img_id_2_vit_feat': img_id_2_vit_feat
    #                     , 'img_id_2_img_feat': img_id_2_img_feat}, dump_f)
    cur_pkl = pickle.load(open('DATASET_PATH/imagenet_r_clip_feat.pkl', 'rb'))