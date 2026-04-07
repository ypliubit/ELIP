
import pickle
import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES, COCO_CLASSNAMES, SIMPLE_COCO_TEMPLATES, \
    COCO_CLASSNAMES_KNOWN, COCO_CLASSNAMES_NOVEL
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast

from training.distributed import world_info_from_env
from torch import distributed as dist
import ipdb

from pytorch_metric_learning import losses as losses_metric_learning
import matplotlib.pyplot as plt
import shutil


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def backward_avoid_nan(total_loss, scaler):
    # Check if the loss is NaN
    is_nan = torch.isnan(total_loss).any()
    
    if is_nan:
        logging.warning("NaN loss detected, skipping backward but updating scaler")
        # Skip backward entirely
        # Just let the optimizer step with whatever gradients it has
        # The scaler will be updated in the optimizer step
        return False
    else:
        # Normal path - loss is not NaN
        if scaler is not None:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        return True


def train_one_epoch_spt_vpt_iter(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    train_vpt = False
    iter_freq = args.iter_freq
    for i, batch in enumerate(dataloader):
        if (i+1) % iter_freq == 0:
            train_vpt = not train_vpt

        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler[0](step)
            scheduler[1](step)

        images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        if train_vpt:
            optimizer[0].zero_grad()
        else:
            optimizer[1].zero_grad()

        # with autocast():
        model_out = model(images, texts)
        logit_scale = model_out["logit_scale"]
        # if args.distill:
        #     with torch.no_grad():
        #         dist_model_out = dist_model(images, texts)
        #     model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
        losses = loss(**model_out, output_dict=True)

        total_loss = sum(losses.values())
        losses["loss"] = total_loss

        backward(total_loss, scaler)
        if train_vpt:
            optimizer[0].step()
        else:
            optimizer[1].step()
        # if args.accum_freq == 1:
        #     with autocast():
        #         model_out = model(images, texts)
        #         logit_scale = model_out["logit_scale"]
        #         if args.distill:
        #             with torch.no_grad():
        #                 dist_model_out = dist_model(images, texts)
        #             model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
        #         losses = loss(**model_out, output_dict=True)
        #
        #         total_loss = sum(losses.values())
        #         losses["loss"] = total_loss
        #
        #     backward(total_loss, scaler)
        # else:
        #     # First, cache the features without any gradient tracking.
        #     with torch.no_grad():
        #         with autocast():
        #             model_out = model(images, texts)
        #
        #             for f in ("logit_scale", "logit_bias"):
        #                 model_out.pop(f, None)
        #
        #             for key, val in model_out.items():
        #                 if key in accum_features:
        #                     accum_features[key].append(val)
        #                 else:
        #                     accum_features[key] = [val]
        #
        #         accum_images.append(images)
        #         accum_texts.append(texts)
        #
        #     # If (i + 1) % accum_freq is not zero, move on to the next batch.
        #     if ((i + 1) % args.accum_freq) > 0:
        #         # FIXME this makes data time logging unreliable when accumulating
        #         continue
        #
        #     # Now, ready to take gradients for the last accum_freq batches.
        #     # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
        #     # Call backwards each time, but only step optimizer at the end.
        #     optimizer.zero_grad()
        #     for j in range(args.accum_freq):
        #         images = accum_images[j]
        #         texts = accum_texts[j]
        #         with autocast():
        #             model_out = model(images, texts)
        #
        #             inputs_no_accum = {}
        #             inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
        #             if "logit_bias" in model_out:
        #                 inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")
        #
        #             inputs = {}
        #             for key, val in accum_features.items():
        #                 accumulated = accum_features[key]
        #                 inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])
        #
        #             losses = loss(**inputs, **inputs_no_accum, output_dict=True)
        #             del inputs
        #             del inputs_no_accum
        #             total_loss = sum(losses.values())
        #             losses["loss"] = total_loss
        #
        #         backward(total_loss, scaler)
        #
        # if scaler is not None:
        #     if args.horovod:
        #         optimizer.synchronize()
        #         scaler.unscale_(optimizer)
        #         if args.grad_clip_norm is not None:
        #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
        #         with optimizer.skip_synchronize():
        #             scaler.step(optimizer)
        #     else:
        #         if args.grad_clip_norm is not None:
        #             scaler.unscale_(optimizer)
        #             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
        #         scaler.step(optimizer)
        #     scaler.update()
        # else:
        #     if args.grad_clip_norm is not None:
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
        #     optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer[0].param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer[0].param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        # if i > 100:
        #     return

        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        if args.csv_label_key != "none":
            images, texts, _ = batch
        else:
            images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        # print('type(images.data) ', type(images.data))

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, texts)
                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                losses = loss(**model_out, output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def train_one_epoch_tgvpt(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        # if i > 100:
        #     return
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        if args.csv_label_key != "none":
            images, texts, _ = batch
        else:
            images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        # print('type(images.data) ', type(images.data))

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                # model_out = model(images, texts)
                # text_features = model_out["text_features"]
                # logit_scale = model_out["logit_scale"]
                # if args.distill:
                #     with torch.no_grad():
                #         dist_model_out = dist_model(images, texts)
                #     model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                # losses = loss(**model_out, output_dict=True)
                text_features = model.encode_text(texts, normalize=True)
                logit_scale = model.logit_scale.exp()
                losses = {}

                batch_size = images.shape[0]
                all_image_features = []
                for j in range(batch_size):
                    image_features_i = []
                    for k in range(batch_size):
                        with autocast():
                            # model_out = model(images[j].unsqueeze(0), texts[k].unsqueeze(0))
                            # image_features = model_out["image_features"]
                            image_features = model.encode_image(images[j].unsqueeze(0), text_embed=text_features[k].unsqueeze(0), normalize=True)
                            image_features_i.append(image_features)
                    all_image_features.append(torch.concat(image_features_i, dim=0).unsqueeze(0))
                all_image_features = torch.cat(all_image_features, dim=0).to(device=device, non_blocking=True)

                # using all bsz x bsz logits
                logits_per_image = logit_scale * all_image_features @ text_features.T
                # print(logits_per_image.shape)
                logits_per_image_flat = torch.reshape(logits_per_image, (batch_size, -1))
                labels = torch.arange(batch_size, device=device, dtype=torch.long)
                gt_labels = labels * batch_size + labels
                losses['token_loss'] = F.cross_entropy(logits_per_image_flat, gt_labels)

                # image_features_i = torch.cat(image_features_i).to(device=device, non_blocking=True)
                # all_logits = image_features_i @ all_text_features.T
                # logits_cate = torch.diagonal(all_logits, dim1=-2, dim2=-1)
                # prompt_idx = torch.argmax(logits_cate)
                # all_image_features.append(image_features_i[prompt_idx].unsqueeze(0))

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for



def train_mp_one_epoch_tgvpt3_logo_prompt(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        # if i > 100:
        #     return
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        # ------------------------ logo prompt ------------------------
        logo_prompt = args.logo_prompt
        if logo_prompt:
            images, texts, logo_prompt_im = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
            logo_prompt_im = logo_prompt_im.to(device=device, non_blocking=True)
        # ------------------------ logo prompt ------------------------
        else:
            if args.csv_label_key != "none":
                images, texts, _ = batch
            else:
                images, texts = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
            # print('type(images.data) ', type(images.data))

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                # model_out = model(images, texts)
                # text_features = model_out["text_features"]
                # logit_scale = model_out["logit_scale"]
                # if args.distill:
                #     with torch.no_grad():
                #         dist_model_out = dist_model(images, texts)
                #     model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                # losses = loss(**model_out, output_dict=True)
                text_features = model.module.encode_text(texts, normalize=True)
                logit_scale = model.module.logit_scale.exp()
                losses = {}

                # ----------- multi-GPU running 2024.9.8 -----------
                # all_text_features = gather_features(text_features, rank, world_size)
                all_text_features = torch.cat(GatherLayer.apply(text_features), dim=0)
                all_texts = torch.cat(GatherLayer.apply(texts), dim=0)
                all_logo_prompt_im = torch.cat(GatherLayer.apply(logo_prompt_im), dim=0)


                batch_size = images.shape[0]
                all_batch_size = all_texts.shape[0]
                all_image_features = []
                prompt_ = torch.zeros(1, 3, 224, 224).to(device)
                for j in range(batch_size):
                    # print(j)
                    # image_features_i = []
                    img_forward_list = []
                    for k in range(all_batch_size):
                        # print(k)
                        # with autocast():
                        # model_out = model(images[j].unsqueeze(0), texts[k].unsqueeze(0))
                        # image_features = model_out["image_features"]
                        img = images[j].unsqueeze(0)
                        # import time
                        # start_time = time.time()
                        if logo_prompt:
                            x_ = np.random.choice(224 - 16)
                            y_ = np.random.choice(224 - 192)
                            # print('time 1: ', time.time() - start_time)
                            # x_ = 0
                            # y_ = 0
                            prompt = torch.zeros_like(prompt_)
                            # print('time 2: ', time.time() - start_time)
                            prompt[:, :, x_:x_ + 16, y_:y_ + 192] = all_logo_prompt_im[k,:,:,:].unsqueeze(0)

                            min_r, max_r = torch.min(img[:, 0, :, :]), torch.max(img[:, 0, :, :])
                            min_g, max_g = torch.min(img[:, 1, :, :]), torch.max(img[:, 1, :, :])
                            min_b, max_b = torch.min(img[:, 2, :, :]), torch.max(img[:, 2, :, :])
                            img = img + prompt
                            img[:, 0, :, :] = torch.clip(img[:, 0, :, :], min_r, max_r)
                            img[:, 1, :, :] = torch.clip(img[:, 1, :, :], min_g, max_g)
                            img[:, 2, :, :] = torch.clip(img[:, 2, :, :], min_b, max_b)

                        img_forward_list.append(img)
                    img_forward_list = torch.cat(img_forward_list, dim=0)
                    image_features = model(img_forward_list, None, all_text_features)["image_features"]
                    all_image_features.append(image_features.unsqueeze(0))

                    #     image_features = model(img, None, all_text_features[k].unsqueeze(0))["image_features"]
                    #     image_features_i.append(image_features)
                    
                    # all_image_features.append(torch.concat(image_features_i, dim=0).unsqueeze(0))
                    # ipdb.set_trace()
                all_image_features = torch.cat(all_image_features, dim=0).to(device=device, non_blocking=True)

                all_image_features = torch.cat(GatherLayer.apply(all_image_features), dim=0)
                # ipdb.set_trace()
                B = all_text_features.shape[0]
                logits_per_text = torch.zeros((B, B), device=device)
                for txt_i in range(B):
                    cur_logits_per_text = logit_scale * all_text_features[txt_i].unsqueeze(0) @ all_image_features[:, txt_i, :].T
                    logits_per_text[txt_i] = cur_logits_per_text[0]
                gt_labels = torch.arange(all_batch_size, device=device, dtype=torch.long)
                losses['token_loss'] = F.cross_entropy(logits_per_text, gt_labels)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def train_mp_one_epoch_tgvpt2(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    # ipdb.set_trace()
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    # ipdb.set_trace()
    dataloader = data['train'].dataloader
    # ipdb.set_trace()
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    local_rank, rank, world_size = world_info_from_env()
    # ipdb.set_trace()
    for i, batch in enumerate(dataloader):
        # print(i)
        # ipdb.set_trace()
        # if i > 500:
        #     return
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        if args.csv_label_key != "none":
            images, texts, _ = batch
        else:
            images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                # model_out = model(images, texts)
                # text_features = model_out["text_features"]
                text_features = model.module.encode_text(texts, normalize=True)#.detach()
                logit_scale = model.module.logit_scale.exp()
                losses = {}

                # ----------- multi-GPU running 2024.9.8 -----------
                # all_text_features = gather_features(text_features, rank, world_size)
                all_text_features = torch.cat(GatherLayer.apply(text_features), dim=0)
                all_texts = torch.cat(GatherLayer.apply(texts), dim=0)


                batch_size = images.shape[0]
                all_batch_size = all_texts.shape[0]
                all_image_features = []
                # ipdb.set_trace()
                for j in range(batch_size):
                    # ipdb.set_trace()
                    # loop
                    # image_features_i = []
                    # for k in range(all_batch_size):
                    #     with autocast():
                    #         image_features = model(images[j].unsqueeze(0), None, all_text_features[k].unsqueeze(0))["image_features"]
                    #         image_features_i.append(image_features)
                    # all_image_features.append(torch.concat(image_features_i, dim=0).unsqueeze(0))
                    # batch
                    image_features = model(images[j].unsqueeze(0).repeat(all_batch_size,1,1,1), None, all_text_features)["image_features"]
                    all_image_features.append(image_features.unsqueeze(0))
                all_image_features = torch.cat(all_image_features, dim=0).to(device=device, non_blocking=True)

                # all_image_features = model(images[0].unsqueeze(0).repeat(all_batch_size*batch_size, 1, 1, 1), None
                #                      , all_text_features.repeat(batch_size, 1))["image_features"]
                # all_image_features = gather_features(all_image_features, rank, world_size)
                # all_image_features = torch.cat(GatherLayer.apply(all_image_features), dim=0)
                
                #print('all_image_features.requires_grad', all_image_features.requires_grad)
                #print('all_text_features.requires_grad', all_text_features.requires_grad)
                # using all bsz x bsz logits
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_image = torch.cat(GatherLayer.apply(logits_per_image), dim=0)
                # ----------- multi-GPU running 2024.9.8 -----------
                logits_per_image_flat = torch.reshape(logits_per_image, (all_batch_size, -1))
                labels = torch.arange(all_batch_size, device=device, dtype=torch.long)
                gt_labels = labels * all_batch_size + labels
                losses['token_loss'] = F.cross_entropy(logits_per_image_flat, gt_labels)
                # ipdb.set_trace()

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for




def smooth_ap_loss(logits_matrix, gamma=10.0):
    """
    Compute the SmoothAP loss given the logits matrix.

    Args:
        logits_matrix (torch.Tensor): A (B, B) tensor containing similarity scores.
        gamma (float): Scaling factor for the sigmoid function.

    Returns:
        torch.Tensor: The SmoothAP loss.
    """
    batch_size = logits_matrix.size(0)
    device = logits_matrix.device

    # Identity matrix to select positives
    eye = torch.eye(batch_size, device=device).bool()

    # Masks to select positive and negative similarities
    pos_mask = eye
    neg_mask = ~eye

    # Positive similarities for image-to-text retrieval
    s_p_i2t = logits_matrix[pos_mask].unsqueeze(1)  # Shape: (B, 1)

    # Negative similarities for image-to-text retrieval
    s_n_i2t = logits_matrix[neg_mask].view(batch_size, -1)  # Shape: (B, B-1)

    # Compute pairwise differences
    delta_i2t = s_n_i2t - s_p_i2t  # Shape: (B, B-1)

    # Apply sigmoid function
    sigma_i2t = torch.sigmoid(gamma * delta_i2t)

    # Compute approximate rank
    rank_i2t = 1 + sigma_i2t.sum(dim=1)  # Shape: (B,)

    # Compute AP
    ap_i2t = 1.0 / rank_i2t

    # Compute SmoothAP loss for image-to-text retrieval
    loss_i2t = 1.0 - ap_i2t  # Shape: (B,)

    # Combine the losses
    loss = loss_i2t.sum() / batch_size

    return loss



def train_mp_one_epoch_tgvpt3_smooth_ap(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    # ipdb.set_trace()
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    # ipdb.set_trace()
    dataloader = data['train'].dataloader
    # ipdb.set_trace()
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    local_rank, rank, world_size = world_info_from_env()
    # ipdb.set_trace()
    for i, batch in enumerate(dataloader):
        # print(i)
        # ipdb.set_trace()
        # if i > 500:
        #     return
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        if args.csv_label_key != "none":
            images, texts, _ = batch
        else:
            images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                # model_out = model(images, texts)
                # text_features = model_out["text_features"]
                text_features = model.module.encode_text(texts, normalize=True)#.detach()
                logit_scale = model.module.logit_scale.exp()
                losses = {}

                # ----------- multi-GPU running 2024.9.8 -----------
                # all_text_features = gather_features(text_features, rank, world_size)
                all_text_features = torch.cat(GatherLayer.apply(text_features), dim=0)
                all_texts = torch.cat(GatherLayer.apply(texts), dim=0)


                batch_size = images.shape[0]
                all_batch_size = all_texts.shape[0]
                all_image_features = []
                # ipdb.set_trace()
                for j in range(batch_size):
                    # ipdb.set_trace()
                    # loop
                    # image_features_i = []
                    # for k in range(all_batch_size):
                    #     with autocast():
                    #         image_features = model(images[j].unsqueeze(0), None, all_text_features[k].unsqueeze(0))["image_features"]
                    #         image_features_i.append(image_features)
                    # all_image_features.append(torch.concat(image_features_i, dim=0).unsqueeze(0))
                    # batch
                    # ipdb.set_trace()
                    image_features = model(images[j].unsqueeze(0).repeat(all_batch_size,1,1,1), None, all_text_features)["image_features"]
                    all_image_features.append(image_features.unsqueeze(0))
                all_image_features = torch.cat(all_image_features, dim=0).to(device=device, non_blocking=True)

                # all_image_features = model(images[0].unsqueeze(0).repeat(all_batch_size*batch_size, 1, 1, 1), None
                #                      , all_text_features.repeat(batch_size, 1))["image_features"]
                # all_image_features = gather_features(all_image_features, rank, world_size)
                # all_image_features = torch.cat(GatherLayer.apply(all_image_features), dim=0)
                
                #print('all_image_features.requires_grad', all_image_features.requires_grad)
                #print('all_text_features.requires_grad', all_text_features.requires_grad)
                # using all bsz x bsz logits
                all_image_features = torch.cat(GatherLayer.apply(all_image_features), dim=0)
                # all_image_features = all_image_features.reshape(-1, all_image_features.shape[-1])
                B = all_text_features.shape[0]
                logits_per_text = torch.zeros((B, B), device=device)
                for txt_i in range(B):
                    cur_logits_per_text = logit_scale * all_text_features[txt_i].unsqueeze(0) @ all_image_features[:, txt_i, :].T
                    logits_per_text[txt_i] = cur_logits_per_text[0]
                # ipdb.set_trace()
                # logits_per_text = logit_scale * all_text_features @ all_image_features.T
                # B = logits_per_text.shape[0]
                # logits_per_text = logits_per_text.reshape(B, B, B).reshape(B*B, B)
                # # logits_per_text = logits_per_text[torch.arange(B*B), torch.arange(B), torch.arange(B)]
                # ipdb.set_trace()
                # logits_per_image = torch.cat(GatherLayer.apply(logits_per_image), dim=0)
                # # ----------- multi-GPU running 2024.9.8 -----------
                # logits_per_image_flat = torch.reshape(logits_per_image, (all_batch_size, -1))
                # gt_labels = torch.arange(all_batch_size, device=device, dtype=torch.long)
                # gt_labels = labels * all_batch_size + labels
                losses['token_loss'] = smooth_ap_loss(logits_per_text) * 100
                # ipdb.set_trace()

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for



def train_mp_one_epoch_tgvpt3_simple_contrast(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    # ipdb.set_trace()
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    # print('args.distill: ', args.distill)
    # print('\n')
    # print('\n')
    # print('\n')
    # print('\n')
    # print('\n')
    # print('\n')
    # print('\n')
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    # ipdb.set_trace()
    dataloader = data['train'].dataloader
    # ipdb.set_trace()
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    local_rank, rank, world_size = world_info_from_env()
    # ipdb.set_trace()
    for i, batch in enumerate(dataloader):
        # print(i)
        # ipdb.set_trace()
        # if i > 500:
        #     return
        if i % 10000 == 0:
            checkpoint_dict = {
                "epoch": epoch,
                "name": 'test',
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": i
            }
            torch.save(checkpoint_dict,os.path.join('PROJECT_PATH/v5_4', f"epoch_{epoch}_iter_{i}.pt"),)

        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        if args.csv_label_key != "none":
            images, texts, _ = batch
        else:
            images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                # model_out = model(images, texts)
                # text_features = model_out["text_features"]
                text_features = model.module.encode_text(texts, normalize=True)#.detach()
                logit_scale = model.module.logit_scale.exp()
                losses = {}

                # ----------- multi-GPU running 2024.9.8 -----------
                # all_text_features = gather_features(text_features, rank, world_size)
                all_text_features = torch.cat(GatherLayer.apply(text_features), dim=0)
                all_texts = torch.cat(GatherLayer.apply(texts), dim=0)


                batch_size = images.shape[0]
                all_batch_size = all_texts.shape[0]
                all_image_features = []
                # ipdb.set_trace()
                for j in range(batch_size):
                    # ipdb.set_trace()
                    # loop
                    # image_features_i = []
                    # for k in range(all_batch_size):
                    #     with autocast():
                    #         image_features = model(images[j].unsqueeze(0), None, all_text_features[k].unsqueeze(0))["image_features"]
                    #         image_features_i.append(image_features)
                    # all_image_features.append(torch.concat(image_features_i, dim=0).unsqueeze(0))
                    # batch
                    # ipdb.set_trace()
                    image_features = model(images[j].unsqueeze(0).repeat(all_batch_size,1,1,1), None, all_text_features)["image_features"]
                    all_image_features.append(image_features.unsqueeze(0))
                all_image_features = torch.cat(all_image_features, dim=0).to(device=device, non_blocking=True)

                # all_image_features = model(images[0].unsqueeze(0).repeat(all_batch_size*batch_size, 1, 1, 1), None
                #                      , all_text_features.repeat(batch_size, 1))["image_features"]
                # all_image_features = gather_features(all_image_features, rank, world_size)
                # all_image_features = torch.cat(GatherLayer.apply(all_image_features), dim=0)
                
                #print('all_image_features.requires_grad', all_image_features.requires_grad)
                #print('all_text_features.requires_grad', all_text_features.requires_grad)
                # using all bsz x bsz logits
                all_image_features = torch.cat(GatherLayer.apply(all_image_features), dim=0)
                # all_image_features = all_image_features.reshape(-1, all_image_features.shape[-1])
                B = all_text_features.shape[0]
                logits_per_text = torch.zeros((B, B), device=device)
                for txt_i in range(B):
                    cur_logits_per_text = logit_scale * all_text_features[txt_i].unsqueeze(0) @ all_image_features[:, txt_i, :].T
                    logits_per_text[txt_i] = cur_logits_per_text[0]
                # ipdb.set_trace()
                # logits_per_text = logit_scale * all_text_features @ all_image_features.T
                # B = logits_per_text.shape[0]
                # logits_per_text = logits_per_text.reshape(B, B, B).reshape(B*B, B)
                # # logits_per_text = logits_per_text[torch.arange(B*B), torch.arange(B), torch.arange(B)]
                # ipdb.set_trace()
                # logits_per_image = torch.cat(GatherLayer.apply(logits_per_image), dim=0)
                # # ----------- multi-GPU running 2024.9.8 -----------
                # logits_per_image_flat = torch.reshape(logits_per_image, (all_batch_size, -1))
                gt_labels = torch.arange(all_batch_size, device=device, dtype=torch.long)
                # gt_labels = labels * all_batch_size + labels
                losses['token_loss'] = F.cross_entropy(logits_per_text, gt_labels)
                # ipdb.set_trace()

                # # inject NaN to test implementation manually - 5.12
                # losses['token_loss'] = torch.ones_like(losses['token_loss']) * float('nan')

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            # ipdb.set_trace()
            # backward(torch.zeros_like(total_loss), scaler)
            # if not torch.isnan(total_loss).item():
            backward(total_loss, scaler)
            # if not torch.isnan(total_loss).any():
            #     backward(total_loss, scaler)
            # else:
            #     logging.warning(f"NaN loss detected at step {step}, skipping backward pass")
            # else:
            #     ipdb.set_trace()
                # backward(0, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for



def train_mp_one_epoch_tgvpt3_simple_contrast_avoid_nan(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    # ipdb.set_trace()
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    # print('args.distill: ', args.distill)
    # print('\n')
    # print('\n')
    # print('\n')
    # print('\n')
    # print('\n')
    # print('\n')
    # print('\n')
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    # ipdb.set_trace()
    dataloader = data['train'].dataloader
    # ipdb.set_trace()
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    local_rank, rank, world_size = world_info_from_env()
    # ipdb.set_trace()
    for i, batch in enumerate(dataloader):

        if i % 10000 == 0:
            checkpoint_dict = {
                "epoch": epoch,
                "name": 'test',
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": i
            }
            torch.save(checkpoint_dict,os.path.join('PROJECT_PATH/v4', f"epoch_{epoch}_iter_{i}.pt"),)

        # print(i)
        # ipdb.set_trace()
        # if i > 500:
        #     return
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        if args.csv_label_key != "none":
            images, texts, _ = batch
        else:
            images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                # model_out = model(images, texts)
                # text_features = model_out["text_features"]
                text_features = model.module.encode_text(texts, normalize=True)#.detach()
                logit_scale = model.module.logit_scale.exp()
                losses = {}

                # ----------- multi-GPU running 2024.9.8 -----------
                # all_text_features = gather_features(text_features, rank, world_size)
                all_text_features = torch.cat(GatherLayer.apply(text_features), dim=0)
                all_texts = torch.cat(GatherLayer.apply(texts), dim=0)


                batch_size = images.shape[0]
                all_batch_size = all_texts.shape[0]
                all_image_features = []
                # ipdb.set_trace()
                for j in range(batch_size):
                    # ipdb.set_trace()
                    # loop
                    # image_features_i = []
                    # for k in range(all_batch_size):
                    #     with autocast():
                    #         image_features = model(images[j].unsqueeze(0), None, all_text_features[k].unsqueeze(0))["image_features"]
                    #         image_features_i.append(image_features)
                    # all_image_features.append(torch.concat(image_features_i, dim=0).unsqueeze(0))
                    # batch
                    # ipdb.set_trace()
                    image_features = model(images[j].unsqueeze(0).repeat(all_batch_size,1,1,1), None, all_text_features)["image_features"]
                    all_image_features.append(image_features.unsqueeze(0))
                all_image_features = torch.cat(all_image_features, dim=0).to(device=device, non_blocking=True)

                # all_image_features = model(images[0].unsqueeze(0).repeat(all_batch_size*batch_size, 1, 1, 1), None
                #                      , all_text_features.repeat(batch_size, 1))["image_features"]
                # all_image_features = gather_features(all_image_features, rank, world_size)
                # all_image_features = torch.cat(GatherLayer.apply(all_image_features), dim=0)
                
                #print('all_image_features.requires_grad', all_image_features.requires_grad)
                #print('all_text_features.requires_grad', all_text_features.requires_grad)
                # using all bsz x bsz logits
                all_image_features = torch.cat(GatherLayer.apply(all_image_features), dim=0)
                # all_image_features = all_image_features.reshape(-1, all_image_features.shape[-1])
                B = all_text_features.shape[0]
                logits_per_text = torch.zeros((B, B), device=device)
                for txt_i in range(B):
                    cur_logits_per_text = logit_scale * all_text_features[txt_i].unsqueeze(0) @ all_image_features[:, txt_i, :].T
                    logits_per_text[txt_i] = cur_logits_per_text[0]
                # ipdb.set_trace()
                # logits_per_text = logit_scale * all_text_features @ all_image_features.T
                # B = logits_per_text.shape[0]
                # logits_per_text = logits_per_text.reshape(B, B, B).reshape(B*B, B)
                # # logits_per_text = logits_per_text[torch.arange(B*B), torch.arange(B), torch.arange(B)]
                # ipdb.set_trace()
                # logits_per_image = torch.cat(GatherLayer.apply(logits_per_image), dim=0)
                # # ----------- multi-GPU running 2024.9.8 -----------
                # logits_per_image_flat = torch.reshape(logits_per_image, (all_batch_size, -1))
                gt_labels = torch.arange(all_batch_size, device=device, dtype=torch.long)
                # gt_labels = labels * all_batch_size + labels
                losses['token_loss'] = F.cross_entropy(logits_per_text, gt_labels)
                # ipdb.set_trace()

                # # inject NaN to test implementation manually - 5.12
                # losses['token_loss'] = torch.ones_like(losses['token_loss']) * float('nan')

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            # ipdb.set_trace()
            # backward(torch.zeros_like(total_loss), scaler)
            # if not torch.isnan(total_loss).item():
            print(total_loss)
            backward_successful = backward_avoid_nan(total_loss, scaler)
            # if not torch.isnan(total_loss).any():
            #     backward(total_loss, scaler)
            # else:
            #     logging.warning(f"NaN loss detected at step {step}, skipping backward pass")
            # else:
            #     ipdb.set_trace()
                # backward(0, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                # backward(total_loss, scaler)
                if not torch.isnan(loss).item():
                    backward(total_loss, scaler)
                # else:
                #     backward(0, scaler)

        
        # Skip optimizer step if backward wasn't successful
        if backward_successful:
            if scaler is not None:
                if args.horovod:
                    optimizer.synchronize()
                    scaler.unscale_(optimizer)
                    if args.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                    with optimizer.skip_synchronize():
                        scaler.step(optimizer)
                else:
                    if args.grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                    scaler.step(optimizer)
                scaler.update()
            else:
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                optimizer.step()
        # else:
        #     # If backward wasn't successful (due to NaN), skip the optimizer step
        #     # but still update the scaler to maintain its state
        #     if scaler is not None:
        #         scaler.update()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for




def train_mp_one_epoch_tgvpt3_simple_contrast_2txtfeat(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    # ipdb.set_trace()
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    # print('args.distill: ', args.distill)
    # print('\n')
    # print('\n')
    # print('\n')
    # print('\n')
    # print('\n')
    # print('\n')
    # print('\n')
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    # ipdb.set_trace()
    dataloader = data['train'].dataloader
    # ipdb.set_trace()
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    local_rank, rank, world_size = world_info_from_env()
    # ipdb.set_trace()
    for i, batch in enumerate(dataloader):
        # print(i)
        # ipdb.set_trace()
        # if i > 500:
        #     return
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        if args.csv_label_key != "none":
            images, texts, _ = batch
        else:
            images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                # model_out = model(images, texts)
                # text_features = model_out["text_features"]
                text_features, text_features_tgvpt = model.module.encode_text(texts, normalize=True)#.detach()
                logit_scale = model.module.logit_scale.exp()
                losses = {}

                # ----------- multi-GPU running 2024.9.8 -----------
                # all_text_features = gather_features(text_features, rank, world_size)
                all_text_features = torch.cat(GatherLayer.apply(text_features), dim=0)
                all_text_features_tgvpt = torch.cat(GatherLayer.apply(text_features_tgvpt), dim=0)
                all_texts = torch.cat(GatherLayer.apply(texts), dim=0)


                batch_size = images.shape[0]
                all_batch_size = all_texts.shape[0]
                all_image_features = []
                # ipdb.set_trace()
                for j in range(batch_size):
                    # ipdb.set_trace()
                    # loop
                    # image_features_i = []
                    # for k in range(all_batch_size):
                    #     with autocast():
                    #         image_features = model(images[j].unsqueeze(0), None, all_text_features[k].unsqueeze(0))["image_features"]
                    #         image_features_i.append(image_features)
                    # all_image_features.append(torch.concat(image_features_i, dim=0).unsqueeze(0))
                    # batch
                    # ipdb.set_trace()
                    # image_features = model(images[j].unsqueeze(0).repeat(all_batch_size,1,1,1), None, all_text_features)["image_features"]
                    image_features = model(images[j].unsqueeze(0).repeat(all_batch_size,1,1,1), None, all_text_features_tgvpt)["image_features"]
                    all_image_features.append(image_features.unsqueeze(0))
                all_image_features = torch.cat(all_image_features, dim=0).to(device=device, non_blocking=True)

                # all_image_features = model(images[0].unsqueeze(0).repeat(all_batch_size*batch_size, 1, 1, 1), None
                #                      , all_text_features.repeat(batch_size, 1))["image_features"]
                # all_image_features = gather_features(all_image_features, rank, world_size)
                # all_image_features = torch.cat(GatherLayer.apply(all_image_features), dim=0)
                
                #print('all_image_features.requires_grad', all_image_features.requires_grad)
                #print('all_text_features.requires_grad', all_text_features.requires_grad)
                # using all bsz x bsz logits
                all_image_features = torch.cat(GatherLayer.apply(all_image_features), dim=0)
                # all_image_features = all_image_features.reshape(-1, all_image_features.shape[-1])
                B = all_text_features.shape[0]
                logits_per_text = torch.zeros((B, B), device=device)
                for txt_i in range(B):
                    cur_logits_per_text = logit_scale * all_text_features[txt_i].unsqueeze(0) @ all_image_features[:, txt_i, :].T
                    logits_per_text[txt_i] = cur_logits_per_text[0]
                # ipdb.set_trace()
                # logits_per_text = logit_scale * all_text_features @ all_image_features.T
                # B = logits_per_text.shape[0]
                # logits_per_text = logits_per_text.reshape(B, B, B).reshape(B*B, B)
                # # logits_per_text = logits_per_text[torch.arange(B*B), torch.arange(B), torch.arange(B)]
                # ipdb.set_trace()
                # logits_per_image = torch.cat(GatherLayer.apply(logits_per_image), dim=0)
                # # ----------- multi-GPU running 2024.9.8 -----------
                # logits_per_image_flat = torch.reshape(logits_per_image, (all_batch_size, -1))
                gt_labels = torch.arange(all_batch_size, device=device, dtype=torch.long)
                # gt_labels = labels * all_batch_size + labels
                losses['token_loss'] = F.cross_entropy(logits_per_text, gt_labels)
                # ipdb.set_trace()

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for




def train_mp_one_epoch_tgvpt3_simple_contrast_both_real_sync_caption(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    # ipdb.set_trace()
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    # print('args.distill: ', args.distill)
    # print('\n')
    # print('\n')
    # print('\n')
    # print('\n')
    # print('\n')
    # print('\n')
    # print('\n')
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    # ipdb.set_trace()
    dataloader = data['train'].dataloader
    # ipdb.set_trace()
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    local_rank, rank, world_size = world_info_from_env()
    # ipdb.set_trace()
    for i, batch in enumerate(dataloader):
        # print(i)
        # ipdb.set_trace()
        # if i > 500:
        #     return
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        if args.csv_label_key != "none":
            images, texts, _ = batch
        else:
            images, texts, texts_real = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        texts_real = texts_real.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                # model_out = model(images, texts)
                # text_features = model_out["text_features"]
                text_features = model.module.encode_text(texts, normalize=True)#.detach()
                text_features_real = model.module.encode_text(texts_real, normalize=True)#.detach()
                logit_scale = model.module.logit_scale.exp()
                losses = {}

                # ----------- multi-GPU running 2024.9.8 -----------
                # all_text_features = gather_features(text_features, rank, world_size)
                all_text_features = torch.cat(GatherLayer.apply(text_features), dim=0)
                all_texts = torch.cat(GatherLayer.apply(texts), dim=0)
                all_text_features_real = torch.cat(GatherLayer.apply(text_features_real), dim=0)
                all_texts_real = torch.cat(GatherLayer.apply(texts_real), dim=0)


                batch_size = images.shape[0]
                all_batch_size = all_texts.shape[0]
                all_image_features = []
                all_image_features_real = []
                # ipdb.set_trace()
                for j in range(batch_size):
                    # ipdb.set_trace()
                    # loop
                    # image_features_i = []
                    # for k in range(all_batch_size):
                    #     with autocast():
                    #         image_features = model(images[j].unsqueeze(0), None, all_text_features[k].unsqueeze(0))["image_features"]
                    #         image_features_i.append(image_features)
                    # all_image_features.append(torch.concat(image_features_i, dim=0).unsqueeze(0))
                    # batch
                    # ipdb.set_trace()
                    image_features = model(images[j].unsqueeze(0).repeat(all_batch_size,1,1,1), None, all_text_features)["image_features"]
                    all_image_features.append(image_features.unsqueeze(0))
                    image_features_real = model(images[j].unsqueeze(0).repeat(all_batch_size,1,1,1), None, all_text_features_real)["image_features"]
                    all_image_features_real.append(image_features_real.unsqueeze(0))
                all_image_features = torch.cat(all_image_features, dim=0).to(device=device, non_blocking=True)
                all_image_features_real = torch.cat(all_image_features_real, dim=0).to(device=device, non_blocking=True)

                # all_image_features = model(images[0].unsqueeze(0).repeat(all_batch_size*batch_size, 1, 1, 1), None
                #                      , all_text_features.repeat(batch_size, 1))["image_features"]
                # all_image_features = gather_features(all_image_features, rank, world_size)
                # all_image_features = torch.cat(GatherLayer.apply(all_image_features), dim=0)
                
                #print('all_image_features.requires_grad', all_image_features.requires_grad)
                #print('all_text_features.requires_grad', all_text_features.requires_grad)
                # using all bsz x bsz logits
                all_image_features = torch.cat(GatherLayer.apply(all_image_features), dim=0)
                all_image_features_real = torch.cat(GatherLayer.apply(all_image_features_real), dim=0)
                # all_image_features = all_image_features.reshape(-1, all_image_features.shape[-1])
                B = all_text_features.shape[0]
                logits_per_text = torch.zeros((B, B), device=device)
                for txt_i in range(B):
                    cur_logits_per_text = logit_scale * all_text_features[txt_i].unsqueeze(0) @ all_image_features[:, txt_i, :].T
                    logits_per_text[txt_i] = cur_logits_per_text[0]

                logits_per_text_real = torch.zeros((B, B), device=device)
                for txt_i in range(B):
                    cur_logits_per_text_real = logit_scale * all_text_features_real[txt_i].unsqueeze(0) @ all_image_features_real[:, txt_i, :].T
                    logits_per_text_real[txt_i] = cur_logits_per_text_real[0]
                # ipdb.set_trace()
                # logits_per_text = logit_scale * all_text_features @ all_image_features.T
                # B = logits_per_text.shape[0]
                # logits_per_text = logits_per_text.reshape(B, B, B).reshape(B*B, B)
                # # logits_per_text = logits_per_text[torch.arange(B*B), torch.arange(B), torch.arange(B)]
                # ipdb.set_trace()
                # logits_per_image = torch.cat(GatherLayer.apply(logits_per_image), dim=0)
                # # ----------- multi-GPU running 2024.9.8 -----------
                # logits_per_image_flat = torch.reshape(logits_per_image, (all_batch_size, -1))
                gt_labels = torch.arange(all_batch_size, device=device, dtype=torch.long)
                # gt_labels = labels * all_batch_size + labels
                losses['token_loss'] = 0.5 * (F.cross_entropy(logits_per_text, gt_labels) + F.cross_entropy(logits_per_text_real, gt_labels))
                # ipdb.set_trace()

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for




def train_mp_one_epoch_tgvpt3_simple_contrast_transformer(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    # ipdb.set_trace()
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()

    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    # ipdb.set_trace()
    dataloader = data['train'].dataloader
    # ipdb.set_trace()
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    local_rank, rank, world_size = world_info_from_env()
    # ipdb.set_trace()
    for i, batch in enumerate(dataloader):
        # print(i)
        # ipdb.set_trace()
        # if i > 500:
        #     return
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        if args.csv_label_key != "none":
            images, texts, _ = batch
        else:
            images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                # model_out = model(images, texts)
                # text_features = model_out["text_features"]
                text_features = model.module.encode_text(texts, normalize=True)#.detach()
                logit_scale = model.module.logit_scale.exp()
                losses = {}

                # ----------- multi-GPU running 2024.9.8 -----------
                # all_text_features = gather_features(text_features, rank, world_size)
                all_text_features = torch.cat(GatherLayer.apply(text_features), dim=0)
                all_texts = torch.cat(GatherLayer.apply(texts), dim=0)

                text_features_org = model.module.encode_text_org(texts, normalize=True)#.detach()
                all_text_features_org = torch.cat(GatherLayer.apply(text_features_org), dim=0)


                batch_size = images.shape[0]
                all_batch_size = all_texts.shape[0]
                all_image_features = []
                # ipdb.set_trace()
                for j in range(batch_size):

                    image_features = model(images[j].unsqueeze(0).repeat(all_batch_size,1,1,1), None, all_text_features)["image_features"]
                    all_image_features.append(image_features.unsqueeze(0))
                all_image_features = torch.cat(all_image_features, dim=0).to(device=device, non_blocking=True)

               
                all_image_features = torch.cat(GatherLayer.apply(all_image_features), dim=0)
                # all_image_features = all_image_features.reshape(-1, all_image_features.shape[-1])
                B = all_text_features.shape[0]
                logits_per_text = torch.zeros((B, B), device=device)
                for txt_i in range(B):
                    cur_logits_per_text = logit_scale * all_text_features_org[txt_i].unsqueeze(0) @ all_image_features[:, txt_i, :].T
                    logits_per_text[txt_i] = cur_logits_per_text[0]
                # # ----------- multi-GPU running 2024.9.8 -----------
                # logits_per_image_flat = torch.reshape(logits_per_image, (all_batch_size, -1))
                gt_labels = torch.arange(all_batch_size, device=device, dtype=torch.long)
                # gt_labels = labels * all_batch_size + labels
                losses['token_loss'] = F.cross_entropy(logits_per_text, gt_labels)
                # ipdb.set_trace()

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for



def train_mp_one_epoch_tgvpt3_simple_sigliploss(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()

    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    # ipdb.set_trace()
    dataloader = data['train'].dataloader
    # ipdb.set_trace()
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    local_rank, rank, world_size = world_info_from_env()
    # ipdb.set_trace()
    for i, batch in enumerate(dataloader):

        if i % 10000 == 0:
            checkpoint_dict = {
                "epoch": epoch,
                "name": 'test',
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": i
            }
            torch.save(checkpoint_dict,os.path.join('PROJECT_PATH/v5_2_10tokens', f"epoch_{epoch}_iter_{i}.pt"),)

        # print(i)
        # ipdb.set_trace()
        # if i > 500:
        #     return
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        if args.csv_label_key != "none":
            images, texts, _ = batch
        else:
            images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        # ipdb.set_trace()

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():

                text_features = model.module.encode_text(texts, normalize=True)
                logit_scale = model.module.logit_scale.exp()
                # logit_bias = model.module.logit_bias # -10
                # logit_scale = np.log(10)
                logit_bias = -10
                losses = {}

                
                all_text_features = torch.cat(GatherLayer.apply(text_features), dim=0)
                all_texts = torch.cat(GatherLayer.apply(texts), dim=0)


                batch_size = images.shape[0]
                all_batch_size = all_texts.shape[0]
                all_image_features = []
                for j in range(batch_size):
                    image_features = model(images[j].unsqueeze(0).repeat(all_batch_size,1,1,1), None, all_text_features)["image_features"]
                    all_image_features.append(image_features.unsqueeze(0))
                all_image_features = torch.cat(all_image_features, dim=0).to(device=device, non_blocking=True)
                all_image_features = torch.cat(GatherLayer.apply(all_image_features), dim=0)
                B = all_text_features.shape[0]
                logits_per_text = torch.zeros((B, B), device=device)
                for txt_i in range(B):
                    cur_logits_per_text = logit_scale * all_text_features[txt_i].unsqueeze(0) @ all_image_features[:, txt_i, :].T
                    logits_per_text[txt_i] = cur_logits_per_text[0]
                
                labels = -torch.ones((B, B), device=device, dtype=all_image_features.dtype)
                labels = 2 * torch.eye(B, device=device, dtype=all_image_features.dtype) + labels
                # ipdb.set_trace()
                losses['token_loss'] = -F.logsigmoid(labels * (logits_per_text + logit_bias)).sum() / B
                
                # gt_labels = torch.arange(all_batch_size, device=device, dtype=torch.long)
                
                # losses['token_loss'] = F.cross_entropy(logits_per_text, gt_labels)
                

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for

def tgvpt_evaluate_reranking_logo_prompt(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            all_images, all_texts = [], []
            if args.logo_prompt:
                all_logo_prompt_im = []
            for i, batch in enumerate(dataloader):
                with autocast():
                    if args.logo_prompt:
                        images, texts, prompt_im = batch
                    else:
                        images, texts = batch
                    images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                    texts = texts.to(device=device, non_blocking=True)
                    all_images.append(images)
                    all_texts.append(texts)
                    if args.logo_prompt:
                        prompt_im = prompt_im.to(device=device, non_blocking=True)
                        all_logo_prompt_im.append(prompt_im)

                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                                         F.cross_entropy(logits_per_image, labels) +
                                         F.cross_entropy(logits_per_text, labels)
                                 ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")
            all_text_features = torch.cat(all_text_features).to(device=device, non_blocking=True)

            all_images = torch.cat(all_images, dim=0)
            all_texts = torch.cat(all_texts, dim=0)
            if args.logo_prompt:
                all_logo_prompt_im = torch.cat(all_logo_prompt_im, dim=0)

            import pickle
            topk = 100
            org_clip_features = pickle.load(
                open('DATASET_PATH/clip_vitb16_coco_infer_img_txt_features_9.4.pkl', 'rb'))
            org_image_features = org_clip_features['image_features']
            org_text_features = org_clip_features['text_features']
            logits_per_image = (org_image_features @ org_text_features.t()).detach().cpu()
            logits_per_text = logits_per_image.t().detach().cpu()
            # logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
            ranking = torch.argsort(logits_per_text, descending=True)
            top1_recall = 0
            top5_recall = 0
            top10_recall = 0
            recall_cnt = 0
            for text_i in range(ranking.shape[0]):
                print('text_i: ', text_i)
                topk_img_indices = ranking[text_i][:topk]
                # image_i_2_tg_img_feat = {}
                logit_imgi_list = []
                for image_i in topk_img_indices.detach().cpu().tolist():
                    img = all_images[image_i].unsqueeze(0)
                    if args.logo_prompt:
                        x_ = 0
                        y_ = 0
                        prompt = torch.zeros(1, 3, 224, 224).cuda()
                        prompt[:, :, x_:x_ + 16, y_:y_ + 192] = all_logo_prompt_im[text_i, :, :, :].unsqueeze(0)
                        min_r, max_r = torch.min(img[:, 0, :, :]), torch.max(img[:, 0, :, :])
                        min_g, max_g = torch.min(img[:, 1, :, :]), torch.max(img[:, 1, :, :])
                        min_b, max_b = torch.min(img[:, 2, :, :]), torch.max(img[:, 2, :, :])
                        img = img + prompt
                        img[:, 0, :, :] = torch.clip(img[:, 0, :, :], min_r, max_r)
                        img[:, 1, :, :] = torch.clip(img[:, 1, :, :], min_g, max_g)
                        img[:, 2, :, :] = torch.clip(img[:, 2, :, :], min_b, max_b)
                    tg_img_feat = model.encode_image(img, text_embed=all_text_features[text_i].unsqueeze(0), normalize=True)
                    tg_img_feat = tg_img_feat.squeeze()
                    # image_i_2_tg_img_feat[image_i] = tg_img_feat
                    logit_imgi_list.append(
                        [(all_text_features[text_i] * tg_img_feat).sum().detach().cpu().tolist(), image_i])
                logit_imgi_list.sort(key=lambda x: -x[0])
                # ipdb.set_trace()
                logit_list = np.array(logit_imgi_list)[:, 1].tolist()
                if text_i in logit_list[:1]:
                    print('top1')
                    top1_recall += 1
                if text_i in logit_list[:5]:
                    print('top5')
                    top5_recall += 1
                if text_i in logit_list[:10]:
                    print('top10')
                    top10_recall += 1
                recall_cnt += 1
            print('top1_recall: ', top1_recall / recall_cnt)
            print('top5_recall: ', top5_recall / recall_cnt)
            print('top10_recall: ', top10_recall / recall_cnt)
    return 0



def train_mp_one_epoch_tgvpt3_sd(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    # ipdb.set_trace()
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    # ipdb.set_trace()
    dataloader = data['train'].dataloader
    # ipdb.set_trace()
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    local_rank, rank, world_size = world_info_from_env()
    # ipdb.set_trace()
    for i, batch in enumerate(dataloader):
        # print(i)
        # ipdb.set_trace()
        # if i > 500:
        #     return
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        if args.csv_label_key != "none":
            images, texts, _ = batch
        else:
            images, texts, images_sd = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        images_sd = images_sd.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                # model_out = model(images, texts)
                # text_features = model_out["text_features"]
                text_features = model.module.encode_text(texts, normalize=True).detach()
                # ipdb.set_trace()
                text_features_sd = model.module.encode_image(images_sd, normalize=True).detach()
                logit_scale = model.module.logit_scale.exp()
                losses = {}

                # ----------- multi-GPU running 2024.9.8 -----------
                # all_text_features = gather_features(text_features, rank, world_size)
                all_text_features = torch.cat(GatherLayer.apply(text_features), dim=0)
                all_text_features_sd = torch.cat(GatherLayer.apply(text_features_sd), dim=0)
                all_text_features = all_text_features + all_text_features_sd
                # all_text_features = torch.cat([all_text_features, all_text_features_sd], dim=1)
                all_texts = torch.cat(GatherLayer.apply(texts), dim=0)


                batch_size = images.shape[0]
                all_batch_size = all_texts.shape[0]
                all_image_features = []
                # ipdb.set_trace()
                for j in range(batch_size):
                    # ipdb.set_trace()
                    # loop
                    # image_features_i = []
                    # for k in range(all_batch_size):
                    #     with autocast():
                    #         image_features = model(images[j].unsqueeze(0), None, all_text_features[k].unsqueeze(0))["image_features"]
                    #         image_features_i.append(image_features)
                    # all_image_features.append(torch.concat(image_features_i, dim=0).unsqueeze(0))
                    # batch
                    image_features = model(images[j].unsqueeze(0).repeat(all_batch_size,1,1,1), None, all_text_features)["image_features"]
                    all_image_features.append(image_features.unsqueeze(0))
                all_image_features = torch.cat(all_image_features, dim=0).to(device=device, non_blocking=True)

                # all_image_features = model(images[0].unsqueeze(0).repeat(all_batch_size*batch_size, 1, 1, 1), None
                #                      , all_text_features.repeat(batch_size, 1))["image_features"]
                # all_image_features = gather_features(all_image_features, rank, world_size)
                # all_image_features = torch.cat(GatherLayer.apply(all_image_features), dim=0)

                # using all bsz x bsz logits
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_image = torch.cat(GatherLayer.apply(logits_per_image), dim=0)
                # ----------- multi-GPU running 2024.9.8 -----------
                logits_per_image_flat = torch.reshape(logits_per_image, (all_batch_size, -1))
                labels = torch.arange(all_batch_size, device=device, dtype=torch.long)
                gt_labels = labels * all_batch_size + labels
                losses['token_loss'] = F.cross_entropy(logits_per_image_flat, gt_labels)
                # ipdb.set_trace()

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for



def train_mp_one_epoch_tgvpt3_sd1(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    # ipdb.set_trace()
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    # ipdb.set_trace()
    dataloader = data['train'].dataloader
    # ipdb.set_trace()
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    local_rank, rank, world_size = world_info_from_env()
    # ipdb.set_trace()
    for i, batch in enumerate(dataloader):
        # print(i)
        # ipdb.set_trace()
        # if i > 500:
        #     return
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        if args.csv_label_key != "none":
            images, texts, _ = batch
        else:
            images, texts, images_sd = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        images_sd = images_sd.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                # model_out = model(images, texts)
                # text_features = model_out["text_features"]
                # text_features = model.module.encode_text(texts, normalize=True).detach()
                # ipdb.set_trace()
                text_features_sd = model.module.encode_image(images_sd, normalize=True).detach()
                logit_scale = model.module.logit_scale.exp()
                losses = {}

                # ----------- multi-GPU running 2024.9.8 -----------
                # all_text_features = gather_features(text_features, rank, world_size)
                # all_text_features = torch.cat(GatherLayer.apply(text_features), dim=0)
                all_text_features = torch.cat(GatherLayer.apply(text_features_sd), dim=0)
                # all_text_features = all_text_features_sd
                # all_text_features = torch.cat([all_text_features, all_text_features_sd], dim=1)
                all_texts = torch.cat(GatherLayer.apply(texts), dim=0)


                batch_size = images.shape[0]
                all_batch_size = all_texts.shape[0]
                all_image_features = []
                # ipdb.set_trace()
                for j in range(batch_size):
                    # ipdb.set_trace()
                    # loop
                    # image_features_i = []
                    # for k in range(all_batch_size):
                    #     with autocast():
                    #         image_features = model(images[j].unsqueeze(0), None, all_text_features[k].unsqueeze(0))["image_features"]
                    #         image_features_i.append(image_features)
                    # all_image_features.append(torch.concat(image_features_i, dim=0).unsqueeze(0))
                    # batch
                    image_features = model(images[j].unsqueeze(0).repeat(all_batch_size,1,1,1), None, all_text_features)["image_features"]
                    all_image_features.append(image_features.unsqueeze(0))
                all_image_features = torch.cat(all_image_features, dim=0).to(device=device, non_blocking=True)

                # all_image_features = model(images[0].unsqueeze(0).repeat(all_batch_size*batch_size, 1, 1, 1), None
                #                      , all_text_features.repeat(batch_size, 1))["image_features"]
                # all_image_features = gather_features(all_image_features, rank, world_size)
                # all_image_features = torch.cat(GatherLayer.apply(all_image_features), dim=0)

                # using all bsz x bsz logits
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_image = torch.cat(GatherLayer.apply(logits_per_image), dim=0)
                # ----------- multi-GPU running 2024.9.8 -----------
                logits_per_image_flat = torch.reshape(logits_per_image, (all_batch_size, -1))
                labels = torch.arange(all_batch_size, device=device, dtype=torch.long)
                gt_labels = labels * all_batch_size + labels
                losses['token_loss'] = F.cross_entropy(logits_per_image_flat, gt_labels)
                # ipdb.set_trace()

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def train_mp_one_epoch_tgvpt3_sd2(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    # ipdb.set_trace()
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    # ipdb.set_trace()
    dataloader = data['train'].dataloader
    # ipdb.set_trace()
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    local_rank, rank, world_size = world_info_from_env()
    # ipdb.set_trace()
    for i, batch in enumerate(dataloader):
        # print(i)
        # ipdb.set_trace()
        # if i > 500:
        #     return
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        if args.csv_label_key != "none":
            images, texts, _ = batch
        else:
            images, texts, images_sd = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        images_sd = images_sd.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                # model_out = model(images, texts)
                # text_features = model_out["text_features"]
                text_features = model.module.encode_text(texts, normalize=True).detach()
                # ipdb.set_trace()
                text_features_sd = model.module.encode_image(images_sd, normalize=True).detach()
                logit_scale = model.module.logit_scale.exp()
                losses = {}

                # ----------- multi-GPU running 2024.9.8 -----------
                # all_text_features = gather_features(text_features, rank, world_size)
                all_text_features = torch.cat(GatherLayer.apply(text_features), dim=0)
                all_text_features_sd = torch.cat(GatherLayer.apply(text_features_sd), dim=0)
                # all_text_features = all_text_features + all_text_features_sd
                # all_text_features = torch.cat([all_text_features, all_text_features_sd], dim=1)
                all_texts = torch.cat(GatherLayer.apply(texts), dim=0)


                batch_size = images.shape[0]
                all_batch_size = all_texts.shape[0]
                all_image_features = []
                # ipdb.set_trace()
                for j in range(batch_size):
                    # ipdb.set_trace()
                    # loop
                    # image_features_i = []
                    # for k in range(all_batch_size):
                    #     with autocast():
                    #         image_features = model(images[j].unsqueeze(0), None, all_text_features[k].unsqueeze(0))["image_features"]
                    #         image_features_i.append(image_features)
                    # all_image_features.append(torch.concat(image_features_i, dim=0).unsqueeze(0))
                    # batch
                    image_features = model(images[j].unsqueeze(0).repeat(all_batch_size,1,1,1), None, all_text_features)["image_features"]
                    all_image_features.append(image_features.unsqueeze(0))
                all_image_features = torch.cat(all_image_features, dim=0).to(device=device, non_blocking=True)

                all_text_features = all_text_features + all_text_features_sd
                # all_image_features = model(images[0].unsqueeze(0).repeat(all_batch_size*batch_size, 1, 1, 1), None
                #                      , all_text_features.repeat(batch_size, 1))["image_features"]
                # all_image_features = gather_features(all_image_features, rank, world_size)
                # all_image_features = torch.cat(GatherLayer.apply(all_image_features), dim=0)

                # using all bsz x bsz logits
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_image = torch.cat(GatherLayer.apply(logits_per_image), dim=0)
                # ----------- multi-GPU running 2024.9.8 -----------
                logits_per_image_flat = torch.reshape(logits_per_image, (all_batch_size, -1))
                labels = torch.arange(all_batch_size, device=device, dtype=torch.long)
                gt_labels = labels * all_batch_size + labels
                losses['token_loss'] = F.cross_entropy(logits_per_image_flat, gt_labels)
                # ipdb.set_trace()

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for



def train_mp_one_epoch_tgvpt3_weighted_loss(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    local_rank, rank, world_size = world_info_from_env()
    for i, batch in enumerate(dataloader):
        # if i > 500:
        #     return
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        if args.csv_label_key != "none":
            images, texts, _ = batch
        else:
            images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                # model_out = model(images, texts)
                # text_features = model_out["text_features"]
                text_features = model.module.encode_text(texts, normalize=True).detach()
                logit_scale = model.module.logit_scale.exp()
                losses = {}

                # ----------- multi-GPU running 2024.9.8 -----------
                # all_text_features = gather_features(text_features, rank, world_size)
                all_text_features = torch.cat(GatherLayer.apply(text_features), dim=0)
                all_texts = torch.cat(GatherLayer.apply(texts), dim=0)


                batch_size = images.shape[0]
                all_batch_size = all_texts.shape[0]
                all_image_features = []
                for j in range(batch_size):
                    # loop
                    # image_features_i = []
                    # for k in range(all_batch_size):
                    #     with autocast():
                    #         image_features = model(images[j].unsqueeze(0), None, all_text_features[k].unsqueeze(0))["image_features"]
                    #         image_features_i.append(image_features)
                    # all_image_features.append(torch.concat(image_features_i, dim=0).unsqueeze(0))
                    # batch
                    image_features = model(images[j].unsqueeze(0).repeat(all_batch_size,1,1,1), None, all_text_features)["image_features"]
                    all_image_features.append(image_features.unsqueeze(0))
                all_image_features = torch.cat(all_image_features, dim=0).to(device=device, non_blocking=True)

                # all_image_features = model(images[0].unsqueeze(0).repeat(all_batch_size*batch_size, 1, 1, 1), None
                #                      , all_text_features.repeat(batch_size, 1))["image_features"]
                # all_image_features = gather_features(all_image_features, rank, world_size)
                # all_image_features = torch.cat(GatherLayer.apply(all_image_features), dim=0)

                # ----------- multi-GPU running 2024.9.8 -----------
                # using logits per image
                # logits_per_image = logit_scale * all_image_features @ all_text_features.T
                # logits_per_image = torch.cat(GatherLayer.apply(logits_per_image), dim=0)
                # logits_per_image_flat = torch.reshape(logits_per_image, (all_batch_size, -1))
                # labels = torch.arange(all_batch_size, device=device, dtype=torch.long)
                # gt_labels = labels * all_batch_size + labels
                # losses['token_loss'] = F.cross_entropy(logits_per_image_flat, gt_labels)

                # using logits per text
                weighted_param = 0.01
                gather_all_image_features = torch.cat(GatherLayer.apply(all_image_features), dim=0)
                gather_all_image_features_flat = torch.reshape(gather_all_image_features, (all_batch_size*all_batch_size, -1))
                logits_per_text_flat = logit_scale * all_text_features @ gather_all_image_features_flat.T
                labels = torch.arange(all_batch_size, device=device, dtype=torch.long)
                gt_labels = labels * all_batch_size + labels

                all_text_feat_sim_matrix = all_text_features @ all_text_features.T # B x B
                all_text_feat_sim_matrix = all_text_feat_sim_matrix.unsqueeze(-1).repeat(1, 1, all_text_feat_sim_matrix.shape[0]).reshape(all_text_feat_sim_matrix.shape[0], -1) # B x (B x B)
                all_text_feat_sim_matrix_mask = torch.ones_like(all_text_feat_sim_matrix)
                # ipdb.set_trace()
                all_text_feat_sim_matrix_mask[torch.arange(all_text_feat_sim_matrix.shape[0]), gt_labels] = 0
                logits_per_text_flat = logits_per_text_flat - weighted_param * logit_scale * (all_text_feat_sim_matrix * all_text_feat_sim_matrix_mask)

                # ipdb.set_trace()
                losses['token_loss'] = F.cross_entropy(logits_per_text_flat, gt_labels)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for



def train_mp_one_epoch_tgvpt3_contrast_textwise_loss(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    local_rank, rank, world_size = world_info_from_env()
    for i, batch in enumerate(dataloader):
        # if i > 500:
        #     return
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        if args.csv_label_key != "none":
            images, texts, _ = batch
        else:
            images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                # model_out = model(images, texts)
                # text_features = model_out["text_features"]
                text_features = model.module.encode_text(texts, normalize=True).detach()
                logit_scale = model.module.logit_scale.exp()
                losses = {}

                # ----------- multi-GPU running 2024.9.8 -----------
                # all_text_features = gather_features(text_features, rank, world_size)
                all_text_features = torch.cat(GatherLayer.apply(text_features), dim=0)
                all_texts = torch.cat(GatherLayer.apply(texts), dim=0)


                batch_size = images.shape[0]
                all_batch_size = all_texts.shape[0]
                all_image_features = []
                for j in range(batch_size):
                    # loop
                    # image_features_i = []
                    # for k in range(all_batch_size):
                    #     with autocast():
                    #         image_features = model(images[j].unsqueeze(0), None, all_text_features[k].unsqueeze(0))["image_features"]
                    #         image_features_i.append(image_features)
                    # all_image_features.append(torch.concat(image_features_i, dim=0).unsqueeze(0))
                    # batch
                    image_features = model(images[j].unsqueeze(0).repeat(all_batch_size,1,1,1), None, all_text_features)["image_features"]
                    all_image_features.append(image_features.unsqueeze(0))
                all_image_features = torch.cat(all_image_features, dim=0).to(device=device, non_blocking=True)

                # all_image_features = model(images[0].unsqueeze(0).repeat(all_batch_size*batch_size, 1, 1, 1), None
                #                      , all_text_features.repeat(batch_size, 1))["image_features"]
                # all_image_features = gather_features(all_image_features, rank, world_size)
                # all_image_features = torch.cat(GatherLayer.apply(all_image_features), dim=0)

                # ----------- multi-GPU running 2024.9.8 -----------
                # using logits per image
                # logits_per_image = logit_scale * all_image_features @ all_text_features.T
                # logits_per_image = torch.cat(GatherLayer.apply(logits_per_image), dim=0)
                # logits_per_image_flat = torch.reshape(logits_per_image, (all_batch_size, -1))
                # labels = torch.arange(all_batch_size, device=device, dtype=torch.long)
                # gt_labels = labels * all_batch_size + labels
                # losses['token_loss'] = F.cross_entropy(logits_per_image_flat, gt_labels)

                # using logits per text
                gather_all_image_features = torch.cat(GatherLayer.apply(all_image_features), dim=0)
                gather_all_image_features_flat = torch.reshape(gather_all_image_features, (all_batch_size*all_batch_size, -1))
                logits_per_text_flat = logit_scale * all_text_features @ gather_all_image_features_flat.T
                labels = torch.arange(all_batch_size, device=device, dtype=torch.long)
                gt_labels = labels * all_batch_size + labels
                losses['token_loss'] = F.cross_entropy(logits_per_text_flat, gt_labels)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def train_mp_one_epoch_tgvpt3_circle_loss(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    local_rank, rank, world_size = world_info_from_env()
    for i, batch in enumerate(dataloader):
        # if i > 500:
        #     return
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        if args.csv_label_key != "none":
            images, texts, _ = batch
        else:
            images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                # model_out = model(images, texts)
                # text_features = model_out["text_features"]
                text_features = model.module.encode_text(texts, normalize=True).detach()
                logit_scale = model.module.logit_scale.exp()
                losses = {}

                # ----------- multi-GPU running 2024.9.8 -----------
                # all_text_features = gather_features(text_features, rank, world_size)
                all_text_features = torch.cat(GatherLayer.apply(text_features), dim=0)
                all_texts = torch.cat(GatherLayer.apply(texts), dim=0)


                batch_size = images.shape[0]
                all_batch_size = all_texts.shape[0]
                all_image_features = []
                for j in range(batch_size):
                    # loop
                    # image_features_i = []
                    # for k in range(all_batch_size):
                    #     with autocast():
                    #         image_features = model(images[j].unsqueeze(0), None, all_text_features[k].unsqueeze(0))["image_features"]
                    #         image_features_i.append(image_features)
                    # all_image_features.append(torch.concat(image_features_i, dim=0).unsqueeze(0))
                    # batch
                    image_features = model(images[j].unsqueeze(0).repeat(all_batch_size,1,1,1), None, all_text_features)["image_features"]
                    all_image_features.append(image_features.unsqueeze(0))
                all_image_features = torch.cat(all_image_features, dim=0).to(device=device, non_blocking=True)

                # all_image_features = model(images[0].unsqueeze(0).repeat(all_batch_size*batch_size, 1, 1, 1), None
                #                      , all_text_features.repeat(batch_size, 1))["image_features"]
                # all_image_features = gather_features(all_image_features, rank, world_size)
                # all_image_features = torch.cat(GatherLayer.apply(all_image_features), dim=0)

                # criterion = losses_metric_learning.CircleLoss(m=0.4, gamma=80)

                m = 0.4     # Margin
                gamma = 80  # Scaling factor

                delta_p = 1 - m  # Positive margin
                delta_n = m      # Negative margin

                # using all bsz x bsz logits
                # logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_image = all_image_features @ all_text_features.T
                # ipdb.set_trace()
                logits_per_image = torch.cat(GatherLayer.apply(logits_per_image), dim=0)
                # ipdb.set_trace()
                # ----------- multi-GPU running 2024.9.8 -----------
                logits_per_image_flat = torch.reshape(logits_per_image, (all_batch_size, -1))
                labels = torch.arange(all_batch_size, device=device, dtype=torch.long)
                gt_labels = labels * all_batch_size + labels
                # ipdb.set_trace()

                # logits_per_image_flat = torch.clamp(logits_per_image_flat, min=-10, max=10)

                positive_mask = torch.zeros_like(logits_per_image_flat, dtype=torch.bool, device=device)
                positive_mask[torch.arange(positive_mask.shape[0]), gt_labels] = 1
                # for idx_i in range(positive_mask.shape[0]):
                #     positive_mask[idx_i][gt_labels[idx_i]] = 1
                # positive_mask[:,]
                negative_mask = ~positive_mask
                
                s_p = logits_per_image_flat[positive_mask]  # [batch_size]
                s_n = logits_per_image_flat[negative_mask].view(all_batch_size, -1)  # [batch_size, batch_size - 1]
                
                # Compute alpha values
                alpha_p = torch.relu(-s_p + delta_p)  # [batch_size]
                alpha_n = torch.relu(s_n - delta_n)   # [batch_size, batch_size - 1]
                
                # Compute exponential terms
                exp_p = torch.exp(-gamma * alpha_p * (s_p - delta_p))  # [batch_size]
                exp_n = torch.exp(gamma * alpha_n * (s_n - delta_n))   # [batch_size, batch_size - 1]
                
                # Compute loss for each sample
                sum_exp_n = torch.sum(exp_n, dim=1)  # [batch_size]
                loss_per_sample = torch.log1p(exp_p * sum_exp_n)  # [batch_size]
                
                # Compute the mean loss over the batch
                losses['token_loss'] = torch.mean(loss_per_sample)



                # losses['token_loss'] = criterion(logits_per_image_flat, labels)
                # losses['token_loss'] = F.cross_entropy(logits_per_image_flat, gt_labels)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for





def tgvpt_evaluate_reranking(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    print('start evaluation: tgvpt_evaluate_reranking')

    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            all_images, all_texts = [], []
            for i, batch in enumerate(dataloader):
                with autocast():
                    images, texts = batch
                    images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                    texts = texts.to(device=device, non_blocking=True)
                    all_images.append(images)
                    all_texts.append(texts)

                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")
            all_text_features = torch.cat(all_text_features).to(device=device, non_blocking=True)

            all_images = torch.cat(all_images, dim=0)
            all_texts = torch.cat(all_texts, dim=0)

            import pickle
            topk = 100
            org_clip_features = pickle.load(open('DATASET_PATH/clip_vitb16_coco_infer_img_txt_features_9.4.pkl', 'rb'))
            org_image_features = org_clip_features['image_features']
            org_text_features = org_clip_features['text_features']
            logits_per_image = (org_image_features @ org_text_features.t()).detach().cpu()
            logits_per_text = logits_per_image.t().detach().cpu()
            # logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
            ranking = torch.argsort(logits_per_text, descending=True)
            top1_recall = 0
            top5_recall = 0
            top10_recall = 0
            recall_cnt = 0
            recall_topk_curve = {}
            name = 'text_to_image'
            for topi in range(1, 100):
                recall_topk_curve[f"{name}_R@{topi}"] = 0
            for text_i in range(ranking.shape[0]):
                print('text_i: ', text_i)
                topk_img_indices = ranking[text_i][:topk]
                # image_i_2_tg_img_feat = {}
                logit_imgi_list = []
                for image_i in topk_img_indices.detach().cpu().tolist():
                    tg_img_feat = model.encode_image(all_images[image_i].unsqueeze(0), text_embed=all_text_features[text_i].unsqueeze(0), normalize=True)
                    tg_img_feat = tg_img_feat.squeeze()
                    # image_i_2_tg_img_feat[image_i] = tg_img_feat
                    logit_imgi_list.append([(all_text_features[text_i] * tg_img_feat).sum().detach().cpu().tolist(), image_i])
                logit_imgi_list.sort(key=lambda x: -x[0])
                # ipdb.set_trace()
                logit_list = np.array(logit_imgi_list)[:,1].tolist()
                if text_i in logit_list[:1]:
                    print('top1')
                    top1_recall += 1
                if text_i in logit_list[:5]:
                    print('top5')
                    top5_recall += 1
                if text_i in logit_list[:10]:
                    print('top10')
                    top10_recall += 1
                recall_cnt += 1

                for topi in range(1, 100):
                    recall_topk_curve[f"{name}_R@{topi}"] += text_i in logit_list[:topi]
                    
            print('top1_recall: ', top1_recall / recall_cnt)
            print('top5_recall: ', top5_recall / recall_cnt)
            print('top10_recall: ', top10_recall / recall_cnt)

            for topi in range(1, 100):
                 recall_topk_curve[f"{name}_R@{topi}"] =  recall_topk_curve[f"{name}_R@{topi}"] / recall_cnt
           
            ipdb.set_trace()
            with open('PROJECT_PATH/recall_k_coco_data_transformer_best_10.30.json', 'w') as dump_f:
                json.dump(recall_topk_curve, dump_f)
                
    return 0



def tgvpt_evaluate_reranking_standard_coco(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    print('start evaluation: tgvpt_evaluate_reranking')

    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    # # vis 11.8
    # import shutil
    # import pandas as pd
    # df = pd.read_csv(csv_path, sep="\t")
    # raw_images = df['filepath'].tolist()
    # raw_captions = df['title'].tolist()

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():

            # all_images, all_texts = [], []
            # for i, batch in enumerate(dataloader):
            #     with autocast():
            #         images, texts = batch
            #         images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            #         texts = texts.to(device=device, non_blocking=True)
            #         all_images.append(images)
            #         all_texts.append(texts)

            #         model_out = model(images, texts)
            #         image_features = model_out["image_features"]
            #         text_features = model_out["text_features"]
            #         logit_scale = model_out["logit_scale"]
            #         # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
            #         # however, system RAM is easily exceeded and compute time becomes problematic
            #         all_text_features.append(text_features.cpu())
            #         logit_scale = logit_scale.mean()
            #         logits_per_image = logit_scale * image_features @ text_features.t()
            #         logits_per_text = logits_per_image.t()

            #         batch_size = images.shape[0]
            #         labels = torch.arange(batch_size, device=device).long()
            #         total_loss = (
            #             F.cross_entropy(logits_per_image, labels) +
            #             F.cross_entropy(logits_per_text, labels)
            #         ) / 2

            #         gen_loss = maybe_compute_generative_loss(model_out)

            #     cumulative_loss += total_loss * batch_size
            #     num_samples += batch_size
            #     if is_master(args) and (i % 100) == 0:
            #         logging.info(
            #             f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
            #             f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

            #         if gen_loss is not None:
            #             cumulative_gen_loss += gen_loss * batch_size
            #             logging.info(
            #                 f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")
            # all_text_features = torch.cat(all_text_features).to(device=device, non_blocking=True)

            # all_images = torch.cat(all_images, dim=0)
            # all_texts = torch.cat(all_texts, dim=0)

            import pickle
            txti_2_imgi_json = json.load(open('standard_benchmarks/coco/coco_test_txtid2imgid.json'))
            topk = 100
            num_total_img = 5000
            org_clip_features = pickle.load(open('standard_benchmarks/coco/clip_vitb16_standard_coco_infer_img_txt_features_10.24.pkl', 'rb'))
            org_image_features = org_clip_features['image_features'][:num_total_img].cuda()
            org_text_features = org_clip_features['text_features'].cuda()
            logits_per_image = (org_image_features @ org_text_features.t()).detach().cpu()
            logits_per_text = logits_per_image.t().detach().cpu()
            # logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
            ranking = torch.argsort(logits_per_text, descending=True)
            top1_recall = 0
            top5_recall = 0
            top10_recall = 0
            recall_cnt = 0
            recall_topk_curve = {}
            name = 'text_to_image'
            for topi in range(1, 100):
                recall_topk_curve[f"{name}_R@{topi}"] = 0
            

            for text_i in range(ranking.shape[0]):
                print('text_i: ', text_i)
                topk_img_indices = ranking[text_i][:topk]
                # image_i_2_tg_img_feat = {}
                logit_imgi_list = []


                # if txti_2_imgi_json[str(text_i)] in topk_img_indices:
                #     recall_topk_initial += 1
                #     print('initial recalled within topk')
                # recall_cnt += 1
                # continue


                # # for original CLIP
                # image_iter_i = 0
                # for image_i in topk_img_indices.detach().cpu().tolist():
                #     logit_imgi_list.append([(org_text_features[text_i] * org_image_features[image_i]).sum().detach().cpu().tolist(), image_i])
                #     image_iter_i += 1

                # new implementation - faster 10.24
                forward_inference_img_list = []
                for image_i in topk_img_indices.detach().cpu().tolist():
                    forward_inference_img_list.append(dataloader.dataset[image_i][0].unsqueeze(0))
                forward_inference_img_list = torch.cat(forward_inference_img_list, dim=0).cuda()
                tg_img_feat = model.encode_image(forward_inference_img_list.cuda(), text_embed=org_text_features[text_i].unsqueeze(0).repeat(topk, 1), normalize=True)
                image_iter_i = 0
                for image_i in topk_img_indices.detach().cpu().tolist():
                    logit_imgi_list.append([(org_text_features[text_i] * tg_img_feat[image_iter_i]).sum().detach().cpu().tolist(), image_i])
                    image_iter_i += 1

                # old implementation
                # for image_i in topk_img_indices.detach().cpu().tolist():
                #     # ipdb.set_trace()
                #     tg_img_feat = model.encode_image(dataloader.dataset[image_i][0].unsqueeze(0).cuda(), text_embed=org_text_features[text_i].unsqueeze(0), normalize=True)
                #     tg_img_feat = tg_img_feat.squeeze()
                #     # image_i_2_tg_img_feat[image_i] = tg_img_feat
                #     logit_imgi_list.append([(org_text_features[text_i] * tg_img_feat).sum().detach().cpu().tolist(), image_i])
                
                # ipdb.set_trace()
                # print(topk_img_indices[:10])
                logit_imgi_list.sort(key=lambda x: -x[0])
                # print(logit_imgi_list[:10])
                # ipdb.set_trace()
                logit_list = np.array(logit_imgi_list)[:,1].tolist()
                if txti_2_imgi_json[str(text_i)] in logit_list[:1]:
                    print('top1')
                    top1_recall += 1
                if txti_2_imgi_json[str(text_i)] in logit_list[:5]:
                    print('top5')
                    top5_recall += 1
                if txti_2_imgi_json[str(text_i)] in logit_list[:10]:
                    print('top10')
                    top10_recall += 1
                recall_cnt += 1

                # # for vis
                # if text_i < 5000:
                #     if txti_2_imgi_json[str(text_i)] in logit_list[:1] and txti_2_imgi_json[str(text_i)] not in topk_img_indices[:5]: 
                #         # ipdb.set_trace()
                #         for vis_img_id in range(10):
                #             if '/' not in raw_captions[int(logit_list[vis_img_id])] and '/' not in raw_captions[int(topk_img_indices[vis_img_id].item())]:
                #                 shutil.copy(raw_images[int(logit_list[vis_img_id])], os.path.join(vis_dir, 'txti_' + str(text_i) + '-after-top_' + str(vis_img_id) + '-csvid_' + str(int(logit_list[vis_img_id])) + '-caption_' + raw_captions[int(logit_list[vis_img_id])] + '.png'))
                #                 shutil.copy(raw_images[int(topk_img_indices[vis_img_id].item())], os.path.join(vis_dir, 'txti_' + str(text_i) + '-before-top_' + str(vis_img_id) + '-csvid_' + str(int(topk_img_indices[vis_img_id].item())) + '-caption_' + raw_captions[int(topk_img_indices[vis_img_id].item())] + '.png'))

                    # ipdb.set_trace()
                for topi in range(1, 100):
                    recall_topk_curve[f"{name}_R@{topi}"] += txti_2_imgi_json[str(text_i)] in logit_list[:topi]
                    
            # print('initial recall@topk: ', recall_topk_initial / recall_cnt)


            print('top1_recall: ', top1_recall / recall_cnt)
            print('top5_recall: ', top5_recall / recall_cnt)
            print('top10_recall: ', top10_recall / recall_cnt)

            for topi in range(1, 100):
                recall_topk_curve[f"{name}_R@{topi}"] =  recall_topk_curve[f"{name}_R@{topi}"] / recall_cnt
           
            ipdb.set_trace()
            with open('PROJECT_PATH/recall_k_standard_coco_ours_2prompt_2.25.json', 'w') as dump_f:
                json.dump(recall_topk_curve, dump_f)
                
    return 0



def tgvpt_evaluate_reranking_standard_coco_transformer(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    print('start evaluation: tgvpt_evaluate_reranking')

    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    # # vis 11.8
    # import shutil
    # import pandas as pd
    # df = pd.read_csv(csv_path, sep="\t")
    # raw_images = df['filepath'].tolist()
    # raw_captions = df['title'].tolist()

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():

            all_images, all_texts = [], []
            for i, batch in enumerate(dataloader):
                with autocast():
                    images, texts = batch
                    images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                    texts = texts.to(device=device, non_blocking=True)
                    all_images.append(images)
                    all_texts.append(texts)

                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    # logits_per_image = logit_scale * image_features @ text_features.t()
                    # logits_per_text = logits_per_image.t()

                    # batch_size = images.shape[0]
                    # labels = torch.arange(batch_size, device=device).long()
                    # total_loss = (
                    #     F.cross_entropy(logits_per_image, labels) +
                    #     F.cross_entropy(logits_per_text, labels)
                    # ) / 2

                    # gen_loss = maybe_compute_generative_loss(model_out)

                # cumulative_loss += total_loss * batch_size
                # num_samples += batch_size
                # if is_master(args) and (i % 100) == 0:
                #     logging.info(
                #         f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                #         f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                #     if gen_loss is not None:
                #         cumulative_gen_loss += gen_loss * batch_size
                #         logging.info(
                #             f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")
            all_text_features = torch.cat(all_text_features).to(device=device, non_blocking=True)

            all_images = torch.cat(all_images, dim=0)
            all_texts = torch.cat(all_texts, dim=0)

            import pickle
            txti_2_imgi_json = json.load(open('DATASET_PATH/coco_test_txtid2imgid.json'))
            topk = 100
            num_total_img = 5000
            org_clip_features = pickle.load(open('DATASET_PATH/clip_vitb16_standard_coco_infer_img_txt_features_10.24.pkl', 'rb'))
            org_image_features = org_clip_features['image_features'][:num_total_img].cuda()
            org_text_features = org_clip_features['text_features'].cuda()
            logits_per_image = (org_image_features @ org_text_features.t()).detach().cpu()
            logits_per_text = logits_per_image.t().detach().cpu()
            # logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
            ranking = torch.argsort(logits_per_text, descending=True)
            top1_recall = 0
            top5_recall = 0
            top10_recall = 0
            recall_cnt = 0
            recall_topk_curve = {}
            name = 'text_to_image'
            for topi in range(1, 100):
                recall_topk_curve[f"{name}_R@{topi}"] = 0
            

            for text_i in range(ranking.shape[0]):
                print('text_i: ', text_i)
                topk_img_indices = ranking[text_i][:topk]
                # image_i_2_tg_img_feat = {}
                logit_imgi_list = []


                # if txti_2_imgi_json[str(text_i)] in topk_img_indices:
                #     recall_topk_initial += 1
                #     print('initial recalled within topk')
                # recall_cnt += 1
                # continue


                # # for original CLIP
                # image_iter_i = 0
                # for image_i in topk_img_indices.detach().cpu().tolist():
                #     logit_imgi_list.append([(org_text_features[text_i] * org_image_features[image_i]).sum().detach().cpu().tolist(), image_i])
                #     image_iter_i += 1

                # new implementation - faster 10.24
                forward_inference_img_list = []
                for image_i in topk_img_indices.detach().cpu().tolist():
                    forward_inference_img_list.append(dataloader.dataset[image_i][0].unsqueeze(0))
                forward_inference_img_list = torch.cat(forward_inference_img_list, dim=0).cuda()
                tg_img_feat = model.encode_image(forward_inference_img_list.cuda(), text_embed=all_text_features[text_i].unsqueeze(0).repeat(topk, 1), normalize=True)
                image_iter_i = 0
                for image_i in topk_img_indices.detach().cpu().tolist():
                    logit_imgi_list.append([(org_text_features[text_i] * tg_img_feat[image_iter_i]).sum().detach().cpu().tolist(), image_i])
                    image_iter_i += 1

                # old implementation
                # for image_i in topk_img_indices.detach().cpu().tolist():
                #     # ipdb.set_trace()
                #     tg_img_feat = model.encode_image(dataloader.dataset[image_i][0].unsqueeze(0).cuda(), text_embed=org_text_features[text_i].unsqueeze(0), normalize=True)
                #     tg_img_feat = tg_img_feat.squeeze()
                #     # image_i_2_tg_img_feat[image_i] = tg_img_feat
                #     logit_imgi_list.append([(org_text_features[text_i] * tg_img_feat).sum().detach().cpu().tolist(), image_i])
                
                # ipdb.set_trace()
                # print(topk_img_indices[:10])
                logit_imgi_list.sort(key=lambda x: -x[0])
                # print(logit_imgi_list[:10])
                # ipdb.set_trace()
                logit_list = np.array(logit_imgi_list)[:,1].tolist()
                if txti_2_imgi_json[str(text_i)] in logit_list[:1]:
                    print('top1')
                    top1_recall += 1
                if txti_2_imgi_json[str(text_i)] in logit_list[:5]:
                    print('top5')
                    top5_recall += 1
                if txti_2_imgi_json[str(text_i)] in logit_list[:10]:
                    print('top10')
                    top10_recall += 1
                recall_cnt += 1

                # # for vis
                # if text_i < 5000:
                #     if txti_2_imgi_json[str(text_i)] in logit_list[:1] and txti_2_imgi_json[str(text_i)] not in topk_img_indices[:5]: 
                #         # ipdb.set_trace()
                #         for vis_img_id in range(10):
                #             if '/' not in raw_captions[int(logit_list[vis_img_id])] and '/' not in raw_captions[int(topk_img_indices[vis_img_id].item())]:
                #                 shutil.copy(raw_images[int(logit_list[vis_img_id])], os.path.join(vis_dir, 'txti_' + str(text_i) + '-after-top_' + str(vis_img_id) + '-csvid_' + str(int(logit_list[vis_img_id])) + '-caption_' + raw_captions[int(logit_list[vis_img_id])] + '.png'))
                #                 shutil.copy(raw_images[int(topk_img_indices[vis_img_id].item())], os.path.join(vis_dir, 'txti_' + str(text_i) + '-before-top_' + str(vis_img_id) + '-csvid_' + str(int(topk_img_indices[vis_img_id].item())) + '-caption_' + raw_captions[int(topk_img_indices[vis_img_id].item())] + '.png'))

                    # ipdb.set_trace()
                for topi in range(1, 100):
                    recall_topk_curve[f"{name}_R@{topi}"] += txti_2_imgi_json[str(text_i)] in logit_list[:topi]
                    
            # print('initial recall@topk: ', recall_topk_initial / recall_cnt)


            print('top1_recall: ', top1_recall / recall_cnt)
            print('top5_recall: ', top5_recall / recall_cnt)
            print('top10_recall: ', top10_recall / recall_cnt)

            for topi in range(1, 100):
                 recall_topk_curve[f"{name}_R@{topi}"] =  recall_topk_curve[f"{name}_R@{topi}"] / recall_cnt
           
            ipdb.set_trace()
            with open('PROJECT_PATH/recall_k_standard_coco_ours_11.14.json', 'w') as dump_f:
                json.dump(recall_topk_curve, dump_f)
                
    return 0




def tgvpt_evaluate_reranking_standard_coco_siglip(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    print('start evaluation: tgvpt_evaluate_reranking')

    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    # # vis 11.8
    # import shutil
    # import pandas as pd
    # df = pd.read_csv(csv_path, sep="\t")
    # raw_images = df['filepath'].tolist()
    # raw_captions = df['title'].tolist()

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():

            # all_images, all_texts = [], []
            # for i, batch in enumerate(dataloader):
            #     with autocast():
            #         images, texts = batch
            #         images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            #         texts = texts.to(device=device, non_blocking=True)
            #         all_images.append(images)
            #         all_texts.append(texts)

            #         model_out = model(images, texts)
            #         image_features = model_out["image_features"]
            #         text_features = model_out["text_features"]
            #         logit_scale = model_out["logit_scale"]
            #         # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
            #         # however, system RAM is easily exceeded and compute time becomes problematic
            #         all_text_features.append(text_features.cpu())
            #         logit_scale = logit_scale.mean()
            #         logits_per_image = logit_scale * image_features @ text_features.t()
            #         logits_per_text = logits_per_image.t()

            #         batch_size = images.shape[0]
            #         labels = torch.arange(batch_size, device=device).long()
            #         total_loss = (
            #             F.cross_entropy(logits_per_image, labels) +
            #             F.cross_entropy(logits_per_text, labels)
            #         ) / 2

            #         gen_loss = maybe_compute_generative_loss(model_out)

            #     cumulative_loss += total_loss * batch_size
            #     num_samples += batch_size
            #     if is_master(args) and (i % 100) == 0:
            #         logging.info(
            #             f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
            #             f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

            #         if gen_loss is not None:
            #             cumulative_gen_loss += gen_loss * batch_size
            #             logging.info(
            #                 f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")
            # all_text_features = torch.cat(all_text_features).to(device=device, non_blocking=True)

            # all_images = torch.cat(all_images, dim=0)
            # all_texts = torch.cat(all_texts, dim=0)

            import pickle
            txti_2_imgi_json = json.load(open('standard_benchmarks/coco/coco_test_txtid2imgid.json'))
            topk = 100
            num_total_img = 5000
            # org_clip_features = pickle.load(open('standard_benchmarks/coco/siglipSO400M_standard_coco_infer_img_txt_features_2.26.pkl', 'rb'))
            org_clip_features = pickle.load(open('standard_benchmarks/coco/siglip2G_standard_coco_infer_img_txt_features_2.26.pkl', 'rb'))
            org_image_features = org_clip_features['image_features'][:num_total_img].cuda()
            org_text_features = org_clip_features['text_features'].cuda()
            logits_per_image = (org_image_features @ org_text_features.t()).detach().cpu()
            logits_per_text = logits_per_image.t().detach().cpu()
            # logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
            ranking = torch.argsort(logits_per_text, descending=True)
            top1_recall = 0
            top5_recall = 0
            top10_recall = 0
            recall_cnt = 0
            recall_topk_curve = {}
            name = 'text_to_image'
            for topi in range(1, 100):
            # for topi in range(1, 501):
                recall_topk_curve[f"{name}_R@{topi}"] = 0
            

            for text_i in range(ranking.shape[0]):
                print('text_i: ', text_i)
                topk_img_indices = ranking[text_i][:topk]
                # image_i_2_tg_img_feat = {}
                logit_imgi_list = []


                # if txti_2_imgi_json[str(text_i)] in topk_img_indices:
                #     recall_topk_initial += 1
                #     print('initial recalled within topk')
                # recall_cnt += 1
                # continue


                # for original CLIP
                # image_iter_i = 0
                # for image_i in topk_img_indices.detach().cpu().tolist():
                #     logit_imgi_list.append([(org_text_features[text_i] * org_image_features[image_i]).sum().detach().cpu().tolist(), image_i])
                #     image_iter_i += 1

                # new implementation - faster 10.24
                forward_inference_img_list = []
                for image_i in topk_img_indices.detach().cpu().tolist():
                    forward_inference_img_list.append(dataloader.dataset[image_i][0].unsqueeze(0))
                forward_inference_img_list = torch.cat(forward_inference_img_list, dim=0).cuda()
                tg_img_feat = model.encode_image(forward_inference_img_list.cuda(), text_feat=org_text_features[text_i].unsqueeze(0).repeat(topk, 1), normalize=True)
                image_iter_i = 0
                for image_i in topk_img_indices.detach().cpu().tolist():
                    logit_imgi_list.append([(org_text_features[text_i] * tg_img_feat[image_iter_i]).sum().detach().cpu().tolist(), image_i])
                    image_iter_i += 1

                # old implementation
                # for image_i in topk_img_indices.detach().cpu().tolist():
                #     # ipdb.set_trace()
                #     tg_img_feat = model.encode_image(dataloader.dataset[image_i][0].unsqueeze(0).cuda(), text_embed=org_text_features[text_i].unsqueeze(0), normalize=True)
                #     tg_img_feat = tg_img_feat.squeeze()
                #     # image_i_2_tg_img_feat[image_i] = tg_img_feat
                #     logit_imgi_list.append([(org_text_features[text_i] * tg_img_feat).sum().detach().cpu().tolist(), image_i])
                
                # ipdb.set_trace()
                # print(topk_img_indices[:10])
                logit_imgi_list.sort(key=lambda x: -x[0])
                # print(logit_imgi_list[:10])
                # ipdb.set_trace()
                logit_list = np.array(logit_imgi_list)[:,1].tolist()
                if txti_2_imgi_json[str(text_i)] in logit_list[:1]:
                    print('top1')
                    top1_recall += 1
                if txti_2_imgi_json[str(text_i)] in logit_list[:5]:
                    print('top5')
                    top5_recall += 1
                if txti_2_imgi_json[str(text_i)] in logit_list[:10]:
                    print('top10')
                    top10_recall += 1
                recall_cnt += 1

                # # for vis
                # if text_i < 5000:
                #     if txti_2_imgi_json[str(text_i)] in logit_list[:1] and txti_2_imgi_json[str(text_i)] not in topk_img_indices[:5]: 
                #         # ipdb.set_trace()
                #         for vis_img_id in range(10):
                #             if '/' not in raw_captions[int(logit_list[vis_img_id])] and '/' not in raw_captions[int(topk_img_indices[vis_img_id].item())]:
                #                 shutil.copy(raw_images[int(logit_list[vis_img_id])], os.path.join(vis_dir, 'txti_' + str(text_i) + '-after-top_' + str(vis_img_id) + '-csvid_' + str(int(logit_list[vis_img_id])) + '-caption_' + raw_captions[int(logit_list[vis_img_id])] + '.png'))
                #                 shutil.copy(raw_images[int(topk_img_indices[vis_img_id].item())], os.path.join(vis_dir, 'txti_' + str(text_i) + '-before-top_' + str(vis_img_id) + '-csvid_' + str(int(topk_img_indices[vis_img_id].item())) + '-caption_' + raw_captions[int(topk_img_indices[vis_img_id].item())] + '.png'))

                    # ipdb.set_trace()
                for topi in range(1, 100):
                # for topi in range(1, 501):
                    recall_topk_curve[f"{name}_R@{topi}"] += txti_2_imgi_json[str(text_i)] in logit_list[:topi]
                    
            # print('initial recall@topk: ', recall_topk_initial / recall_cnt)


            print('top1_recall: ', top1_recall / recall_cnt)
            print('top5_recall: ', top5_recall / recall_cnt)
            print('top10_recall: ', top10_recall / recall_cnt)

            for topi in range(1, 100):
            # for topi in range(1, 501):
                 recall_topk_curve[f"{name}_R@{topi}"] =  recall_topk_curve[f"{name}_R@{topi}"] / recall_cnt
           
            ipdb.set_trace()
            with open('PROJECT_PATH/recall_k_standard_coco_siglipSO_1prompt_2.25.json', 'w') as dump_f:
                json.dump(recall_topk_curve, dump_f)
                
    return 0




def tgvpt_evaluate_reranking_standard_flickr(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    print('start evaluation: tgvpt_evaluate_reranking')

    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    # # vis 11.8
    # import shutil
    # import pandas as pd
    # df = pd.read_csv(csv_path, sep="\t")
    # raw_images = df['filepath'].tolist()
    # raw_captions = df['title'].tolist()

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():

            # all_images, all_texts = [], []
            # for i, batch in enumerate(dataloader):
            #     with autocast():
            #         images, texts = batch
            #         images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            #         texts = texts.to(device=device, non_blocking=True)
            #         all_images.append(images)
            #         all_texts.append(texts)

            #         model_out = model(images, texts)
            #         image_features = model_out["image_features"]
            #         text_features = model_out["text_features"]
            #         logit_scale = model_out["logit_scale"]
            #         # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
            #         # however, system RAM is easily exceeded and compute time becomes problematic
            #         all_text_features.append(text_features.cpu())
            #         logit_scale = logit_scale.mean()
            #         logits_per_image = logit_scale * image_features @ text_features.t()
            #         logits_per_text = logits_per_image.t()

            #         batch_size = images.shape[0]
            #         labels = torch.arange(batch_size, device=device).long()
            #         total_loss = (
            #             F.cross_entropy(logits_per_image, labels) +
            #             F.cross_entropy(logits_per_text, labels)
            #         ) / 2

            #         gen_loss = maybe_compute_generative_loss(model_out)

            #     cumulative_loss += total_loss * batch_size
            #     num_samples += batch_size
            #     if is_master(args) and (i % 100) == 0:
            #         logging.info(
            #             f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
            #             f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

            #         if gen_loss is not None:
            #             cumulative_gen_loss += gen_loss * batch_size
            #             logging.info(
            #                 f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")
            # all_text_features = torch.cat(all_text_features).to(device=device, non_blocking=True)

            # all_images = torch.cat(all_images, dim=0)
            # all_texts = torch.cat(all_texts, dim=0)

            import pickle
            txti_2_imgi_json = json.load(open('standard_benchmarks/flickr/flickr_test_txtid2imgid.json'))
            topk = 100
            num_total_img = 1000
            org_clip_features = pickle.load(open('standard_benchmarks/flickr/clip_vitb16_standard_flickr_infer_img_txt_features_10.24.pkl', 'rb'))
            org_image_features = org_clip_features['image_features'][:num_total_img].cuda()
            org_text_features = org_clip_features['text_features'].cuda()
            logits_per_image = (org_image_features @ org_text_features.t()).detach().cpu()
            logits_per_text = logits_per_image.t().detach().cpu()
            # logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
            ranking = torch.argsort(logits_per_text, descending=True)
            top1_recall = 0
            top5_recall = 0
            top10_recall = 0
            recall_cnt = 0
            recall_topk_curve = {}
            name = 'text_to_image'
            for topi in range(1, 100):
                recall_topk_curve[f"{name}_R@{topi}"] = 0


            for text_i in range(ranking.shape[0]):
                print('text_i: ', text_i)
                topk_img_indices = ranking[text_i][:topk]
                # image_i_2_tg_img_feat = {}
                logit_imgi_list = []

                # if txti_2_imgi_json[str(text_i)] in topk_img_indices:
                #     recall_topk_initial += 1
                #     print('initial recalled within topk')
                # recall_cnt += 1
                # continue

                # # for original CLIP
                # image_iter_i = 0
                # for image_i in topk_img_indices.detach().cpu().tolist():
                #     logit_imgi_list.append([(org_text_features[text_i] * org_image_features[image_i]).sum().detach().cpu().tolist(), image_i])
                #     image_iter_i += 1

                # new implementation - faster 10.24
                forward_inference_img_list = []
                for image_i in topk_img_indices.detach().cpu().tolist():
                    forward_inference_img_list.append(dataloader.dataset[image_i][0].unsqueeze(0))
                forward_inference_img_list = torch.cat(forward_inference_img_list, dim=0).cuda()
                tg_img_feat = model.encode_image(forward_inference_img_list.cuda(), text_embed=org_text_features[text_i].unsqueeze(0).repeat(topk, 1), normalize=True)
                image_iter_i = 0
                for image_i in topk_img_indices.detach().cpu().tolist():
                    logit_imgi_list.append([(org_text_features[text_i] * tg_img_feat[image_iter_i]).sum().detach().cpu().tolist(), image_i])
                    image_iter_i += 1

                # old implementation
                # for image_i in topk_img_indices.detach().cpu().tolist():
                #     # ipdb.set_trace()
                #     tg_img_feat = model.encode_image(dataloader.dataset[image_i][0].unsqueeze(0).cuda(), text_embed=org_text_features[text_i].unsqueeze(0), normalize=True)
                #     tg_img_feat = tg_img_feat.squeeze()
                #     # image_i_2_tg_img_feat[image_i] = tg_img_feat
                #     logit_imgi_list.append([(org_text_features[text_i] * tg_img_feat).sum().detach().cpu().tolist(), image_i])
                
                logit_imgi_list.sort(key=lambda x: -x[0])
                # ipdb.set_trace()
                logit_list = np.array(logit_imgi_list)[:,1].tolist()
                if txti_2_imgi_json[str(text_i)] in logit_list[:1]:
                    print('top1')
                    top1_recall += 1
                if txti_2_imgi_json[str(text_i)] in logit_list[:5]:
                    print('top5')
                    top5_recall += 1
                if txti_2_imgi_json[str(text_i)] in logit_list[:10]:
                    print('top10')
                    top10_recall += 1
                recall_cnt += 1

                # # for vis
                # if text_i < 5000:
                #     if txti_2_imgi_json[str(text_i)] in logit_list[:1] and txti_2_imgi_json[str(text_i)] not in topk_img_indices[:2]: 
                #         # ipdb.set_trace()
                #         print("get a example for flickr")
                #         for vis_img_id in range(10):
                #             if '/' not in raw_captions[int(logit_list[vis_img_id])] and '/' not in raw_captions[int(topk_img_indices[vis_img_id].item())] and len(raw_captions[int(logit_list[vis_img_id])]) < 100 and len(raw_captions[int(topk_img_indices[vis_img_id].item())]) < 100:
                #                 shutil.copy(raw_images[int(logit_list[vis_img_id])], os.path.join(vis_dir, 'txti_' + str(text_i) + '-after-top_' + str(vis_img_id) + '-csvid_' + str(int(logit_list[vis_img_id])) + '-caption_' + raw_captions[int(logit_list[vis_img_id])] + '.png'))
                #                 shutil.copy(raw_images[int(topk_img_indices[vis_img_id].item())], os.path.join(vis_dir, 'txti_' + str(text_i) + '-before-top_' + str(vis_img_id) + '-csvid_' + str(int(topk_img_indices[vis_img_id].item())) + '-caption_' + raw_captions[int(topk_img_indices[vis_img_id].item())] + '.png'))

                    # ipdb.set_trace()

                for topi in range(1, 100):
                    recall_topk_curve[f"{name}_R@{topi}"] += txti_2_imgi_json[str(text_i)] in logit_list[:topi]
                    
            # print('initial recall@topk: ', recall_topk_initial / recall_cnt)

            print('top1_recall: ', top1_recall / recall_cnt)
            print('top5_recall: ', top5_recall / recall_cnt)
            print('top10_recall: ', top10_recall / recall_cnt)

            for topi in range(1, 100):
                recall_topk_curve[f"{name}_R@{topi}"] =  recall_topk_curve[f"{name}_R@{topi}"] / recall_cnt
           
            ipdb.set_trace()
            with open('PROJECT_PATH/recall_k_standard_flickr_org_clip_11.14.json', 'w') as dump_f:
                json.dump(recall_topk_curve, dump_f)
                
    return 0



def tgvpt_evaluate_reranking_standard_flickr_2txtfeat(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    print('start evaluation: tgvpt_evaluate_reranking')

    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    # # vis 11.8
    # import shutil
    # import pandas as pd
    # df = pd.read_csv(csv_path, sep="\t")
    # raw_images = df['filepath'].tolist()
    # raw_captions = df['title'].tolist()

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        # cumulative_loss = 0.0
        # cumulative_gen_loss = 0.0
        # all_image_features, all_text_features = [], []
        all_text_features_tgvpt = []
        with torch.no_grad():

            # all_images, all_texts = [], []
            for i, batch in enumerate(dataloader):
                with autocast():
                    images, texts = batch
                    images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                    texts = texts.to(device=device, non_blocking=True)
                    # ipdb.set_trace()
                    text_features, text_features_tgvpt = model.encode_text(texts, normalize=True)#.detach()
                    all_text_features_tgvpt.append(text_features_tgvpt.cpu())
                    # all_images.append(images)
                    # all_texts.append(texts)

                    # model_out = model(images, texts)
                    # image_features = model_out["image_features"]
                    # text_features = model_out["text_features"]
                    # logit_scale = model_out["logit_scale"]
                    # # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # # however, system RAM is easily exceeded and compute time becomes problematic
                    # all_text_features.append(text_features.cpu())
                    # logit_scale = logit_scale.mean()
                    # logits_per_image = logit_scale * image_features @ text_features.t()
                    # logits_per_text = logits_per_image.t()

            #         batch_size = images.shape[0]
            #         labels = torch.arange(batch_size, device=device).long()
            #         total_loss = (
            #             F.cross_entropy(logits_per_image, labels) +
            #             F.cross_entropy(logits_per_text, labels)
            #         ) / 2

            #         gen_loss = maybe_compute_generative_loss(model_out)

            #     cumulative_loss += total_loss * batch_size
            #     num_samples += batch_size
            #     if is_master(args) and (i % 100) == 0:
            #         logging.info(
            #             f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
            #             f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

            #         if gen_loss is not None:
            #             cumulative_gen_loss += gen_loss * batch_size
            #             logging.info(
            #                 f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")
            # all_text_features = torch.cat(all_text_features).to(device=device, non_blocking=True)
            all_text_features_tgvpt = torch.cat(all_text_features_tgvpt).to(device=device, non_blocking=True)

            # all_images = torch.cat(all_images, dim=0)
            # all_texts = torch.cat(all_texts, dim=0)

            import pickle
            txti_2_imgi_json = json.load(open('DATASET_PATH/flickr_test_txtid2imgid.json'))
            topk = 100
            num_total_img = 1000
            org_clip_features = pickle.load(open('DATASET_PATH/clip_vitb16_standard_flickr_infer_img_txt_features_10.24.pkl', 'rb'))
            org_image_features = org_clip_features['image_features'][:num_total_img].cuda()
            org_text_features = org_clip_features['text_features'].cuda()
            logits_per_image = (org_image_features @ org_text_features.t()).detach().cpu()
            logits_per_text = logits_per_image.t().detach().cpu()
            # logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
            ranking = torch.argsort(logits_per_text, descending=True)
            top1_recall = 0
            top5_recall = 0
            top10_recall = 0
            recall_cnt = 0
            recall_topk_curve = {}
            name = 'text_to_image'
            for topi in range(1, 100):
                recall_topk_curve[f"{name}_R@{topi}"] = 0


            for text_i in range(ranking.shape[0]):
                print('text_i: ', text_i)
                topk_img_indices = ranking[text_i][:topk]
                # image_i_2_tg_img_feat = {}
                logit_imgi_list = []

                # if txti_2_imgi_json[str(text_i)] in topk_img_indices:
                #     recall_topk_initial += 1
                #     print('initial recalled within topk')
                # recall_cnt += 1
                # continue

                # # for original CLIP
                # image_iter_i = 0
                # for image_i in topk_img_indices.detach().cpu().tolist():
                #     logit_imgi_list.append([(org_text_features[text_i] * org_image_features[image_i]).sum().detach().cpu().tolist(), image_i])
                #     image_iter_i += 1

                # new implementation - faster 10.24
                forward_inference_img_list = []
                for image_i in topk_img_indices.detach().cpu().tolist():
                    forward_inference_img_list.append(dataloader.dataset[image_i][0].unsqueeze(0))
                forward_inference_img_list = torch.cat(forward_inference_img_list, dim=0).cuda()
                tg_img_feat = model.encode_image(forward_inference_img_list.cuda(), text_embed=all_text_features_tgvpt[text_i].unsqueeze(0).repeat(topk, 1), normalize=True)
                image_iter_i = 0
                for image_i in topk_img_indices.detach().cpu().tolist():
                    logit_imgi_list.append([(org_text_features[text_i] * tg_img_feat[image_iter_i]).sum().detach().cpu().tolist(), image_i])
                    image_iter_i += 1

                # old implementation
                # for image_i in topk_img_indices.detach().cpu().tolist():
                #     # ipdb.set_trace()
                #     tg_img_feat = model.encode_image(dataloader.dataset[image_i][0].unsqueeze(0).cuda(), text_embed=org_text_features[text_i].unsqueeze(0), normalize=True)
                #     tg_img_feat = tg_img_feat.squeeze()
                #     # image_i_2_tg_img_feat[image_i] = tg_img_feat
                #     logit_imgi_list.append([(org_text_features[text_i] * tg_img_feat).sum().detach().cpu().tolist(), image_i])
                
                logit_imgi_list.sort(key=lambda x: -x[0])
                # ipdb.set_trace()
                logit_list = np.array(logit_imgi_list)[:,1].tolist()
                if txti_2_imgi_json[str(text_i)] in logit_list[:1]:
                    print('top1')
                    top1_recall += 1
                if txti_2_imgi_json[str(text_i)] in logit_list[:5]:
                    print('top5')
                    top5_recall += 1
                if txti_2_imgi_json[str(text_i)] in logit_list[:10]:
                    print('top10')
                    top10_recall += 1
                recall_cnt += 1

                # # for vis
                # if text_i < 5000:
                #     if txti_2_imgi_json[str(text_i)] in logit_list[:1] and txti_2_imgi_json[str(text_i)] not in topk_img_indices[:2]: 
                #         # ipdb.set_trace()
                #         print("get a example for flickr")
                #         for vis_img_id in range(10):
                #             if '/' not in raw_captions[int(logit_list[vis_img_id])] and '/' not in raw_captions[int(topk_img_indices[vis_img_id].item())] and len(raw_captions[int(logit_list[vis_img_id])]) < 100 and len(raw_captions[int(topk_img_indices[vis_img_id].item())]) < 100:
                #                 shutil.copy(raw_images[int(logit_list[vis_img_id])], os.path.join(vis_dir, 'txti_' + str(text_i) + '-after-top_' + str(vis_img_id) + '-csvid_' + str(int(logit_list[vis_img_id])) + '-caption_' + raw_captions[int(logit_list[vis_img_id])] + '.png'))
                #                 shutil.copy(raw_images[int(topk_img_indices[vis_img_id].item())], os.path.join(vis_dir, 'txti_' + str(text_i) + '-before-top_' + str(vis_img_id) + '-csvid_' + str(int(topk_img_indices[vis_img_id].item())) + '-caption_' + raw_captions[int(topk_img_indices[vis_img_id].item())] + '.png'))

                    # ipdb.set_trace()

                for topi in range(1, 100):
                    recall_topk_curve[f"{name}_R@{topi}"] += txti_2_imgi_json[str(text_i)] in logit_list[:topi]
                    
            # print('initial recall@topk: ', recall_topk_initial / recall_cnt)

            print('top1_recall: ', top1_recall / recall_cnt)
            print('top5_recall: ', top5_recall / recall_cnt)
            print('top10_recall: ', top10_recall / recall_cnt)

            for topi in range(1, 100):
                recall_topk_curve[f"{name}_R@{topi}"] =  recall_topk_curve[f"{name}_R@{topi}"] / recall_cnt
           
            ipdb.set_trace()
            with open('PROJECT_PATH/recall_k_standard_flickr_org_clip_11.14.json', 'w') as dump_f:
                json.dump(recall_topk_curve, dump_f)
                
    return 0




def tgvpt_evaluate_reranking_capsbench(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    print('start evaluation: tgvpt_evaluate_reranking')

    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    # # vis 11.8
    # import shutil
    # import pandas as pd
    # df = pd.read_csv(csv_path, sep="\t")
    # raw_images = df['filepath'].tolist()
    # raw_captions = df['title'].tolist()

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():

            # all_images, all_texts = [], []
            # for i, batch in enumerate(dataloader):
            #     with autocast():
            #         images, texts = batch
            #         images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            #         texts = texts.to(device=device, non_blocking=True)
            #         all_images.append(images)
            #         all_texts.append(texts)

            #         model_out = model(images, texts)
            #         image_features = model_out["image_features"]
            #         text_features = model_out["text_features"]
            #         logit_scale = model_out["logit_scale"]
            #         # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
            #         # however, system RAM is easily exceeded and compute time becomes problematic
            #         all_text_features.append(text_features.cpu())
            #         logit_scale = logit_scale.mean()
            #         logits_per_image = logit_scale * image_features @ text_features.t()
            #         logits_per_text = logits_per_image.t()

            #         batch_size = images.shape[0]
            #         labels = torch.arange(batch_size, device=device).long()
            #         total_loss = (
            #             F.cross_entropy(logits_per_image, labels) +
            #             F.cross_entropy(logits_per_text, labels)
            #         ) / 2

            #         gen_loss = maybe_compute_generative_loss(model_out)

            #     cumulative_loss += total_loss * batch_size
            #     num_samples += batch_size
            #     if is_master(args) and (i % 100) == 0:
            #         logging.info(
            #             f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
            #             f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

            #         if gen_loss is not None:
            #             cumulative_gen_loss += gen_loss * batch_size
            #             logging.info(
            #                 f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")
            # all_text_features = torch.cat(all_text_features).to(device=device, non_blocking=True)

            # all_images = torch.cat(all_images, dim=0)
            # all_texts = torch.cat(all_texts, dim=0)

            import pickle
            topk = 1
            num_total_img = 1000
            org_clip_features = pickle.load(open('DATASET_PATH/clip_capsbench_pg_captioner_infer_img_txt_features_5.11.pkl', 'rb'))
            org_image_features = org_clip_features['image_features'][:num_total_img].cuda()
            org_text_features = org_clip_features['text_features'].cuda()
            logits_per_image = (org_image_features @ org_text_features.t()).detach().cpu()
            logits_per_text = logits_per_image.t().detach().cpu()
            # logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
            ranking = torch.argsort(logits_per_text, descending=True)
            top1_recall = 0
            top5_recall = 0
            top10_recall = 0
            recall_cnt = 0
            recall_topk_curve = {}
            name = 'text_to_image'
            for topi in range(1, 100):
                recall_topk_curve[f"{name}_R@{topi}"] = 0


            for text_i in range(ranking.shape[0]):
                print('text_i: ', text_i)
                topk_img_indices = ranking[text_i][:topk]
                # image_i_2_tg_img_feat = {}
                logit_imgi_list = []

                # if txti_2_imgi_json[str(text_i)] in topk_img_indices:
                #     recall_topk_initial += 1
                #     print('initial recalled within topk')
                # recall_cnt += 1
                # continue

                # # for original CLIP
                # image_iter_i = 0
                # for image_i in topk_img_indices.detach().cpu().tolist():
                #     logit_imgi_list.append([(org_text_features[text_i] * org_image_features[image_i]).sum().detach().cpu().tolist(), image_i])
                #     image_iter_i += 1

                # new implementation - faster 10.24
                forward_inference_img_list = []
                for image_i in topk_img_indices.detach().cpu().tolist():
                    forward_inference_img_list.append(dataloader.dataset[image_i][0].unsqueeze(0))
                forward_inference_img_list = torch.cat(forward_inference_img_list, dim=0).cuda()
                tg_img_feat = model.encode_image(forward_inference_img_list.cuda(), text_embed=org_text_features[text_i].unsqueeze(0).repeat(topk, 1), normalize=True)
                image_iter_i = 0
                for image_i in topk_img_indices.detach().cpu().tolist():
                    logit_imgi_list.append([(org_text_features[text_i] * tg_img_feat[image_iter_i]).sum().detach().cpu().tolist(), image_i])
                    image_iter_i += 1

                # old implementation
                # for image_i in topk_img_indices.detach().cpu().tolist():
                #     # ipdb.set_trace()
                #     tg_img_feat = model.encode_image(dataloader.dataset[image_i][0].unsqueeze(0).cuda(), text_embed=org_text_features[text_i].unsqueeze(0), normalize=True)
                #     tg_img_feat = tg_img_feat.squeeze()
                #     # image_i_2_tg_img_feat[image_i] = tg_img_feat
                #     logit_imgi_list.append([(org_text_features[text_i] * tg_img_feat).sum().detach().cpu().tolist(), image_i])
                
                logit_imgi_list.sort(key=lambda x: -x[0])
                # ipdb.set_trace()
                logit_list = np.array(logit_imgi_list)[:,1].tolist()
                if text_i in logit_list[:1]:
                    print('top1')
                    top1_recall += 1
                if text_i in logit_list[:5]:
                    print('top5')
                    top5_recall += 1
                if text_i in logit_list[:10]:
                    print('top10')
                    top10_recall += 1
                recall_cnt += 1

                # # for vis
                # if text_i < 5000:
                #     if txti_2_imgi_json[str(text_i)] in logit_list[:1] and txti_2_imgi_json[str(text_i)] not in topk_img_indices[:2]: 
                #         # ipdb.set_trace()
                #         print("get a example for flickr")
                #         for vis_img_id in range(10):
                #             if '/' not in raw_captions[int(logit_list[vis_img_id])] and '/' not in raw_captions[int(topk_img_indices[vis_img_id].item())] and len(raw_captions[int(logit_list[vis_img_id])]) < 100 and len(raw_captions[int(topk_img_indices[vis_img_id].item())]) < 100:
                #                 shutil.copy(raw_images[int(logit_list[vis_img_id])], os.path.join(vis_dir, 'txti_' + str(text_i) + '-after-top_' + str(vis_img_id) + '-csvid_' + str(int(logit_list[vis_img_id])) + '-caption_' + raw_captions[int(logit_list[vis_img_id])] + '.png'))
                #                 shutil.copy(raw_images[int(topk_img_indices[vis_img_id].item())], os.path.join(vis_dir, 'txti_' + str(text_i) + '-before-top_' + str(vis_img_id) + '-csvid_' + str(int(topk_img_indices[vis_img_id].item())) + '-caption_' + raw_captions[int(topk_img_indices[vis_img_id].item())] + '.png'))

                    # ipdb.set_trace()

                for topi in range(1, 100):
                    recall_topk_curve[f"{name}_R@{topi}"] += text_i in logit_list[:topi]
                    
            # print('initial recall@topk: ', recall_topk_initial / recall_cnt)

            print('top1_recall: ', top1_recall / recall_cnt)
            print('top5_recall: ', top5_recall / recall_cnt)
            print('top10_recall: ', top10_recall / recall_cnt)

            for topi in range(1, 100):
                recall_topk_curve[f"{name}_R@{topi}"] =  recall_topk_curve[f"{name}_R@{topi}"] / recall_cnt
           
            ipdb.set_trace()
            with open('PROJECT_PATH/recall_k_standard_flickr_org_clip_11.14.json', 'w') as dump_f:
                json.dump(recall_topk_curve, dump_f)
                
    return 0




def tgvpt_evaluate_reranking_standard_flickr_transformer(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    print('start evaluation: tgvpt_evaluate_reranking')

    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    # # vis 11.8
    # import shutil
    # import pandas as pd
    # df = pd.read_csv(csv_path, sep="\t")
    # raw_images = df['filepath'].tolist()
    # raw_captions = df['title'].tolist()

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():

            all_images, all_texts = [], []
            for i, batch in enumerate(dataloader):
                with autocast():
                    images, texts = batch
                    images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                    texts = texts.to(device=device, non_blocking=True)
                    all_images.append(images)
                    all_texts.append(texts)

                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                #     logits_per_image = logit_scale * image_features @ text_features.t()
                #     logits_per_text = logits_per_image.t()

                #     batch_size = images.shape[0]
                #     labels = torch.arange(batch_size, device=device).long()
                #     total_loss = (
                #         F.cross_entropy(logits_per_image, labels) +
                #         F.cross_entropy(logits_per_text, labels)
                #     ) / 2

                #     gen_loss = maybe_compute_generative_loss(model_out)

                # cumulative_loss += total_loss * batch_size
                # num_samples += batch_size
                # if is_master(args) and (i % 100) == 0:
                #     logging.info(
                #         f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                #         f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                #     if gen_loss is not None:
                #         cumulative_gen_loss += gen_loss * batch_size
                #         logging.info(
                #             f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")
            all_text_features = torch.cat(all_text_features).to(device=device, non_blocking=True)

            all_images = torch.cat(all_images, dim=0)
            all_texts = torch.cat(all_texts, dim=0)

            import pickle
            txti_2_imgi_json = json.load(open('DATASET_PATH/flickr_test_txtid2imgid.json'))
            topk = 100
            num_total_img = 1000
            org_clip_features = pickle.load(open('DATASET_PATH/clip_vitb16_standard_flickr_infer_img_txt_features_10.24.pkl', 'rb'))
            org_image_features = org_clip_features['image_features'][:num_total_img].cuda()
            org_text_features = org_clip_features['text_features'].cuda()
            logits_per_image = (org_image_features @ org_text_features.t()).detach().cpu()
            logits_per_text = logits_per_image.t().detach().cpu()
            # logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
            ranking = torch.argsort(logits_per_text, descending=True)
            top1_recall = 0
            top5_recall = 0
            top10_recall = 0
            recall_cnt = 0
            recall_topk_curve = {}
            name = 'text_to_image'
            for topi in range(1, 100):
                recall_topk_curve[f"{name}_R@{topi}"] = 0


            for text_i in range(ranking.shape[0]):
                print('text_i: ', text_i)
                topk_img_indices = ranking[text_i][:topk]
                # image_i_2_tg_img_feat = {}
                logit_imgi_list = []

                # if txti_2_imgi_json[str(text_i)] in topk_img_indices:
                #     recall_topk_initial += 1
                #     print('initial recalled within topk')
                # recall_cnt += 1
                # continue

                # # for original CLIP
                # image_iter_i = 0
                # for image_i in topk_img_indices.detach().cpu().tolist():
                #     logit_imgi_list.append([(org_text_features[text_i] * org_image_features[image_i]).sum().detach().cpu().tolist(), image_i])
                #     image_iter_i += 1

                # new implementation - faster 10.24
                forward_inference_img_list = []
                for image_i in topk_img_indices.detach().cpu().tolist():
                    forward_inference_img_list.append(dataloader.dataset[image_i][0].unsqueeze(0))
                forward_inference_img_list = torch.cat(forward_inference_img_list, dim=0).cuda()
                tg_img_feat = model.encode_image(forward_inference_img_list.cuda(), text_embed=all_text_features[text_i].unsqueeze(0).repeat(topk, 1), normalize=True)
                image_iter_i = 0
                for image_i in topk_img_indices.detach().cpu().tolist():
                    logit_imgi_list.append([(org_text_features[text_i] * tg_img_feat[image_iter_i]).sum().detach().cpu().tolist(), image_i])
                    image_iter_i += 1

                # old implementation
                # for image_i in topk_img_indices.detach().cpu().tolist():
                #     # ipdb.set_trace()
                #     tg_img_feat = model.encode_image(dataloader.dataset[image_i][0].unsqueeze(0).cuda(), text_embed=org_text_features[text_i].unsqueeze(0), normalize=True)
                #     tg_img_feat = tg_img_feat.squeeze()
                #     # image_i_2_tg_img_feat[image_i] = tg_img_feat
                #     logit_imgi_list.append([(org_text_features[text_i] * tg_img_feat).sum().detach().cpu().tolist(), image_i])
                
                logit_imgi_list.sort(key=lambda x: -x[0])
                # ipdb.set_trace()
                logit_list = np.array(logit_imgi_list)[:,1].tolist()
                if txti_2_imgi_json[str(text_i)] in logit_list[:1]:
                    print('top1')
                    top1_recall += 1
                if txti_2_imgi_json[str(text_i)] in logit_list[:5]:
                    print('top5')
                    top5_recall += 1
                if txti_2_imgi_json[str(text_i)] in logit_list[:10]:
                    print('top10')
                    top10_recall += 1
                recall_cnt += 1

                # # for vis
                # if text_i < 5000:
                #     if txti_2_imgi_json[str(text_i)] in logit_list[:1] and txti_2_imgi_json[str(text_i)] not in topk_img_indices[:2]: 
                #         # ipdb.set_trace()
                #         print("get a example for flickr")
                #         for vis_img_id in range(10):
                #             if '/' not in raw_captions[int(logit_list[vis_img_id])] and '/' not in raw_captions[int(topk_img_indices[vis_img_id].item())] and len(raw_captions[int(logit_list[vis_img_id])]) < 100 and len(raw_captions[int(topk_img_indices[vis_img_id].item())]) < 100:
                #                 shutil.copy(raw_images[int(logit_list[vis_img_id])], os.path.join(vis_dir, 'txti_' + str(text_i) + '-after-top_' + str(vis_img_id) + '-csvid_' + str(int(logit_list[vis_img_id])) + '-caption_' + raw_captions[int(logit_list[vis_img_id])] + '.png'))
                #                 shutil.copy(raw_images[int(topk_img_indices[vis_img_id].item())], os.path.join(vis_dir, 'txti_' + str(text_i) + '-before-top_' + str(vis_img_id) + '-csvid_' + str(int(topk_img_indices[vis_img_id].item())) + '-caption_' + raw_captions[int(topk_img_indices[vis_img_id].item())] + '.png'))

                    # ipdb.set_trace()

                for topi in range(1, 100):
                    recall_topk_curve[f"{name}_R@{topi}"] += txti_2_imgi_json[str(text_i)] in logit_list[:topi]
                    
            # print('initial recall@topk: ', recall_topk_initial / recall_cnt)

            print('top1_recall: ', top1_recall / recall_cnt)
            print('top5_recall: ', top5_recall / recall_cnt)
            print('top10_recall: ', top10_recall / recall_cnt)

            for topi in range(1, 100):
                 recall_topk_curve[f"{name}_R@{topi}"] =  recall_topk_curve[f"{name}_R@{topi}"] / recall_cnt
           
            ipdb.set_trace()
            with open('PROJECT_PATH/recall_k_standard_flickr_org_clip_11.14.json', 'w') as dump_f:
                json.dump(recall_topk_curve, dump_f)
                
    return 0



def tgvpt_evaluate_reranking_standard_flickr_siglip(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    print('start evaluation: tgvpt_evaluate_reranking')

    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    # # vis 11.8
    # import shutil
    # import pandas as pd
    # df = pd.read_csv(csv_path, sep="\t")
    # raw_images = df['filepath'].tolist()
    # raw_captions = df['title'].tolist()

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():

            # all_images, all_texts = [], []
            # for i, batch in enumerate(dataloader):
            #     with autocast():
            #         images, texts = batch
            #         images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            #         texts = texts.to(device=device, non_blocking=True)
            #         all_images.append(images)
            #         all_texts.append(texts)

            #         model_out = model(images, texts)
            #         image_features = model_out["image_features"]
            #         text_features = model_out["text_features"]
            #         logit_scale = model_out["logit_scale"]
            #         # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
            #         # however, system RAM is easily exceeded and compute time becomes problematic
            #         all_text_features.append(text_features.cpu())
            #         logit_scale = logit_scale.mean()
            #         logits_per_image = logit_scale * image_features @ text_features.t()
            #         logits_per_text = logits_per_image.t()

            #         batch_size = images.shape[0]
            #         labels = torch.arange(batch_size, device=device).long()
            #         total_loss = (
            #             F.cross_entropy(logits_per_image, labels) +
            #             F.cross_entropy(logits_per_text, labels)
            #         ) / 2

            #         gen_loss = maybe_compute_generative_loss(model_out)

            #     cumulative_loss += total_loss * batch_size
            #     num_samples += batch_size
            #     if is_master(args) and (i % 100) == 0:
            #         logging.info(
            #             f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
            #             f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

            #         if gen_loss is not None:
            #             cumulative_gen_loss += gen_loss * batch_size
            #             logging.info(
            #                 f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")
            # all_text_features = torch.cat(all_text_features).to(device=device, non_blocking=True)

            # all_images = torch.cat(all_images, dim=0)
            # all_texts = torch.cat(all_texts, dim=0)

            import pickle
            txti_2_imgi_json = json.load(open('standard_benchmarks/flickr/flickr_test_txtid2imgid.json'))
            # topk = 100
            topk = 50
            num_total_img = 1000
            # org_clip_features = pickle.load(open('standard_benchmarks/flickr/siglipSO400M_standard_flickr_infer_img_txt_features_2.26.pkl', 'rb'))
            org_clip_features = pickle.load(open('standard_benchmarks/flickr/siglip2G_standard_flickr_infer_img_txt_features_2.26.pkl', 'rb'))
            org_image_features = org_clip_features['image_features'][:num_total_img].cuda()
            org_text_features = org_clip_features['text_features'].cuda()
            logits_per_image = (org_image_features @ org_text_features.t()).detach().cpu()
            logits_per_text = logits_per_image.t().detach().cpu()
            # logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
            ranking = torch.argsort(logits_per_text, descending=True)
            top1_recall = 0
            top5_recall = 0
            top10_recall = 0
            recall_cnt = 0
            recall_topk_curve = {}
            name = 'text_to_image'
            for topi in range(1, 100):
            # for topi in range(1, 501):
                recall_topk_curve[f"{name}_R@{topi}"] = 0


            for text_i in range(ranking.shape[0]):
                print('text_i: ', text_i)
                topk_img_indices = ranking[text_i][:topk]
                # image_i_2_tg_img_feat = {}
                logit_imgi_list = []

                # if txti_2_imgi_json[str(text_i)] in topk_img_indices:
                #     recall_topk_initial += 1
                #     print('initial recalled within topk')
                # recall_cnt += 1
                # continue

                # # for original CLIP
                # image_iter_i = 0
                # for image_i in topk_img_indices.detach().cpu().tolist():
                #     logit_imgi_list.append([(org_text_features[text_i] * org_image_features[image_i]).sum().detach().cpu().tolist(), image_i])
                #     image_iter_i += 1

                # new implementation - faster 10.24
                forward_inference_img_list = []
                for image_i in topk_img_indices.detach().cpu().tolist():
                    forward_inference_img_list.append(dataloader.dataset[image_i][0].unsqueeze(0))
                forward_inference_img_list = torch.cat(forward_inference_img_list, dim=0).cuda()
                tg_img_feat = model.encode_image(forward_inference_img_list.cuda(), text_feat=org_text_features[text_i].unsqueeze(0).repeat(topk, 1), normalize=True)
                image_iter_i = 0
                for image_i in topk_img_indices.detach().cpu().tolist():
                    logit_imgi_list.append([(org_text_features[text_i] * tg_img_feat[image_iter_i]).sum().detach().cpu().tolist(), image_i])
                    image_iter_i += 1

                # old implementation
                # for image_i in topk_img_indices.detach().cpu().tolist():
                #     # ipdb.set_trace()
                #     tg_img_feat = model.encode_image(dataloader.dataset[image_i][0].unsqueeze(0).cuda(), text_embed=org_text_features[text_i].unsqueeze(0), normalize=True)
                #     tg_img_feat = tg_img_feat.squeeze()
                #     # image_i_2_tg_img_feat[image_i] = tg_img_feat
                #     logit_imgi_list.append([(org_text_features[text_i] * tg_img_feat).sum().detach().cpu().tolist(), image_i])
                
                logit_imgi_list.sort(key=lambda x: -x[0])
                # ipdb.set_trace()
                logit_list = np.array(logit_imgi_list)[:,1].tolist()
                if txti_2_imgi_json[str(text_i)] in logit_list[:1]:
                    print('top1')
                    top1_recall += 1
                if txti_2_imgi_json[str(text_i)] in logit_list[:5]:
                    print('top5')
                    top5_recall += 1
                if txti_2_imgi_json[str(text_i)] in logit_list[:10]:
                    print('top10')
                    top10_recall += 1
                recall_cnt += 1

                # # for vis
                # if text_i < 5000:
                #     if txti_2_imgi_json[str(text_i)] in logit_list[:1] and txti_2_imgi_json[str(text_i)] not in topk_img_indices[:2]: 
                #         # ipdb.set_trace()
                #         print("get a example for flickr")
                #         for vis_img_id in range(10):
                #             if '/' not in raw_captions[int(logit_list[vis_img_id])] and '/' not in raw_captions[int(topk_img_indices[vis_img_id].item())] and len(raw_captions[int(logit_list[vis_img_id])]) < 100 and len(raw_captions[int(topk_img_indices[vis_img_id].item())]) < 100:
                #                 shutil.copy(raw_images[int(logit_list[vis_img_id])], os.path.join(vis_dir, 'txti_' + str(text_i) + '-after-top_' + str(vis_img_id) + '-csvid_' + str(int(logit_list[vis_img_id])) + '-caption_' + raw_captions[int(logit_list[vis_img_id])] + '.png'))
                #                 shutil.copy(raw_images[int(topk_img_indices[vis_img_id].item())], os.path.join(vis_dir, 'txti_' + str(text_i) + '-before-top_' + str(vis_img_id) + '-csvid_' + str(int(topk_img_indices[vis_img_id].item())) + '-caption_' + raw_captions[int(topk_img_indices[vis_img_id].item())] + '.png'))

                    # ipdb.set_trace()

                # for topi in range(1, 501):
                for topi in range(1, 100):
                    recall_topk_curve[f"{name}_R@{topi}"] += txti_2_imgi_json[str(text_i)] in logit_list[:topi]
                    
            # print('initial recall@topk: ', recall_topk_initial / recall_cnt)

            print('top1_recall: ', top1_recall / recall_cnt)
            print('top5_recall: ', top5_recall / recall_cnt)
            print('top10_recall: ', top10_recall / recall_cnt)

            for topi in range(1, 100):
            # for topi in range(1, 501):
                 recall_topk_curve[f"{name}_R@{topi}"] =  recall_topk_curve[f"{name}_R@{topi}"] / recall_cnt
           
            ipdb.set_trace()
            with open('PROJECT_PATH/recall_k_standard_flickr_siglip2G_0prompt_3.2.json', 'w') as dump_f:
                json.dump(recall_topk_curve, dump_f)
                
    return 0



def tgvpt_evaluate_reranking_sd(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    print('start evaluation: tgvpt_evaluate_reranking')

    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            all_images, all_texts, all_images_sd = [], [], []
            all_images_sd1, all_images_sd2, all_images_sd3, all_images_sd4 = [], [], [], []
            for i, batch in enumerate(dataloader):
                print(i)
                with autocast():
                    images, texts, images_sd, images_sd1, images_sd2, images_sd3, images_sd4 = batch
                    images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                    images_sd = images_sd.to(device=device, dtype=input_dtype, non_blocking=True)
                    texts = texts.to(device=device, non_blocking=True)
                    all_images.append(images)
                    all_images_sd.append(images_sd)
                    all_texts.append(texts)

                    all_images_sd1.append(images_sd1)
                    all_images_sd2.append(images_sd2)
                    all_images_sd3.append(images_sd3)
                    all_images_sd4.append(images_sd4)

                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")
            all_text_features = torch.cat(all_text_features).to(device=device, non_blocking=True)

            all_images = torch.cat(all_images, dim=0)
            all_images_sd = torch.cat(all_images_sd, dim=0)
            all_texts = torch.cat(all_texts, dim=0)

            all_images_sd1 = torch.cat(all_images_sd1, dim=0)
            all_images_sd2 = torch.cat(all_images_sd2, dim=0)
            all_images_sd3 = torch.cat(all_images_sd3, dim=0)
            all_images_sd4 = torch.cat(all_images_sd4, dim=0)

            import pickle
            topk = 100
            org_clip_features = pickle.load(open('DATASET_PATH/clip_vitb16_coco_infer_img_txt_features_9.4.pkl', 'rb'))
            org_image_features = org_clip_features['image_features']
            org_text_features = org_clip_features['text_features']
            logits_per_image = (org_image_features @ org_text_features.t()).detach().cpu()
            logits_per_text = logits_per_image.t().detach().cpu()
            # logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
            ranking = torch.argsort(logits_per_text, descending=True)
            top1_recall = 0
            top5_recall = 0
            top10_recall = 0
            recall_cnt = 0
            recall_topk_curve = {}

            all_text_features_sd = model.encode_image(all_images_sd, normalize=True)
            all_text_features_sd1 = model.encode_image(all_images_sd1, normalize=True)
            all_text_features_sd2 = model.encode_image(all_images_sd2, normalize=True)
            all_text_features_sd3 = model.encode_image(all_images_sd3, normalize=True)
            all_text_features_sd4 = model.encode_image(all_images_sd4, normalize=True)
            all_text_features = all_text_features + (all_text_features_sd + all_text_features_sd1 + all_text_features_sd2 + all_text_features_sd3 + all_text_features_sd4) * 0.2

            name = 'text_to_image'
            for topi in range(1, 100):
                recall_topk_curve[f"{name}_R@{topi}"] = 0
            for text_i in range(ranking.shape[0]):
                print('text_i: ', text_i)
                topk_img_indices = ranking[text_i][:topk]
                # image_i_2_tg_img_feat = {}
                logit_imgi_list = []
                for image_i in topk_img_indices.detach().cpu().tolist():
                    # ipdb.set_trace()
                    
                    # ipdb.set_trace()
                    tg_img_feat = model.encode_image(all_images[image_i].unsqueeze(0), text_embed=all_text_features[text_i].unsqueeze(0), normalize=True)
                    
                    tg_img_feat = tg_img_feat.squeeze()
                    # image_i_2_tg_img_feat[image_i] = tg_img_feat
                    logit_imgi_list.append([(all_text_features[text_i] * tg_img_feat).sum().detach().cpu().tolist(), image_i])
                logit_imgi_list.sort(key=lambda x: -x[0])
                # ipdb.set_trace()
                logit_list = np.array(logit_imgi_list)[:,1].tolist()
                if text_i in logit_list[:1]:
                    print('top1')
                    top1_recall += 1
                if text_i in logit_list[:5]:
                    print('top5')
                    top5_recall += 1
                if text_i in logit_list[:10]:
                    print('top10')
                    top10_recall += 1
                recall_cnt += 1

                for topi in range(1, 100):
                    recall_topk_curve[f"{name}_R@{topi}"] += text_i in logit_list[:topi]
                    
            print('top1_recall: ', top1_recall / recall_cnt)
            print('top5_recall: ', top5_recall / recall_cnt)
            print('top10_recall: ', top10_recall / recall_cnt)

            for topi in range(1, 100):
                 recall_topk_curve[f"{name}_R@{topi}"] =  recall_topk_curve[f"{name}_R@{topi}"] / recall_cnt
           
            ipdb.set_trace()
            with open('PROJECT_PATH/recall_k_coco_data_after_reranking_9.17_harder_bs56_ep2.json', 'w') as dump_f:
                json.dump(recall_topk_curve, dump_f)

    return 0



def tgvpt_evaluate_reranking_sd1(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    print('start evaluation: tgvpt_evaluate_reranking')

    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        # all_image_features, all_text_features = [], []
        with torch.no_grad():
            all_images, all_texts, all_images_sd = [], [], []
            # all_images_sd1, all_images_sd2, all_images_sd3, all_images_sd4 = [], [], [], []
            for i, batch in enumerate(dataloader):
                print(i)
                with autocast():
                    images, texts, images_sd, images_sd1, images_sd2, images_sd3, images_sd4 = batch
                    images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                    images_sd = images_sd.to(device=device, dtype=input_dtype, non_blocking=True)
                    texts = texts.to(device=device, non_blocking=True)
                    all_images.append(images)
                    all_images_sd.append(images_sd)
                    all_texts.append(texts)

                    # all_images_sd1.append(images_sd1)
                    # all_images_sd2.append(images_sd2)
                    # all_images_sd3.append(images_sd3)
                    # all_images_sd4.append(images_sd4)

                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    # all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")
            # all_text_features = torch.cat(all_text_features).to(device=device, non_blocking=True)

            all_images = torch.cat(all_images, dim=0)
            all_images_sd = torch.cat(all_images_sd, dim=0)
            all_texts = torch.cat(all_texts, dim=0)

            # all_images_sd1 = torch.cat(all_images_sd1, dim=0)
            # all_images_sd2 = torch.cat(all_images_sd2, dim=0)
            # all_images_sd3 = torch.cat(all_images_sd3, dim=0)
            # all_images_sd4 = torch.cat(all_images_sd4, dim=0)

            import pickle
            topk = 100
            org_clip_features = pickle.load(open('DATASET_PATH/clip_vitb16_coco_infer_img_txt_features_9.4.pkl', 'rb'))
            org_image_features = org_clip_features['image_features']
            org_text_features = org_clip_features['text_features']
            logits_per_image = (org_image_features @ org_text_features.t()).detach().cpu()
            logits_per_text = logits_per_image.t().detach().cpu()
            # logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
            ranking = torch.argsort(logits_per_text, descending=True)
            top1_recall = 0
            top5_recall = 0
            top10_recall = 0
            recall_cnt = 0
            recall_topk_curve = {}

            all_text_features = model.encode_image(all_images_sd, normalize=True)
            # all_text_features_sd1 = model.encode_image(all_images_sd1, normalize=True)
            # all_text_features_sd2 = model.encode_image(all_images_sd2, normalize=True)
            # all_text_features_sd3 = model.encode_image(all_images_sd3, normalize=True)
            # all_text_features_sd4 = model.encode_image(all_images_sd4, normalize=True)
            # all_text_features = all_text_features + all_text_features_sd

            name = 'text_to_image'
            for topi in range(1, 100):
                recall_topk_curve[f"{name}_R@{topi}"] = 0
            for text_i in range(ranking.shape[0]):
                print('text_i: ', text_i)
                topk_img_indices = ranking[text_i][:topk]
                # image_i_2_tg_img_feat = {}
                logit_imgi_list = []
                for image_i in topk_img_indices.detach().cpu().tolist():
                    # ipdb.set_trace()
                    
                    # ipdb.set_trace()
                    tg_img_feat = model.encode_image(all_images[image_i].unsqueeze(0), text_embed=all_text_features[text_i].unsqueeze(0), normalize=True)
                    
                    tg_img_feat = tg_img_feat.squeeze()
                    # image_i_2_tg_img_feat[image_i] = tg_img_feat
                    logit_imgi_list.append([(all_text_features[text_i] * tg_img_feat).sum().detach().cpu().tolist(), image_i])
                logit_imgi_list.sort(key=lambda x: -x[0])
                # ipdb.set_trace()
                logit_list = np.array(logit_imgi_list)[:,1].tolist()
                if text_i in logit_list[:1]:
                    print('top1')
                    top1_recall += 1
                if text_i in logit_list[:5]:
                    print('top5')
                    top5_recall += 1
                if text_i in logit_list[:10]:
                    print('top10')
                    top10_recall += 1
                recall_cnt += 1

                for topi in range(1, 100):
                    recall_topk_curve[f"{name}_R@{topi}"] += text_i in logit_list[:topi]
                    
            print('top1_recall: ', top1_recall / recall_cnt)
            print('top5_recall: ', top5_recall / recall_cnt)
            print('top10_recall: ', top10_recall / recall_cnt)

            for topi in range(1, 100):
                 recall_topk_curve[f"{name}_R@{topi}"] =  recall_topk_curve[f"{name}_R@{topi}"] / recall_cnt
           
            ipdb.set_trace()
            with open('PROJECT_PATH/recall_k_coco_data_after_reranking_9.17_harder_bs56_ep2.json', 'w') as dump_f:
                json.dump(recall_topk_curve, dump_f)

    return 0



def tgvpt_evaluate_reranking_sd2(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    print('start evaluation: tgvpt_evaluate_reranking')

    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            all_images, all_texts, all_images_sd = [], [], []
            # all_images_sd1, all_images_sd2, all_images_sd3, all_images_sd4 = [], [], [], []
            for i, batch in enumerate(dataloader):
                print(i)
                with autocast():
                    images, texts, images_sd, images_sd1, images_sd2, images_sd3, images_sd4 = batch
                    images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                    images_sd = images_sd.to(device=device, dtype=input_dtype, non_blocking=True)
                    texts = texts.to(device=device, non_blocking=True)
                    all_images.append(images)
                    all_images_sd.append(images_sd)
                    all_texts.append(texts)

                    # all_images_sd1.append(images_sd1)
                    # all_images_sd2.append(images_sd2)
                    # all_images_sd3.append(images_sd3)
                    # all_images_sd4.append(images_sd4)

                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")
            all_text_features = torch.cat(all_text_features).to(device=device, non_blocking=True)

            all_images = torch.cat(all_images, dim=0)
            all_images_sd = torch.cat(all_images_sd, dim=0)
            all_texts = torch.cat(all_texts, dim=0)

            # all_images_sd1 = torch.cat(all_images_sd1, dim=0)
            # all_images_sd2 = torch.cat(all_images_sd2, dim=0)
            # all_images_sd3 = torch.cat(all_images_sd3, dim=0)
            # all_images_sd4 = torch.cat(all_images_sd4, dim=0)

            import pickle
            topk = 100
            org_clip_features = pickle.load(open('DATASET_PATH/clip_vitb16_coco_infer_img_txt_features_9.4.pkl', 'rb'))
            org_image_features = org_clip_features['image_features']
            org_text_features = org_clip_features['text_features']
            logits_per_image = (org_image_features @ org_text_features.t()).detach().cpu()
            logits_per_text = logits_per_image.t().detach().cpu()
            # logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
            ranking = torch.argsort(logits_per_text, descending=True)
            top1_recall = 0
            top5_recall = 0
            top10_recall = 0
            recall_cnt = 0
            recall_topk_curve = {}

            all_text_features_sd = model.encode_image(all_images_sd, normalize=True)
            # all_text_features_sd1 = model.encode_image(all_images_sd1, normalize=True)
            # all_text_features_sd2 = model.encode_image(all_images_sd2, normalize=True)
            # all_text_features_sd3 = model.encode_image(all_images_sd3, normalize=True)
            # all_text_features_sd4 = model.encode_image(all_images_sd4, normalize=True)
            # all_text_features = all_text_features + all_text_features_sd

            name = 'text_to_image'
            for topi in range(1, 100):
                recall_topk_curve[f"{name}_R@{topi}"] = 0
            for text_i in range(ranking.shape[0]):
                print('text_i: ', text_i)
                topk_img_indices = ranking[text_i][:topk]
                # image_i_2_tg_img_feat = {}
                logit_imgi_list = []
                for image_i in topk_img_indices.detach().cpu().tolist():
                    # ipdb.set_trace()
                    
                    # ipdb.set_trace()
                    tg_img_feat = model.encode_image(all_images[image_i].unsqueeze(0), text_embed=all_text_features[text_i].unsqueeze(0), normalize=True)
                    
                    tg_img_feat = tg_img_feat.squeeze()
                    # image_i_2_tg_img_feat[image_i] = tg_img_feat
                    logit_imgi_list.append([((all_text_features[text_i] + all_text_features_sd[text_i]) * tg_img_feat).sum().detach().cpu().tolist(), image_i])
                logit_imgi_list.sort(key=lambda x: -x[0])
                # ipdb.set_trace()
                logit_list = np.array(logit_imgi_list)[:,1].tolist()
                if text_i in logit_list[:1]:
                    print('top1')
                    top1_recall += 1
                if text_i in logit_list[:5]:
                    print('top5')
                    top5_recall += 1
                if text_i in logit_list[:10]:
                    print('top10')
                    top10_recall += 1
                recall_cnt += 1

                for topi in range(1, 100):
                    recall_topk_curve[f"{name}_R@{topi}"] += text_i in logit_list[:topi]
                    
            print('top1_recall: ', top1_recall / recall_cnt)
            print('top5_recall: ', top5_recall / recall_cnt)
            print('top10_recall: ', top10_recall / recall_cnt)

            for topi in range(1, 100):
                 recall_topk_curve[f"{name}_R@{topi}"] =  recall_topk_curve[f"{name}_R@{topi}"] / recall_cnt
           
            ipdb.set_trace()
            with open('PROJECT_PATH/recall_k_coco_data_after_reranking_9.17_harder_bs56_ep2.json', 'w') as dump_f:
                json.dump(recall_topk_curve, dump_f)

    return 0




def gather_features(image_features, rank, world_size):
    gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
    dist.all_gather(gathered_image_features, image_features)
    gathered_image_features[rank] = image_features
    all_image_features = torch.cat(gathered_image_features, dim=0)
    return all_image_features


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def train_mp_one_epoch_tgvpt(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    local_rank, rank, world_size = world_info_from_env()

    for i, batch in enumerate(dataloader):
        # if i > 100:
        #     return
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        if args.csv_label_key != "none":
            images, texts, _ = batch
        else:
            images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        # print('type(images.data) ', type(images.data))

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                text_features = model(images, texts)["text_features"]
                # logit_scale = model_out["logit_scale"]
                # if args.distill:
                #     with torch.no_grad():
                #         dist_model_out = dist_model(images, texts)
                #     model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                # losses = loss(**model_out, output_dict=True)
                # text_features = model.module.encode_text(texts, normalize=True)
                all_text_features = gather_features(text_features, rank, world_size)
                # all_text_features = torch.cat(GatherLayer.apply(text_features), dim=0)
                texts = torch.cat(GatherLayer.apply(texts), dim=0)

                logit_scale = model.module.logit_scale.exp()
                losses = {}

                batch_size = images.shape[0]
                all_batch_size = texts.shape[0]
                all_image_features = []
                for j in range(batch_size):
                    image_features_i = []
                    for k in range(all_batch_size):
                        with autocast():
                            # model_out = model(images[j].unsqueeze(0), texts[k].unsqueeze(0))
                            # image_features = model_out["image_features"]
                            # image_features = model.module.encode_image(images[j].unsqueeze(0), text_embed=all_text_features[k].unsqueeze(0), normalize=True)
                            # print('images[j].shape', images[j].shape)
                            # print('texts[k].unsqueeze(0).shape', texts[k].unsqueeze(0).shape)
                            image_features = model(images[j].unsqueeze(0), texts[k].unsqueeze(0))["image_features"]

                            image_features_i.append(image_features)
                    all_image_features.append(torch.concat(image_features_i, dim=0).unsqueeze(0))
                all_image_features = torch.cat(all_image_features, dim=0).to(device=device, non_blocking=True)

                all_image_features = gather_features(all_image_features, rank, world_size)
                # all_image_features = torch.cat(GatherLayer.apply(all_image_features), dim=0)

                # using all bsz x bsz logits
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_image_flat = torch.reshape(logits_per_image, (batch_size, -1))
                labels = torch.arange(batch_size, device=device, dtype=torch.long)
                gt_labels = labels * batch_size + labels
                losses['token_loss'] = F.cross_entropy(logits_per_image_flat, gt_labels)

                # image_features_i = torch.cat(image_features_i).to(device=device, non_blocking=True)
                # all_logits = image_features_i @ all_text_features.T
                # logits_cate = torch.diagonal(all_logits, dim1=-2, dim2=-1)
                # prompt_idx = torch.argmax(logits_cate)
                # all_image_features.append(image_features_i[prompt_idx].unsqueeze(0))

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def train_one_epoch_reid(model, last_layer, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, labels, is_query = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        labels = labels.to(device=device, dtype=input_dtype, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, None)
                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, None)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                # losses = loss(**model_out, output_dict=True)
                feats = model_out["image_features"]
                logits = last_layer(feats)
                # print(logits.shape)
                # print(labels.shape)
                cls_loss = nn.CrossEntropyLoss()(logits/0.1, labels)
                losses = {}
                losses["ce_loss"] = cls_loss
                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()


def train_one_epoch_with_label(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts, labels = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, texts, labels)
                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                losses = loss(**model_out, output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def train_one_epoch_with_mask(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        # images, texts = batch
        images, texts, classes, objects, occluders, occludees = batch

        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        classes = classes.to(device=device, dtype=input_dtype, non_blocking=True)
        objects = objects.to(device=device, dtype=input_dtype, non_blocking=True)
        occluders = occluders.to(device=device, dtype=input_dtype, non_blocking=True)
        occludees = occludees.to(device=device, dtype=input_dtype, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        # print('type(images.data) ', type(images.data))

        if args.accum_freq == 1:
            with autocast():
                model_out = model([images, objects, occluders, occludees], texts)
                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                losses = loss(**model_out, output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()


def train_one_epoch_with_multi_prompts(model, tokenizer, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    cate_num = model.visual.transformer.cate_num

    classnames = IMAGENET_CLASSNAMES
    templates = OPENAI_IMAGENET_TEMPLATES
    if 'coco-val' in data:
        classnames = COCO_CLASSNAMES
        templates = SIMPLE_COCO_TEMPLATES
    classifier = build_zero_shot_classifier(
        model,
        tokenizer=tokenizer,
        classnames=classnames,
        templates=templates,
        num_classes_per_batch=10,
        device=args.device,
        use_tqdm=True,
    )

    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts, labels = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                B = images.shape[0]
                total_loss = 0
                for k in range(0, cate_num):
                    # print('----------------',k,' --------------')
                    cate_label = torch.ones(B).type(torch.long).cuda() * k
                    output = model(image=images, label=cate_label)
                    image_features = output['image_features'] if isinstance(output, dict) else output[0]

                    logits_per_image = image_features @ classifier
                    logits_per_image_cate = logits_per_image[:, k]
                    gt_labels = (labels == k).long()
                    total_loss += F.l1_loss(logits_per_image_cate, gt_labels)
                losses["loss"] = total_loss
                # all_image_features = torch.cat([a.unsqueeze(1) for a in all_image_features], dim=1)
                # if args.distill:
                #     with torch.no_grad():
                #         dist_model_out = dist_model(images, texts)
                #     model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                # losses = loss(**model_out, output_dict=True)
                # total_loss = sum(losses.values())
                # losses["loss"] = total_loss



            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


class SupConLoss(torch.nn.Module):
    def __init__(self, temperature=0.01, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def train_one_epoch_ow(model, tokenizer, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # cate_num = model.visual.transformer.cate_num

    classnames = IMAGENET_CLASSNAMES
    templates = OPENAI_IMAGENET_TEMPLATES
    if 'coco-val' in data:
        classnames = COCO_CLASSNAMES
        templates = SIMPLE_COCO_TEMPLATES
    classifier = build_zero_shot_classifier(
        model,
        tokenizer=tokenizer,
        classnames=classnames,
        templates=templates,
        num_classes_per_batch=10,
        device=args.device,
        use_tqdm=True,
    )

    sup_con_loss = SupConLoss(temperature=0.01)

    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts, labels = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                losses = {}
                B = images.shape[0]
                output = model(image=images, text=texts)
                logit_scale = output["logit_scale"]
                image_features = output['image_features'] if isinstance(output, dict) else output[0]

                if args.res_token:
                    image_features = torch.add(image_features[:,1:,:], image_features[:, 0, :].unsqueeze(1))
                else:
                    image_features = image_features[:,1:,:]
                # print('image_features.shape', image_features.shape)
                if args.token == 'all':
                    # using all 80x80 logits
                    logits_per_image = logit_scale * image_features @ classifier

                    logits_per_image_flat = torch.reshape(logits_per_image, (B,-1))
                    gt_labels = labels*80 + labels
                    losses['loss'] = F.cross_entropy(logits_per_image_flat, gt_labels)
                    total_loss = losses['loss']

                    if args.sup_con:
                        sup_image_features = image_features[:,labels,:]
                        losses['sup_con_loss'] = sup_con_loss(sup_image_features, labels=labels)
                        total_loss += losses['sup_con_loss']
                elif args.token == 'diag':
                    # using the diagonal logits
                    logits_per_image = logit_scale * image_features @ classifier
                    logits = torch.diagonal(logits_per_image, dim1=-2, dim2=-1)
                    total_loss = F.cross_entropy(logits, labels)
                    losses['loss'] = total_loss
                elif args.token == 'first':
                    # using the first token
                    logits = logit_scale * image_features[:,0,:] @ classifier
                    total_loss = F.cross_entropy(logits, labels)
                    losses['loss'] = total_loss
                else:
                    raise NotImplementedError('Not supported token type!')

                # using the first token and using clip loss
                # output['image_features'] = image_features[:,0,:]
                # losses = loss(**output, output_dict=True)
                # total_loss = sum(losses.values())
                # losses["loss"] = total_loss
                # logits = image_features[:,0,:] @ classifier
                # acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
                # logging.info(f'acc1:{acc1}, acc5:{acc5}')

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()


def train_one_epoch_ow2(model, tokenizer, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # cate_num = model.visual.transformer.cate_num

    classnames = IMAGENET_CLASSNAMES
    templates = OPENAI_IMAGENET_TEMPLATES
    if 'coco-val' in data:
        classnames = COCO_CLASSNAMES
        templates = SIMPLE_COCO_TEMPLATES
    if 'val_novel' in data:
        classnames = COCO_CLASSNAMES_KNOWN
        templates = SIMPLE_COCO_TEMPLATES
    classifier = build_zero_shot_classifier(
        model,
        tokenizer=tokenizer,
        classnames=classnames,
        templates=templates,
        num_classes_per_batch=10,
        device=args.device,
        use_tqdm=True,
    )

    sup_con_loss = SupConLoss(temperature=0.01)

    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts, labels = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                losses = {}
                B = images.shape[0]
                output = model(image=images, text=texts)
                logit_scale = output["logit_scale"]
                image_features = output['image_features'] if isinstance(output, dict) else output[0]

                if args.res_token:
                    image_features = torch.add(image_features[:,1:,:], image_features[:, 0, :].unsqueeze(1))
                else:
                    image_features = image_features[:,1:,:]
                # print('image_features.shape', image_features.shape)
                if args.token == 'all':
                    # using all 80x80 logits
                    logits_per_image = logit_scale * image_features @ classifier
                    # print(logits_per_image.shape)
                    logits_per_image_flat = torch.reshape(logits_per_image, (B,-1))
                    gt_labels = labels*len(classnames) + labels
                    losses['loss'] = F.cross_entropy(logits_per_image_flat, gt_labels)
                    total_loss = losses['loss']

                    if args.sup_con:
                        sup_image_features = image_features[:,labels,:]
                        losses['sup_con_loss'] = sup_con_loss(sup_image_features, labels=labels)
                        total_loss += losses['sup_con_loss']
                elif args.token == 'diag':
                    # using the diagonal logits
                    logits_per_image = logit_scale * image_features @ classifier
                    logits = torch.diagonal(logits_per_image, dim1=-2, dim2=-1)
                    total_loss = F.cross_entropy(logits, labels)
                    losses['loss'] = total_loss
                elif args.token == 'first':
                    # using the first token
                    logits = logit_scale * image_features[:,0,:] @ classifier
                    total_loss = F.cross_entropy(logits, labels)
                    losses['loss'] = total_loss
                else:
                    raise NotImplementedError('Not supported token type!')

                # using the first token and using clip loss
                # output['image_features'] = image_features[:,0,:]
                # losses = loss(**output, output_dict=True)
                # total_loss = sum(losses.values())
                # losses["loss"] = total_loss
                # logits = image_features[:,0,:] @ classifier
                # acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
                # logging.info(f'acc1:{acc1}, acc5:{acc5}')

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()


def train_one_epoch_with_category_prompt(k, model, tokenizer, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    logging.info(f'*************************** Train per category prompt for category {k} ***************************')
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    classnames = IMAGENET_CLASSNAMES
    templates = OPENAI_IMAGENET_TEMPLATES
    if 'coco-val' in data:
        classnames = COCO_CLASSNAMES
        templates = SIMPLE_COCO_TEMPLATES
    classifier = build_zero_shot_classifier(
        model,
        tokenizer=tokenizer,
        classnames=classnames,
        templates=templates,
        num_classes_per_batch=10,
        device=args.device,
        use_tqdm=True,
    )

    criterion = nn.BCELoss()

    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts, labels = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]

                # print("image_features.norm", image_features.norm(dim=1))
                # print("classifier.norm", classifier.norm(dim=0))
                # ---------------------- L1 Loss ----------------------
                logit_scale = output["logit_scale"]
                logits_per_image = image_features @ classifier
                logits_per_image_cate = logits_per_image[:, k]
                gt_labels = (labels == k).long()
                total_loss = F.l1_loss((logits_per_image_cate+1)/2, gt_labels)
                # ---------------------- CE loss ----------------------
                # logit_scale = output["logit_scale"]
                # logits_per_image, logits_per_text = loss.get_logits(image_features, classifier.T[k,:], output["logit_scale"])
                # gt_labels = (labels == k).float()
                # total_loss = F.binary_cross_entropy_with_logits(logits_per_text, gt_labels)

                losses = {"loss":0}
                losses["loss"] = total_loss

                # print mean cos sim
                cos_sim = image_features @ classifier
                mean_pos = torch.mean(cos_sim[labels == k, k]).item()
                mean_neg = torch.mean(cos_sim[labels != k, k]).item()
                logging.info(f'mean logits_per_image of cate 1 {mean_pos}')
                logging.info(f'mean logits_per_image of cate 0 {mean_neg}')

                # print("k", k)
                # print("labels", labels)
                # print("gt_labels", gt_labels)

                # all_image_features = torch.cat([a.unsqueeze(1) for a in all_image_features], dim=1)
                # if args.distill:
                #     with torch.no_grad():
                #         dist_model_out = dist_model(images, texts)
                #     model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                # losses = loss(**model_out, output_dict=True)
                # total_loss = sum(losses.values())
                # losses["loss"] = total_loss
            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
                + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    # logging.info('start evaluation')
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        # logging.info('start evaluation')

        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader)):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics



def evaluate_occluded_coco(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    logging.info('start evaluation')
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        logging.info('start evaluation')

        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []


        from torchvision import transforms as pth_transforms
        from PIL import Image
        occluded_coco_retrieval_ann_file = json.load(open('DATASET_PATH/coco2017val_cat_id_2_occ_img_negative_img.json'))
        cat_id_2_occ_img = occluded_coco_retrieval_ann_file['cat_id_2_occ_img']
        cat_id_2_negative_img = occluded_coco_retrieval_ann_file['cat_id_2_negative_img']
        img_id_2_all_ins_id = occluded_coco_retrieval_ann_file['img_id_2_all_ins_id']
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
        img_preprocess = pth_transforms.Compose([
            pth_transforms.Resize([224, 224], pth_transforms.InterpolationMode.BICUBIC),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])


        with torch.no_grad():

            cat_id_2_txt_feat = {}
            iter_txt = 0
            for cat_id in cat_id_2_occ_img.keys():
                print(iter_txt)
                iter_txt += 1
                cur_txt = coco_91[int(cat_id)]
                text = tokenizer([cur_txt]).to(device=device, non_blocking=True)
                text_features = model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                cat_id_2_txt_feat[cat_id] = text_features.detach().cpu().numpy()

            img_id_2_img_feat = {}
            img_folder = '/scratch/shared/beegfs/shared-datasets/COCO/COCO2017/val2017/'
            iter_img = 0
            for img_id in img_id_2_all_ins_id.keys():
                # ipdb.set_trace()
                # print(iter_img)
                iter_img += 1
                cur_image = img_preprocess(Image.open(os.path.join(img_folder, str(img_id).rjust(12,'0')+'.jpg')).convert('RGB')).unsqueeze(0)
                # if img_id == '289343':
                cur_image = cur_image.to(device=device, dtype=input_dtype, non_blocking=True)
                image_features = model.encode_image(cur_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                img_id_2_img_feat[img_id] = image_features.detach().cpu().numpy()

            ipdb.set_trace()
            with open('DATASET_PATH/occluded_coco_clip_feat.pkl', 'wb') as dump_f:
                pickle.dump({'cat_id_2_txt_feat':cat_id_2_txt_feat, 'img_id_2_img_feat':img_id_2_img_feat}, dump_f)
            

            top1_recall = 0
            top5_recall = 0
            top10_recall = 0
            recall_cnt = 0
            for cat_id in cat_id_2_txt_feat.keys():
                if cat_id == '1':
                    continue
                print(coco_91[int(cat_id)])
                cur_txt_feat = cat_id_2_txt_feat[cat_id]
                logit_list = []
                occ_img_id_list = cat_id_2_occ_img[cat_id]
                neg_img_id_list = cat_id_2_negative_img[cat_id]
                for img_id in occ_img_id_list:
                    cur_img_feat = img_id_2_img_feat[str(img_id)]
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0].cpu().numpy().tolist()
                    logit_list.append([cur_logit, 1])
                for img_id in neg_img_id_list:
                    cur_img_feat = img_id_2_img_feat[str(img_id)]
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0].cpu().numpy().tolist()
                    logit_list.append([cur_logit, 0])
                logit_list.sort(key=lambda x: -x[0])
                logit_list = np.array(logit_list)[:,1]
                print(logit_list[:len(occ_img_id_list)])
                if logit_list[:len(occ_img_id_list)].sum() == len(occ_img_id_list):
                    top1_recall += 1
                if logit_list[:5*len(occ_img_id_list)].sum() == len(occ_img_id_list):
                    top5_recall += 1
                if logit_list[:10*len(occ_img_id_list)].sum() == len(occ_img_id_list):
                    top10_recall += 1
                recall_cnt += 1

            print('top1 recall: ', top1_recall / recall_cnt)
            print('top5 recall: ', top5_recall / recall_cnt)
            print('top10 recall: ', top10_recall / recall_cnt)
            print('recall cnt: ', recall_cnt)
    return 0



def evaluate_imagenet_a(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    logging.info('start evaluation')
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        logging.info('start evaluation')

        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []


        from torchvision import transforms as pth_transforms
        from PIL import Image
        occluded_coco_retrieval_ann_file = json.load(open('DATASET_PATH/imagenet-a-annfile.json'))
        cat_id_2_occ_img = occluded_coco_retrieval_ann_file['cat_id_2_occ_img']
        cat_id_2_negative_img = occluded_coco_retrieval_ann_file['cat_id_2_negative_img']
        img_id_2_all_ins_id = occluded_coco_retrieval_ann_file['img_id_2_all_ins_id']
        coco_91 = json.load(open('DATASET_PATH/imagenet-a-cat2name.json'))
        img_preprocess = pth_transforms.Compose([
            pth_transforms.Resize([224, 224], pth_transforms.InterpolationMode.BICUBIC),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])


        with torch.no_grad():

            cat_id_2_txt_feat = {}
            iter_txt = 0
            for cat_id in cat_id_2_occ_img.keys():
                print(iter_txt)
                iter_txt += 1
                cur_txt = coco_91[cat_id]
                text = tokenizer([cur_txt]).to(device=device, non_blocking=True)
                text_features = model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                cat_id_2_txt_feat[cat_id] = text_features.detach().cpu().numpy()

            img_id_2_img_feat = {}
            img_folder = 'DATASET_PATH/imagenet-a'
            iter_img = 0
            for img_id in img_id_2_all_ins_id:
                # ipdb.set_trace()
                print(iter_img)
                iter_img += 1
                cur_image = img_preprocess(Image.open(os.path.join(img_folder, img_id)).convert('RGB')).unsqueeze(0)
                # if img_id == '289343':
                cur_image = cur_image.to(device=device, dtype=input_dtype, non_blocking=True)
                image_features = model.encode_image(cur_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                img_id_2_img_feat[img_id] = image_features.detach().cpu().numpy()

            ipdb.set_trace()
            with open('DATASET_PATH/imagenet-a_clip_feat.pkl', 'wb') as dump_f:
                pickle.dump({'cat_id_2_txt_feat':cat_id_2_txt_feat, 'img_id_2_img_feat':img_id_2_img_feat}, dump_f)
            

            top1_recall = 0
            top5_recall = 0
            top10_recall = 0
            recall_cnt = 0
            for cat_id in cat_id_2_txt_feat.keys():
                if cat_id == '1':
                    continue
                print(coco_91[int(cat_id)])
                cur_txt_feat = cat_id_2_txt_feat[cat_id]
                logit_list = []
                occ_img_id_list = cat_id_2_occ_img[cat_id]
                neg_img_id_list = cat_id_2_negative_img[cat_id]
                for img_id in occ_img_id_list:
                    cur_img_feat = img_id_2_img_feat[str(img_id)]
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0].cpu().numpy().tolist()
                    logit_list.append([cur_logit, 1])
                for img_id in neg_img_id_list:
                    cur_img_feat = img_id_2_img_feat[str(img_id)]
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0].cpu().numpy().tolist()
                    logit_list.append([cur_logit, 0])
                logit_list.sort(key=lambda x: -x[0])
                logit_list = np.array(logit_list)[:,1]
                print(logit_list[:len(occ_img_id_list)])
                if logit_list[:len(occ_img_id_list)].sum() == len(occ_img_id_list):
                    top1_recall += 1
                if logit_list[:5*len(occ_img_id_list)].sum() == len(occ_img_id_list):
                    top5_recall += 1
                if logit_list[:10*len(occ_img_id_list)].sum() == len(occ_img_id_list):
                    top10_recall += 1
                recall_cnt += 1

            print('top1 recall: ', top1_recall / recall_cnt)
            print('top5 recall: ', top5_recall / recall_cnt)
            print('top10 recall: ', top10_recall / recall_cnt)
            print('recall cnt: ', recall_cnt)
    return 0


def evaluate_imagenet_r(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    logging.info('start evaluation')
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        logging.info('start evaluation')

        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []


        from torchvision import transforms as pth_transforms
        from PIL import Image
        occluded_coco_retrieval_ann_file = json.load(open('DATASET_PATH/imagenet-a-annfile.json'))
        cat_id_2_occ_img = occluded_coco_retrieval_ann_file['cat_id_2_occ_img']
        cat_id_2_negative_img = occluded_coco_retrieval_ann_file['cat_id_2_negative_img']
        img_id_2_all_ins_id = occluded_coco_retrieval_ann_file['img_id_2_all_ins_id']
        coco_91 = json.load(open('DATASET_PATH/imagenet-a-cat2name.json'))
        img_preprocess = pth_transforms.Compose([
            pth_transforms.Resize([224, 224], pth_transforms.InterpolationMode.BICUBIC),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])


        with torch.no_grad():

            cat_id_2_txt_feat = {}
            iter_txt = 0
            for cat_id in cat_id_2_occ_img.keys():
                print(iter_txt)
                iter_txt += 1
                cur_txt = coco_91[cat_id]
                text = tokenizer([cur_txt]).to(device=device, non_blocking=True)
                text_features = model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                cat_id_2_txt_feat[cat_id] = text_features.detach().cpu().numpy()

            img_id_2_img_feat = {}
            img_folder = 'DATASET_PATH/imagenet-a'
            iter_img = 0
            for img_id in img_id_2_all_ins_id:
                # ipdb.set_trace()
                print(iter_img)
                iter_img += 1
                cur_image = img_preprocess(Image.open(os.path.join(img_folder, img_id)).convert('RGB')).unsqueeze(0)
                # if img_id == '289343':
                cur_image = cur_image.to(device=device, dtype=input_dtype, non_blocking=True)
                image_features = model.encode_image(cur_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                img_id_2_img_feat[img_id] = image_features.detach().cpu().numpy()

            ipdb.set_trace()
            with open('DATASET_PATH/imagenet-a_clip_feat.pkl', 'wb') as dump_f:
                pickle.dump({'cat_id_2_txt_feat':cat_id_2_txt_feat, 'img_id_2_img_feat':img_id_2_img_feat}, dump_f)
            

            top1_recall = 0
            top5_recall = 0
            top10_recall = 0
            recall_cnt = 0
            for cat_id in cat_id_2_txt_feat.keys():
                if cat_id == '1':
                    continue
                print(coco_91[int(cat_id)])
                cur_txt_feat = cat_id_2_txt_feat[cat_id]
                logit_list = []
                occ_img_id_list = cat_id_2_occ_img[cat_id]
                neg_img_id_list = cat_id_2_negative_img[cat_id]
                for img_id in occ_img_id_list:
                    cur_img_feat = img_id_2_img_feat[str(img_id)]
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0].cpu().numpy().tolist()
                    logit_list.append([cur_logit, 1])
                for img_id in neg_img_id_list:
                    cur_img_feat = img_id_2_img_feat[str(img_id)]
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0].cpu().numpy().tolist()
                    logit_list.append([cur_logit, 0])
                logit_list.sort(key=lambda x: -x[0])
                logit_list = np.array(logit_list)[:,1]
                print(logit_list[:len(occ_img_id_list)])
                if logit_list[:len(occ_img_id_list)].sum() == len(occ_img_id_list):
                    top1_recall += 1
                if logit_list[:5*len(occ_img_id_list)].sum() == len(occ_img_id_list):
                    top5_recall += 1
                if logit_list[:10*len(occ_img_id_list)].sum() == len(occ_img_id_list):
                    top10_recall += 1
                recall_cnt += 1

            print('top1 recall: ', top1_recall / recall_cnt)
            print('top5 recall: ', top5_recall / recall_cnt)
            print('top10 recall: ', top10_recall / recall_cnt)
            print('recall cnt: ', recall_cnt)
    return 0





def evaluate_occluded_coco_tgvpt3(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    logging.info('start evaluation')
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        logging.info('start evaluation')

        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []


        from torchvision import transforms as pth_transforms
        from PIL import Image
        occluded_coco_retrieval_ann_file = json.load(open('standard_benchmarks/occ_coco/karpathy_test_cat_id_2_occ_img_negative_img.json'))
        cat_id_2_occ_img = occluded_coco_retrieval_ann_file['cat_id_2_occ_img']
        cat_id_2_negative_img = occluded_coco_retrieval_ann_file['cat_id_2_negative_img']
        # img_id_2_all_ins_id = occluded_coco_retrieval_ann_file['img_id_2_all_ins_id']
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
        img_preprocess = pth_transforms.Compose([
            pth_transforms.Resize([224, 224], pth_transforms.InterpolationMode.BICUBIC),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])


        with torch.no_grad():

            # ours_cat_id_2_txt_feat = {}
            # iter_txt = 0
            # for cat_id in cat_id_2_occ_img.keys():
            #     print(iter_txt)
            #     iter_txt += 1
            #     cur_txt = coco_91[int(cat_id)]
            #     text = tokenizer([cur_txt]).to(device=device, non_blocking=True)
            #     text_features = model.encode_text(text)
            #     text_features /= text_features.norm(dim=-1, keepdim=True)
            #     ours_cat_id_2_txt_feat[cat_id] = text_features

            # img_id_2_img_feat = {}
            # img_folder = '/scratch/shared/beegfs/shared-datasets/COCO/COCO2017/val2017/'
            # img_folder2 = '/scratch/shared/beegfs/shared-datasets/COCO/COCO2017/train2017/'
            img_folder = '/home/ypliu/datasets/coco/val2017/'
            img_folder2 = '/home/ypliu/datasets/coco/train2017/'
            # iter_img = 0
            # for img_id in img_id_2_all_ins_id.keys():
            #     # ipdb.set_trace()
            #     # print(iter_img)
            #     iter_img += 1
            #     cur_image = img_preprocess(Image.open(os.path.join(img_folder, str(img_id).rjust(12,'0')+'.jpg')).convert('RGB')).unsqueeze(0)
            #     # if img_id == '289343':
            #     cur_image = cur_image.to(device=device, dtype=input_dtype, non_blocking=True)
            #     image_features = model.encode_image(cur_image)
            #     image_features /= image_features.norm(dim=-1, keepdim=True)
            #     img_id_2_img_feat[img_id] = image_features



            # top1_recall = 0
            # top5_recall = 0
            # top10_recall = 0
            
            cur_pkl = pickle.load(open('standard_benchmarks/occ_coco/occluded_coco_clip_feat_revised_11.7.pkl', 'rb'))
            cat_id_2_txt_feat = cur_pkl['cat_id_2_txt_feat']
            img_id_2_img_feat = cur_pkl['img_id_2_img_feat']
            recall_cnt = 0
            ap_sum = 0
            topk = 500 # topK in reranking
            # ipdb.set_trace()
            all_precisions = []
            all_new_logit_list = []

            # recall_topk_list = [100, 200, 500, 1000, 2000]
            # recall_topk_initial = {}
            # for topk_i in recall_topk_list:
            #     recall_topk_initial[topk_i] = 0

            
            # iter_i1 = 0
            # selected_idxs = [0,1,2,6,15,16,17,28,30,34,37,47,48,55,57,58,64,67,69,75]
            
            
            for cat_id in cat_id_2_txt_feat.keys():
                if cat_id == '1':
                    continue

                # if int(cat_id) > 10:
                #     continue
                
                cur_txt_feat = cat_id_2_txt_feat[cat_id]
                # ours_cur_txt_feat = ours_cat_id_2_txt_feat[cat_id]
                logit_list = []
                occ_img_id_list = cat_id_2_occ_img[cat_id]
                neg_img_id_list = cat_id_2_negative_img[cat_id]
                if len(occ_img_id_list) == 0:
                    continue

                print(coco_91[int(cat_id)])

                # if iter_i1 not in selected_idxs:
                #     iter_i1 += 1
                #     continue
                # else:
                #     iter_i1 += 1


                for img_id in occ_img_id_list:
                    cur_img_feat = img_id_2_img_feat[img_id].to(cur_txt_feat.device)
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0].cpu().numpy().tolist()
                    logit_list.append([cur_logit, 1, img_id])
                for img_id in neg_img_id_list:
                    cur_img_feat = img_id_2_img_feat[img_id].to(cur_txt_feat.device)
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0].cpu().numpy().tolist()
                    logit_list.append([cur_logit, 0, img_id])
                logit_list.sort(key=lambda x: -x[0])

                # array_logit_list = np.array(logit_list)[:,1]
                # if len(occ_img_id_list) > 0:
                #     for key_i in recall_topk_initial.keys():
                #         recall_topk_initial[key_i] += array_logit_list[:int(key_i)].sum() / array_logit_list.sum()
                #     recall_cnt += 1
                #     print('initial recalled within topk: ', array_logit_list[:topk].sum() / array_logit_list.sum())
                # continue


                # ipdb.set_trace()
                new_image_list = [] # expected BS=100
                new_logit_list = []
                new_vis_logit_list = []
                for top_i in range(min(topk, len(logit_list))):
                    # print(top_i)
                    if os.path.exists(os.path.join(img_folder, str(logit_list[top_i][-1]).rjust(12,'0')+'.jpg')):
                        cur_image = img_preprocess(Image.open(os.path.join(img_folder, str(logit_list[top_i][-1]).rjust(12,'0')+'.jpg')).convert('RGB'))
                    else:
                        cur_image = img_preprocess(Image.open(os.path.join(img_folder2, str(logit_list[top_i][-1]).rjust(12,'0')+'.jpg')).convert('RGB'))
                    cur_image = cur_image.to(device=device, dtype=input_dtype, non_blocking=True).unsqueeze(0)
                    new_image_list.append(cur_image)
                new_image_list = torch.cat(new_image_list, dim=0) # 100 x 3 x 224 x 224
                # ipdb.set_trace()
                new_image_features = model(new_image_list, None, cur_txt_feat.repeat(new_image_list.shape[0], 1))["image_features"]
                new_logit = (new_image_features @ cur_txt_feat.T)
                for top_i in range(new_logit.shape[0]):
                    new_logit_list.append([new_logit[top_i][0].cpu().numpy().tolist(), logit_list[top_i][1]])
                    new_vis_logit_list.append([new_logit[top_i][0].cpu().numpy().tolist(), logit_list[top_i][1], logit_list[top_i][2]])
                for top_i in range(min(topk, len(logit_list)), len(logit_list)):
                # for top_i in range(len(logit_list)):
                    new_logit_list.append([logit_list[top_i][0]-1000, logit_list[top_i][1]])
                    new_vis_logit_list.append([logit_list[top_i][0]-1000, logit_list[top_i][1], logit_list[top_i][2]])
                # ipdb.set_trace()
                # cur_ap = calculate_ap(new_logit_list)
                # # for vis 11.9 
                # new_vis_logit_list.sort(key=lambda x: -x[0])
                # if not os.path.exists(vis_dir):
                #     os.makedirs(vis_dir, exist_ok=True)
                # # ipdb.set_trace()
                # sum_org = 0
                # sum_ours = 0
                # for top_i in range(10):
                #     sum_org += logit_list[top_i][1]
                #     sum_ours += new_vis_logit_list[top_i][1]
                # # if sum_ours >= sum_org + 3:
                # if 1 > 0:
                #     # ipdb.set_trace()
                #     for top_i in range(10):
                #         print(top_i)
                #         if os.path.exists(os.path.join(img_folder, str(logit_list[top_i][-1]).rjust(12,'0')+'.jpg')):
                #             shutil.copy(os.path.join(img_folder, str(logit_list[top_i][-1]).rjust(12,'0')+'.jpg'), os.path.join(vis_dir, 'txti_' + coco_91[int(cat_id)] + '-before-top_' + str(top_i) + '-csvid_' + str(logit_list[top_i][-1]) + '-label-' + str(logit_list[top_i][1]) + '-score_' + str(logit_list[top_i][0]) + '.png'))
                #         else:
                #             shutil.copy(os.path.join(img_folder2, str(logit_list[top_i][-1]).rjust(12,'0')+'.jpg'), os.path.join(vis_dir, 'txti_' + coco_91[int(cat_id)] + '-before-top_' + str(top_i) + '-csvid_' + str(logit_list[top_i][-1]) + '-label-' + str(logit_list[top_i][1]) + '-score_' + str(logit_list[top_i][0]) + '.png'))
                #         if os.path.exists(os.path.join(img_folder, str(new_vis_logit_list[top_i][-1]).rjust(12,'0')+'.jpg')):   
                #             shutil.copy(os.path.join(img_folder, str(new_vis_logit_list[top_i][-1]).rjust(12,'0')+'.jpg'), os.path.join(vis_dir, 'txti_' + coco_91[int(cat_id)] + '-after-top_' + str(top_i) + '-csvid_' + str(new_vis_logit_list[top_i][-1]) + '-label-' + str(new_vis_logit_list[top_i][1]) + '-score_' + str(new_vis_logit_list[top_i][0]) + '.png'))
                #         else:
                #             shutil.copy(os.path.join(img_folder2, str(new_vis_logit_list[top_i][-1]).rjust(12,'0')+'.jpg'), os.path.join(vis_dir, 'txti_' + coco_91[int(cat_id)] + '-after-top_' + str(top_i) + '-csvid_' + str(new_vis_logit_list[top_i][-1]) + '-label-' + str(new_vis_logit_list[top_i][1]) + '-score_' + str(new_vis_logit_list[top_i][0]) + '.png'))
                cur_ap, cur_precisions = calculate_ap_new(new_logit_list)
                all_precisions.append(cur_precisions)
                
                # if cur_precisions[0] < 0.5:
                #     ipdb.set_trace()

                ap_sum += cur_ap
                recall_cnt += 1
                all_new_logit_list.extend(new_logit_list)
                # except:
                #     # ipdb.set_trace()
                #     continue
                

        # for key_i in recall_topk_initial.keys():
        #     print('initial recall@top' + str(int(key_i)) + ': ', recall_topk_initial[key_i] / recall_cnt)
        # print(recall_cnt)
        
        all_cur_ap, all_cur_precisions = calculate_ap_new(all_new_logit_list)
        all_cur_precisions_recall_target = np.arange(len(all_cur_precisions))
            
        print('recall_cnt: ', recall_cnt)
        print('mAP: ', ap_sum / recall_cnt)
        ipdb.set_trace()
        from scipy.interpolate import interp1d
        updated_precisions = []
        target_length = max(len(p) for p in all_precisions)
        for precisions in all_precisions:
            if len(precisions) == 1:
                continue
            x_original = np.linspace(0, 1, len(precisions))
            x_target = np.linspace(0, 1, target_length)
            # interpolation_function = interp1d(x_original, precisions, kind='nearest', fill_value="extrapolate")
            interpolation_function = interp1d(x_original, precisions, kind='linear')
            interpolated_precisions = interpolation_function(x_target)
            updated_precisions.append(interpolated_precisions)
        recall_target = np.linspace(0, 1, target_length)
        avg_precisions = np.mean(updated_precisions, axis=0)
        # plot_average_precision_curve(avg_precisions, recall_target, 'occluded_coco_mAP_hard_sample.png')
        np_save_folder = 'occluded_coco_mAP_curve_3.5'
        if not os.path.exists(np_save_folder):
            os.makedirs(np_save_folder, exist_ok=True)
        # np.save(os.path.join(np_save_folder, 'v1_ours_recall_target.npy'), recall_target)
        # np.save(os.path.join(np_save_folder, 'v1_ours_avg_precisions.npy'), avg_precisions)
        # np.save(os.path.join(np_save_folder, 'v1_ours_all_precisions.npy'), updated_precisions)
        np.save(os.path.join(np_save_folder, 'original_clip_recall_target.npy'), recall_target)
        np.save(os.path.join(np_save_folder, 'original_clip_avg_precisions.npy'), avg_precisions)
        np.save(os.path.join(np_save_folder, 'original_clip_all_precisions.npy'), updated_precisions)
        # np.save(os.path.join(np_save_folder, 'v1_ft_recall_target.npy'), recall_target)
        # np.save(os.path.join(np_save_folder, 'v1_ft_avg_precisions.npy'), avg_precisions)
        # np.save(os.path.join(np_save_folder, 'v1_ft_all_precisions.npy'), updated_precisions)

        # np.save(os.path.join(np_save_folder, 'v1_original_clip_recall_target_method2.npy'), all_cur_precisions_recall_target)
        # np.save(os.path.join(np_save_folder, 'v1_original_clip_avg_precisions_method2.npy'), np.array(all_cur_precisions))
        # np.save(os.path.join(np_save_folder, 'v1_ours_recall_target_method2.npy'), all_cur_precisions_recall_target)
        # np.save(os.path.join(np_save_folder, 'v1_ours_avg_precisions_method2.npy'), np.array(all_cur_precisions))
        

        ipdb.set_trace()
        # with open('occluded_coco_mAP_org_blip_avg_precisions.json', 'w') as dump_f:
        #     json.dump(avg_precisions, dump_f)

    return 0


def evaluate_occluded_coco_tgvpt3_siglip(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    logging.info('start evaluation')
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        logging.info('start evaluation')

        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []


        from torchvision import transforms as pth_transforms
        from PIL import Image
        occluded_coco_retrieval_ann_file = json.load(open('standard_benchmarks/occ_coco/karpathy_test_cat_id_2_occ_img_negative_img.json'))
        cat_id_2_occ_img = occluded_coco_retrieval_ann_file['cat_id_2_occ_img']
        cat_id_2_negative_img = occluded_coco_retrieval_ann_file['cat_id_2_negative_img']
        # img_id_2_all_ins_id = occluded_coco_retrieval_ann_file['img_id_2_all_ins_id']
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
        img_preprocess = pth_transforms.Compose([
            # pth_transforms.Resize([224, 224], pth_transforms.InterpolationMode.BICUBIC),
            pth_transforms.Resize([384, 384], pth_transforms.InterpolationMode.BICUBIC), # for resolution 384
            pth_transforms.ToTensor(),
            # pth_transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # For SigLIP
        ])


        with torch.no_grad():

            # ours_cat_id_2_txt_feat = {}
            # iter_txt = 0
            # for cat_id in cat_id_2_occ_img.keys():
            #     print(iter_txt)
            #     iter_txt += 1
            #     cur_txt = coco_91[int(cat_id)]
            #     text = tokenizer([cur_txt]).to(device=device, non_blocking=True)
            #     text_features = model.encode_text(text)
            #     text_features /= text_features.norm(dim=-1, keepdim=True)
            #     ours_cat_id_2_txt_feat[cat_id] = text_features

            # img_id_2_img_feat = {}
            # img_folder = '/scratch/shared/beegfs/shared-datasets/COCO/COCO2017/val2017/'
            # img_folder2 = '/scratch/shared/beegfs/shared-datasets/COCO/COCO2017/train2017/'
            img_folder = '/home/ypliu/datasets/coco/val2017/'
            img_folder2 = '/home/ypliu/datasets/coco/train2017/'
            # iter_img = 0
            # for img_id in img_id_2_all_ins_id.keys():
            #     # ipdb.set_trace()
            #     # print(iter_img)
            #     iter_img += 1
            #     cur_image = img_preprocess(Image.open(os.path.join(img_folder, str(img_id).rjust(12,'0')+'.jpg')).convert('RGB')).unsqueeze(0)
            #     # if img_id == '289343':
            #     cur_image = cur_image.to(device=device, dtype=input_dtype, non_blocking=True)
            #     image_features = model.encode_image(cur_image)
            #     image_features /= image_features.norm(dim=-1, keepdim=True)
            #     img_id_2_img_feat[img_id] = image_features



            # top1_recall = 0
            # top5_recall = 0
            # top10_recall = 0

            cur_pkl = pickle.load(open('standard_benchmarks/occ_coco/occluded_coco_revised_siglipSO_feat.pkl', 'rb'))
            # cur_pkl = pickle.load(open('standard_benchmarks/occ_coco/occluded_coco_revised_siglip2G_feat.pkl', 'rb'))
            cat_id_2_txt_feat = cur_pkl['cat_id_2_txt_feat']
            img_id_2_img_feat = cur_pkl['img_id_2_img_feat']
            recall_cnt = 0
            ap_sum = 0
            topk = 500 # 3000 # topK in reranking
            # ipdb.set_trace()
            all_precisions = []
            all_new_logit_list = []

            # recall_topk_list = [50, 100, 200, 500, 1000]
            # recall_topk_initial = {}
            # for topk_i in recall_topk_list:
            #     recall_topk_initial[topk_i] = 0

            for cat_id in cat_id_2_txt_feat.keys():
                if cat_id == '1':
                    continue
                # if int(cat_id) > 10:
                #     continue
                print(coco_91[int(cat_id)])
                cur_txt_feat = cat_id_2_txt_feat[cat_id]
                # ours_cur_txt_feat = ours_cat_id_2_txt_feat[cat_id]
                logit_list = []
                occ_img_id_list = cat_id_2_occ_img[cat_id]
                neg_img_id_list = cat_id_2_negative_img[cat_id]
                if len(occ_img_id_list) == 0:
                    continue
                for img_id in occ_img_id_list:
                    cur_img_feat = img_id_2_img_feat[img_id].to(cur_txt_feat.device)
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0].cpu().numpy().tolist()
                    logit_list.append([cur_logit, 1, img_id])
                for img_id in neg_img_id_list:
                    cur_img_feat = img_id_2_img_feat[img_id].to(cur_txt_feat.device)
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0].cpu().numpy().tolist()
                    logit_list.append([cur_logit, 0, img_id])
                logit_list.sort(key=lambda x: -x[0])

                # array_logit_list = np.array(logit_list)[:,1]
                # if len(occ_img_id_list) > 0:
                #     for key_i in recall_topk_initial.keys():
                #         recall_topk_initial[key_i] += array_logit_list[:int(key_i)].sum() / array_logit_list.sum()
                #     recall_cnt += 1
                #     print('initial recalled within topk: ', array_logit_list[:topk].sum() / array_logit_list.sum())
                # continue


                # # ipdb.set_trace()
                new_image_list = [] # expected BS=100
                new_logit_list = []
                new_vis_logit_list = []
                for top_i in range(min(topk, len(logit_list))):
                    # print(top_i)
                    if os.path.exists(os.path.join(img_folder, str(logit_list[top_i][-1]).rjust(12,'0')+'.jpg')):
                        cur_image = img_preprocess(Image.open(os.path.join(img_folder, str(logit_list[top_i][-1]).rjust(12,'0')+'.jpg')).convert('RGB'))
                    else:
                        cur_image = img_preprocess(Image.open(os.path.join(img_folder2, str(logit_list[top_i][-1]).rjust(12,'0')+'.jpg')).convert('RGB'))
                    cur_image = cur_image.to(device=device, dtype=input_dtype, non_blocking=True).unsqueeze(0)
                    new_image_list.append(cur_image)
                new_image_list = torch.cat(new_image_list, dim=0) # 100 x 3 x 224 x 224
                # ipdb.set_trace()
                # new_image_features = model(new_image_list, None, cur_txt_feat.repeat(new_image_list.shape[0], 1))["image_features"]
                # new_image_features = model.encode_image(new_image_list.cuda(), text_feat=cur_txt_feat.repeat(new_image_list.shape[0], 1), normalize=True)
                # -------------------------------- run the model via batches --------------------------------
                bs = 32  # choose a batch size that fits your GPU
                N = new_image_list.shape[0]
                outs = []
                for i in range(0, N, bs):
                    imgs = new_image_list[i:i + bs].to(device)  # (b, ...)
                    out = model(imgs, None, cur_txt_feat.repeat(imgs.shape[0], 1))["image_features"]  # (b, F)
                    outs.append(out)
                new_image_features = torch.cat(outs, dim=0)
                # -------------------------------- run the model via batches --------------------------------

                new_logit = (new_image_features @ cur_txt_feat.T)
                for top_i in range(new_logit.shape[0]):
                    new_logit_list.append([new_logit[top_i][0].cpu().numpy().tolist(), logit_list[top_i][1]])
                    new_vis_logit_list.append([new_logit[top_i][0].cpu().numpy().tolist(), logit_list[top_i][1], logit_list[top_i][2]])
                for top_i in range(min(topk, len(logit_list)), len(logit_list)):
                # for top_i in range(len(logit_list)):
                    new_logit_list.append([logit_list[top_i][0]-1000, logit_list[top_i][1]])
                    new_vis_logit_list.append([logit_list[top_i][0]-1000, logit_list[top_i][1], logit_list[top_i][2]])
                # ipdb.set_trace()
                # cur_ap = calculate_ap(new_logit_list)
                # # for vis 11.9 
                # new_vis_logit_list.sort(key=lambda x: -x[0])
                # # ipdb.set_trace()
                # sum_org = 0
                # sum_ours = 0
                # for top_i in range(10):
                #     sum_org += logit_list[top_i][1]
                #     sum_ours += new_vis_logit_list[top_i][1]
                # if sum_ours >= sum_org + 3:
                #     # ipdb.set_trace()
                #     for top_i in range(10):
                #         print(top_i)
                #         if os.path.exists(os.path.join(img_folder, str(logit_list[top_i][-1]).rjust(12,'0')+'.jpg')):
                #             shutil.copy(os.path.join(img_folder, str(logit_list[top_i][-1]).rjust(12,'0')+'.jpg'), os.path.join(vis_dir, 'txti_' + coco_91[int(cat_id)] + '-before-top_' + str(top_i) + '-csvid_' + str(logit_list[top_i][-1]) + '-label-' + str(logit_list[top_i][1]) + '-score_' + str(logit_list[top_i][0]) + '.png'))
                #         else:
                #             shutil.copy(os.path.join(img_folder2, str(logit_list[top_i][-1]).rjust(12,'0')+'.jpg'), os.path.join(vis_dir, 'txti_' + coco_91[int(cat_id)] + '-before-top_' + str(top_i) + '-csvid_' + str(logit_list[top_i][-1]) + '-label-' + str(logit_list[top_i][1]) + '-score_' + str(logit_list[top_i][0]) + '.png'))
                #         if os.path.exists(os.path.join(img_folder, str(new_vis_logit_list[top_i][-1]).rjust(12,'0')+'.jpg')):   
                #             shutil.copy(os.path.join(img_folder, str(new_vis_logit_list[top_i][-1]).rjust(12,'0')+'.jpg'), os.path.join(vis_dir, 'txti_' + coco_91[int(cat_id)] + '-after-top_' + str(top_i) + '-csvid_' + str(new_vis_logit_list[top_i][-1]) + '-label-' + str(new_vis_logit_list[top_i][1]) + '-score_' + str(new_vis_logit_list[top_i][0]) + '.png'))
                #         else:
                #             shutil.copy(os.path.join(img_folder2, str(new_vis_logit_list[top_i][-1]).rjust(12,'0')+'.jpg'), os.path.join(vis_dir, 'txti_' + coco_91[int(cat_id)] + '-after-top_' + str(top_i) + '-csvid_' + str(new_vis_logit_list[top_i][-1]) + '-label-' + str(new_vis_logit_list[top_i][1]) + '-score_' + str(new_vis_logit_list[top_i][0]) + '.png'))
                cur_ap, cur_precisions = calculate_ap_new(new_logit_list)
                all_precisions.append(cur_precisions)
                
                # if cur_precisions[0] < 0.5:
                #     ipdb.set_trace()

                ap_sum += cur_ap
                recall_cnt += 1
                all_new_logit_list.extend(new_logit_list)
                # except:
                #     # ipdb.set_trace()
                #     continue
                

        # for key_i in recall_topk_initial.keys():
        #     print('initial recall@top' + str(int(key_i)) + ': ', recall_topk_initial[key_i] / recall_cnt)
        # print(recall_cnt)
        
        all_cur_ap, all_cur_precisions = calculate_ap_new(all_new_logit_list)
        all_cur_precisions_recall_target = np.arange(len(all_cur_precisions))
            
        print('recall_cnt: ', recall_cnt)
        print('mAP: ', ap_sum / recall_cnt)
        ipdb.set_trace()
        from scipy.interpolate import interp1d
        updated_precisions = []
        target_length = max(len(p) for p in all_precisions)
        iter_ii = 0
        for precisions in all_precisions:
            print(iter_ii)
            iter_ii += 1
            if len(precisions) == 1:
                ipdb.set_trace()
                continue
            x_original = np.linspace(0, 1, len(precisions))
            x_target = np.linspace(0, 1, target_length)
            # interpolation_function = interp1d(x_original, precisions, kind='nearest', fill_value="extrapolate")
            interpolation_function = interp1d(x_original, precisions, kind='linear')
            interpolated_precisions = interpolation_function(x_target)
            updated_precisions.append(interpolated_precisions)
        recall_target = np.linspace(0, 1, target_length)
        avg_precisions = np.mean(updated_precisions, axis=0)
        # plot_average_precision_curve(avg_precisions, recall_target, 'occluded_coco_mAP_hard_sample.png')
        np_save_folder = 'occluded_coco_mAP_curve_3.7'
        if not os.path.exists(np_save_folder):
            os.makedirs(np_save_folder, exist_ok=True)
        # np.save(os.path.join(np_save_folder, 'v1_ours_recall_target.npy'), recall_target)
        # np.save(os.path.join(np_save_folder, 'v1_ours_avg_precisions.npy'), avg_precisions)
        # np.save(os.path.join(np_save_folder, 'v1_ours_all_precisions.npy'), updated_precisions)
        # np.save(os.path.join(np_save_folder, 'v1_original_clip_recall_target.npy'), recall_target)
        # np.save(os.path.join(np_save_folder, 'v1_original_clip_avg_precisions.npy'), avg_precisions)
        # np.save(os.path.join(np_save_folder, 'v1_original_clip_all_precisions.npy'), updated_precisions)
        # np.save(os.path.join(np_save_folder, 'siglip2_ft_recall_target.npy'), recall_target)
        # np.save(os.path.join(np_save_folder, 'siglip2_ft_avg_precisions.npy'), avg_precisions)
        # np.save(os.path.join(np_save_folder, 'siglip2_ft_all_precisions.npy'), updated_precisions)
        np.save(os.path.join(np_save_folder, 'siglip_org_recall_target.npy'), recall_target)
        np.save(os.path.join(np_save_folder, 'siglip_org_avg_precisions.npy'), avg_precisions)
        np.save(os.path.join(np_save_folder, 'siglip_org_all_precisions.npy'), updated_precisions)

        # np.save(os.path.join(np_save_folder, 'v1_original_clip_recall_target_method2.npy'), all_cur_precisions_recall_target)
        # np.save(os.path.join(np_save_folder, 'v1_original_clip_avg_precisions_method2.npy'), np.array(all_cur_precisions))
        # np.save(os.path.join(np_save_folder, 'v1_ours_recall_target_method2.npy'), all_cur_precisions_recall_target)
        # np.save(os.path.join(np_save_folder, 'v1_ours_avg_precisions_method2.npy'), np.array(all_cur_precisions))
        

        ipdb.set_trace()
        # with open('occluded_coco_mAP_org_blip_avg_precisions.json', 'w') as dump_f:
        #     json.dump(avg_precisions, dump_f)

    return 0




# Plot the average mAP curve
def plot_average_precision_curve(avg_precisions, recall_target, save_pth):
    plt.plot(recall_target, avg_precisions, label='mAP')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Average Precision Curve for Occluded COCO')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_pth)





def evaluate_occluded_coco_tgvpt3_logo_prompt(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    logging.info('start evaluation')
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        logging.info('start evaluation')

        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []


        from torchvision import transforms as pth_transforms
        from PIL import Image
        from PIL import ImageDraw,ImageFont
        import random
        occluded_coco_retrieval_ann_file = json.load(open('DATASET_PATH/coco2017val_cat_id_2_occ_img_negative_img.json'))
        cat_id_2_occ_img = occluded_coco_retrieval_ann_file['cat_id_2_occ_img']
        cat_id_2_negative_img = occluded_coco_retrieval_ann_file['cat_id_2_negative_img']
        img_id_2_all_ins_id = occluded_coco_retrieval_ann_file['img_id_2_all_ins_id']
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
        img_preprocess = pth_transforms.Compose([
            pth_transforms.Resize([224, 224], pth_transforms.InterpolationMode.BICUBIC),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        logo_prompt_transform = pth_transforms.Compose([
            pth_transforms.Resize((16, 192), interpolation=pth_transforms.InterpolationMode.BICUBIC),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711])
        ])
        character_font = ImageFont.truetype(r'PROJECT_PATH/Arial.ttf', size=16)

        with torch.no_grad():

            # cat_id_2_txt_feat = {}
            # iter_txt = 0
            # for cat_id in cat_id_2_occ_img.keys():
            #     print(iter_txt)
            #     iter_txt += 1
            #     cur_txt = coco_91[int(cat_id)]
            #     text = tokenizer([cur_txt]).to(device=device, non_blocking=True)
            #     text_features = model.encode_text(text)
            #     text_features /= text_features.norm(dim=-1, keepdim=True)
            #     cat_id_2_txt_feat[cat_id] = text_features

            # img_id_2_img_feat = {}
            img_folder = '/scratch/shared/beegfs/shared-datasets/COCO/COCO2017/val2017/'
            # iter_img = 0
            # for img_id in img_id_2_all_ins_id.keys():
            #     # ipdb.set_trace()
            #     # print(iter_img)
            #     iter_img += 1
            #     cur_image = img_preprocess(Image.open(os.path.join(img_folder, str(img_id).rjust(12,'0')+'.jpg')).convert('RGB')).unsqueeze(0)
            #     # if img_id == '289343':
            #     cur_image = cur_image.to(device=device, dtype=input_dtype, non_blocking=True)
            #     image_features = model.encode_image(cur_image)
            #     image_features /= image_features.norm(dim=-1, keepdim=True)
            #     img_id_2_img_feat[img_id] = image_features



            # top1_recall = 0
            # top5_recall = 0
            # top10_recall = 0
            
            cur_pkl = pickle.load(open('DATASET_PATH/occluded_coco_clip_feat.pkl', 'rb'))
            cat_id_2_txt_feat = cur_pkl['cat_id_2_txt_feat']
            img_id_2_img_feat = cur_pkl['img_id_2_img_feat']
            recall_cnt = 0
            ap_sum = 0
            topk = 100 # topK in reranking
            for cat_id in cat_id_2_txt_feat.keys():

                # generate logo prompt
                size = character_font.getsize(text=(coco_91[int(cat_id)]))
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                im = Image.new("RGB", size, (0, 0, 0))
                draw = ImageDraw.Draw(im)
                draw.text((0, 0), str(coco_91[int(cat_id)]), fill=(255 - r, 255 - g, 255 - b), font=character_font, align="right")
                logo_prompt_im = logo_prompt_transform(im).to(device)

                if cat_id == '1':
                    continue
                print(coco_91[int(cat_id)])
                cur_txt_feat = cat_id_2_txt_feat[cat_id]
                logit_list = []
                occ_img_id_list = cat_id_2_occ_img[cat_id]
                neg_img_id_list = cat_id_2_negative_img[cat_id]
                for img_id in occ_img_id_list:
                    cur_img_feat = img_id_2_img_feat[str(img_id)]
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0].cpu().numpy().tolist()
                    logit_list.append([cur_logit, 1, img_id])
                for img_id in neg_img_id_list:
                    cur_img_feat = img_id_2_img_feat[str(img_id)]
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0].cpu().numpy().tolist()
                    logit_list.append([cur_logit, 0, img_id])
                logit_list.sort(key=lambda x: -x[0])
                # ipdb.set_trace()
                new_image_list = [] # expected BS=100
                new_logit_list = []
                for top_i in range(min(topk, len(logit_list))):
                    # print(top_i)
                    # cur_image = Image.open(os.path.join(img_folder, str(logit_list[top_i][-1]).rjust(12,'0')+'.jpg')).convert('RGB')
                    cur_image = img_preprocess(Image.open(os.path.join(img_folder, str(logit_list[top_i][-1]).rjust(12,'0')+'.jpg')).convert('RGB'))
                    cur_image = cur_image.to(device=device, dtype=input_dtype, non_blocking=True).unsqueeze(0)
                    
                    
                    # if args.logo_prompt:
                    x_ = 0
                    y_ = 0
                    prompt = torch.zeros(1, 3, 224, 224).cuda()
                    prompt[:, :, x_:x_ + 16, y_:y_ + 192] = logo_prompt_im.unsqueeze(0)
                    cur_image = cur_image + prompt

                    new_image_list.append(cur_image)
                new_image_list = torch.cat(new_image_list, dim=0) # 100 x 3 x 224 x 224
                # ipdb.set_trace()
                new_image_features = model(new_image_list, None, cur_txt_feat.repeat(new_image_list.shape[0], 1))["image_features"]
                new_logit = (new_image_features @ cur_txt_feat.T)
                for top_i in range(new_logit.shape[0]):
                    new_logit_list.append([new_logit[top_i][0].cpu().numpy().tolist(), logit_list[top_i][1]])
                for top_i in range(min(topk, len(logit_list)), len(logit_list)):
                    new_logit_list.append([logit_list[top_i][0], logit_list[top_i][1]])
                # ipdb.set_trace()
                cur_ap = calculate_ap(new_logit_list)
                ap_sum += cur_ap
                recall_cnt += 1

            print('mAP: ', ap_sum / recall_cnt)

    return 0



def evaluate_occluded_coco_tgvpt3_sd(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    logging.info('start evaluation')
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        logging.info('start evaluation')

        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []


        from torchvision import transforms as pth_transforms
        from PIL import Image
        occluded_coco_retrieval_ann_file = json.load(open('DATASET_PATH/coco2017val_cat_id_2_occ_img_negative_img.json'))
        cat_id_2_occ_img = occluded_coco_retrieval_ann_file['cat_id_2_occ_img']
        cat_id_2_negative_img = occluded_coco_retrieval_ann_file['cat_id_2_negative_img']
        img_id_2_all_ins_id = occluded_coco_retrieval_ann_file['img_id_2_all_ins_id']
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
        img_preprocess = pth_transforms.Compose([
            pth_transforms.Resize([224, 224], pth_transforms.InterpolationMode.BICUBIC),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])


        with torch.no_grad():

            # cat_id_2_txt_feat = {}
            # iter_txt = 0
            # for cat_id in cat_id_2_occ_img.keys():
            #     print(iter_txt)
            #     iter_txt += 1
            #     cur_txt = coco_91[int(cat_id)]
            #     text = tokenizer([cur_txt]).to(device=device, non_blocking=True)
            #     text_features = model.encode_text(text)
            #     text_features /= text_features.norm(dim=-1, keepdim=True)
            #     cat_id_2_txt_feat[cat_id] = text_features

            # img_id_2_img_feat = {}
            img_folder = '/scratch/shared/beegfs/shared-datasets/COCO/COCO2017/val2017/'
            # iter_img = 0
            # for img_id in img_id_2_all_ins_id.keys():
            #     # ipdb.set_trace()
            #     # print(iter_img)
            #     iter_img += 1
            #     cur_image = img_preprocess(Image.open(os.path.join(img_folder, str(img_id).rjust(12,'0')+'.jpg')).convert('RGB')).unsqueeze(0)
            #     # if img_id == '289343':
            #     cur_image = cur_image.to(device=device, dtype=input_dtype, non_blocking=True)
            #     image_features = model.encode_image(cur_image)
            #     image_features /= image_features.norm(dim=-1, keepdim=True)
            #     img_id_2_img_feat[img_id] = image_features



            # top1_recall = 0
            # top5_recall = 0
            # top10_recall = 0
            
            cur_pkl = pickle.load(open('DATASET_PATH/occluded_coco_clip_feat.pkl', 'rb'))
            cat_id_2_txt_feat = cur_pkl['cat_id_2_txt_feat']
            img_id_2_img_feat = cur_pkl['img_id_2_img_feat']
            recall_cnt = 0
            ap_sum = 0
            topk = 100 # topK in reranking
            sd_img_folder = 'DATASET_PATH/coco_sd_generated_imgs_9.22_3'
            img_preprocess = pth_transforms.Compose([
                pth_transforms.Resize([224, 224], pth_transforms.InterpolationMode.BICUBIC),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
            for cat_id in cat_id_2_txt_feat.keys():
                if cat_id == '1':
                    continue
                print(coco_91[int(cat_id)])
                cur_txt_feat = cat_id_2_txt_feat[cat_id]
                cur_txt_sd_image = img_preprocess(Image.open(os.path.join(sd_img_folder, coco_91[int(cat_id)]+'.png')).convert('RGB')).unsqueeze(0)
                # if img_id == '289343':
                cur_txt_sd_image = cur_txt_sd_image.to(device=device, dtype=input_dtype, non_blocking=True)
                cur_txt_sd_feat = model.encode_image(cur_txt_sd_image, normalize=True)
                logit_list = []
                occ_img_id_list = cat_id_2_occ_img[cat_id]
                neg_img_id_list = cat_id_2_negative_img[cat_id]
                for img_id in occ_img_id_list:
                    cur_img_feat = img_id_2_img_feat[str(img_id)]
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0].cpu().numpy().tolist()
                    logit_list.append([cur_logit, 1, img_id])
                for img_id in neg_img_id_list:
                    cur_img_feat = img_id_2_img_feat[str(img_id)]
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0].cpu().numpy().tolist()
                    logit_list.append([cur_logit, 0, img_id])
                logit_list.sort(key=lambda x: -x[0])
                # ipdb.set_trace()
                new_image_list = [] # expected BS=100
                new_logit_list = []
                for top_i in range(min(topk, len(logit_list))):
                    # print(top_i)
                    cur_image = img_preprocess(Image.open(os.path.join(img_folder, str(logit_list[top_i][-1]).rjust(12,'0')+'.jpg')).convert('RGB'))
                    cur_image = cur_image.to(device=device, dtype=input_dtype, non_blocking=True).unsqueeze(0)
                    new_image_list.append(cur_image)
                new_image_list = torch.cat(new_image_list, dim=0) # 100 x 3 x 224 x 224
                # ipdb.set_trace()
                cur_txt_feat = cur_txt_feat + cur_txt_sd_feat
                new_image_features = model(new_image_list, None, cur_txt_feat.repeat(new_image_list.shape[0], 1))["image_features"]
                new_logit = (new_image_features @ cur_txt_feat.T)
                for top_i in range(new_logit.shape[0]):
                    new_logit_list.append([new_logit[top_i][0].cpu().numpy().tolist(), logit_list[top_i][1]])
                for top_i in range(min(topk, len(logit_list)), len(logit_list)):
                    new_logit_list.append([logit_list[top_i][0], logit_list[top_i][1]])
                # ipdb.set_trace()
                cur_ap = calculate_ap(new_logit_list)
                print(cur_ap)
                ap_sum += cur_ap
                recall_cnt += 1

            print('mAP: ', ap_sum / recall_cnt)

    return 0



# old version in October 2024
def evaluate_imagenet_a_tgvpt3_old_2024oct(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    logging.info('start evaluation')
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        logging.info('start evaluation')

        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []


        from torchvision import transforms as pth_transforms
        from PIL import Image
        occluded_coco_retrieval_ann_file = json.load(open('DATASET_PATH/imagenet-a-annfile.json'))
        cat_id_2_occ_img = occluded_coco_retrieval_ann_file['cat_id_2_occ_img']
        cat_id_2_negative_img = occluded_coco_retrieval_ann_file['cat_id_2_negative_img']
        img_id_2_all_ins_id = occluded_coco_retrieval_ann_file['img_id_2_all_ins_id']
        coco_91 = json.load(open('DATASET_PATH/imagenet-a-cat2name.json'))
        img_preprocess = pth_transforms.Compose([
            pth_transforms.Resize([224, 224], pth_transforms.InterpolationMode.BICUBIC),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])


        with torch.no_grad():

            # cat_id_2_txt_feat = {}
            # iter_txt = 0
            # for cat_id in cat_id_2_occ_img.keys():
            #     print(iter_txt)
            #     iter_txt += 1
            #     cur_txt = coco_91[int(cat_id)]
            #     text = tokenizer([cur_txt]).to(device=device, non_blocking=True)
            #     text_features = model.encode_text(text)
            #     text_features /= text_features.norm(dim=-1, keepdim=True)
            #     cat_id_2_txt_feat[cat_id] = text_features

            # img_id_2_img_feat = {}
            img_folder = 'DATASET_PATH/imagenet-a'
            # iter_img = 0
            # for img_id in img_id_2_all_ins_id.keys():
            #     # ipdb.set_trace()
            #     # print(iter_img)
            #     iter_img += 1
            #     cur_image = img_preprocess(Image.open(os.path.join(img_folder, str(img_id).rjust(12,'0')+'.jpg')).convert('RGB')).unsqueeze(0)
            #     # if img_id == '289343':
            #     cur_image = cur_image.to(device=device, dtype=input_dtype, non_blocking=True)
            #     image_features = model.encode_image(cur_image)
            #     image_features /= image_features.norm(dim=-1, keepdim=True)
            #     img_id_2_img_feat[img_id] = image_features



            # top1_recall = 0
            # top5_recall = 0
            # top10_recall = 0
            
            cur_pkl = pickle.load(open('DATASET_PATH/imagenet-a_clip_feat.pkl', 'rb'))
            cat_id_2_txt_feat = cur_pkl['cat_id_2_txt_feat']
            img_id_2_img_feat = cur_pkl['img_id_2_img_feat']
            recall_cnt = 0
            ap_sum = 0
            topk = 1000 # topK in reranking
            for cat_id in cat_id_2_txt_feat.keys():
                print(coco_91[cat_id])
                cur_txt_feat = cat_id_2_txt_feat[cat_id]
                logit_list = []
                occ_img_id_list = cat_id_2_occ_img[cat_id]
                neg_img_id_list = cat_id_2_negative_img[cat_id]
                for img_id in occ_img_id_list:
                    cur_img_feat = img_id_2_img_feat[str(img_id)]
                    # ipdb.set_trace()
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0]#.cpu().numpy().tolist()
                    logit_list.append([cur_logit, 1, img_id])
                for img_id in neg_img_id_list:
                    cur_img_feat = img_id_2_img_feat[str(img_id)]
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0]#.cpu().numpy().tolist()
                    logit_list.append([cur_logit, 0, img_id])
                logit_list.sort(key=lambda x: -x[0])
                # ipdb.set_trace()
                new_image_list = [] # expected BS=100
                new_logit_list = []
                for top_i in range(min(topk, len(logit_list))):
                    # print(top_i)
                    cur_image = img_preprocess(Image.open(os.path.join(img_folder, logit_list[top_i][-1])).convert('RGB'))
                    cur_image = cur_image.to(device=device, dtype=input_dtype, non_blocking=True).unsqueeze(0)
                    new_image_list.append(cur_image)
                new_image_list = torch.cat(new_image_list, dim=0) # 100 x 3 x 224 x 224
                # ipdb.set_trace()
                new_image_features = model(new_image_list, None, torch.from_numpy(cur_txt_feat).to(device).repeat(new_image_list.shape[0], 1))["image_features"].detach().cpu().numpy()
                new_logit = (new_image_features @ cur_txt_feat.T)
                for top_i in range(new_logit.shape[0]):
                    new_logit_list.append([new_logit[top_i][0], logit_list[top_i][1]])
                for top_i in range(min(topk, len(logit_list)), len(logit_list)):
                    new_logit_list.append([logit_list[top_i][0], logit_list[top_i][1]])
                # ipdb.set_trace()
                cur_ap = calculate_ap(new_logit_list)
                print(cur_ap)
                ap_sum += cur_ap
                recall_cnt += 1

            print('mAP: ', ap_sum / recall_cnt)

    return 0


def evaluate_imagenet_a_tgvpt3(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    logging.info('start evaluation')
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        logging.info('start evaluation')

        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []


        from torchvision import transforms as pth_transforms
        from PIL import Image
        occluded_coco_retrieval_ann_file = json.load(open('DATASET_PATH/imagenet-a-annfile_balanced_1.29_3.json'))
        cat_id_2_occ_img = occluded_coco_retrieval_ann_file['cat_id_2_occ_img']
        cat_id_2_negative_img = occluded_coco_retrieval_ann_file['cat_id_2_negative_img']
        # img_id_2_all_ins_id = occluded_coco_retrieval_ann_file['img_id_2_all_ins_id']
        coco_91 = json.load(open('DATASET_PATH/imagenet-a-cat2name.json'))
        img_preprocess = pth_transforms.Compose([
            pth_transforms.Resize([224, 224], pth_transforms.InterpolationMode.BICUBIC),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])


        with torch.no_grad():

            # ours_cat_id_2_txt_feat = {}
            # iter_txt = 0
            # for cat_id in cat_id_2_occ_img.keys():
            #     print(iter_txt)
            #     iter_txt += 1
            #     cur_txt = coco_91[cat_id]
            #     text = tokenizer([cur_txt]).to(device=device, non_blocking=True)
            #     text_features = model.encode_text(text)
            #     text_features /= text_features.norm(dim=-1, keepdim=True)
            #     ours_cat_id_2_txt_feat[cat_id] = text_features

            img_folder = 'DATASET_PATH/imagenet-a'

            cur_pkl = pickle.load(open('DATASET_PATH/imagenet-a_clip_feat.pkl', 'rb'))
            cat_id_2_txt_feat = cur_pkl['cat_id_2_txt_feat']
            img_id_2_img_feat = cur_pkl['img_id_2_img_feat']
            recall_cnt = 0
            ap_sum = 0
            topk = 100#100#3000 # topK in reranking
            all_precisions = []



            for cat_id in cat_id_2_txt_feat.keys():
                print(coco_91[cat_id])
                cur_txt_feat = cat_id_2_txt_feat[cat_id]
                # ours_cur_txt_feat = ours_cat_id_2_txt_feat[cat_id]
                logit_list = []
                occ_img_id_list = cat_id_2_occ_img[cat_id]
                neg_img_id_list = cat_id_2_negative_img[cat_id]
                for img_id in occ_img_id_list:
                    cur_img_feat = img_id_2_img_feat[str(img_id)]
                    # ipdb.set_trace()
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0]#.cpu().numpy().tolist()
                    logit_list.append([cur_logit, 1, img_id])
                for img_id in neg_img_id_list:
                    cur_img_feat = img_id_2_img_feat[str(img_id)]
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0]#.cpu().numpy().tolist()
                    logit_list.append([cur_logit, 0, img_id])
                logit_list.sort(key=lambda x: -x[0])

                # recall_logit_list = [v[1] for v in logit_list]
                # array_logit_list = np.array(recall_logit_list)
                # if len(occ_img_id_list) > 0:
                #     # ipdb.set_trace()
                #     recall_topk_initial += array_logit_list[:topk].sum() / array_logit_list.sum()
                #     recall_cnt += 1
                #     print('initial recalled within topk: ', array_logit_list[:topk].sum() / array_logit_list.sum())
                # continue

                # ipdb.set_trace()
                new_image_list = [] # expected BS=100
                new_logit_list = []
                new_vis_logit_list = []
                for top_i in range(min(topk, len(logit_list))):
                    # print(top_i)
                    cur_image = img_preprocess(Image.open(os.path.join(img_folder, logit_list[top_i][-1])).convert('RGB'))
                    cur_image = cur_image.to(device=device, dtype=input_dtype, non_blocking=True).unsqueeze(0)
                    new_image_list.append(cur_image)
                new_image_list = torch.cat(new_image_list, dim=0) # 100 x 3 x 224 x 224
                # ipdb.set_trace()
                new_image_features = model(new_image_list, None, torch.from_numpy(cur_txt_feat).to(device).repeat(new_image_list.shape[0], 1))["image_features"].detach().cpu().numpy()
                new_logit = (new_image_features @ cur_txt_feat.T)
                for top_i in range(new_logit.shape[0]):
                    new_logit_list.append([new_logit[top_i][0], logit_list[top_i][1]])
                    new_vis_logit_list.append([new_logit[top_i][0], logit_list[top_i][1], logit_list[top_i][2]])
                for top_i in range(min(topk, len(logit_list)), len(logit_list)):
                # for top_i in range(len(logit_list)):
                    new_logit_list.append([logit_list[top_i][0]-1000, logit_list[top_i][1]])
                    new_vis_logit_list.append([logit_list[top_i][0]-1000, logit_list[top_i][1], logit_list[top_i][2]])
                # ipdb.set_trace()
                # cur_ap = calculate_ap(new_logit_list)

                # # for vis 11.9 
                # new_vis_logit_list.sort(key=lambda x: -x[0])
                # # ipdb.set_trace()
                # sum_org = 0
                # sum_ours = 0
                # for top_i in range(100):
                #     sum_org += logit_list[top_i][1]
                #     sum_ours += new_vis_logit_list[top_i][1]
                # print(sum_ours - sum_org)
                # if sum_ours >= sum_org + 0:#5:
                #     # ipdb.set_trace()
                #     for top_i in range(100):
                #         print(top_i)
                #         # if os.path.exists(os.path.join(img_folder, str(logit_list[top_i][-1]).rjust(12,'0')+'.jpg')):
                #         # ipdb.set_trace()
                #         shutil.copy(os.path.join(img_folder, logit_list[top_i][-1]), os.path.join(vis_dir, 'txti_' + coco_91[cat_id] + '-before-top_' + str(top_i) + '-csvid_' + coco_91[str(logit_list[top_i][-1]).split('/')[0]] + '_' + str(logit_list[top_i][-1]).split('/')[1] + '-label-' + str(logit_list[top_i][1]) + '-score_' + str(logit_list[top_i][0]) + '.png'))
                #         # else:
                #         #     shutil.copy(os.path.join(img_folder2, str(logit_list[top_i][-1]).rjust(12,'0')+'.jpg'), os.path.join(vis_dir, 'txti_' + coco_91[int(cat_id)] + '-before-top_' + str(top_i) + '-csvid_' + str(logit_list[top_i][-1]) + '-label-' + str(logit_list[top_i][1]) + '-score_' + str(logit_list[top_i][0]) + '.png'))
                #         # if os.path.exists(os.path.join(img_folder, str(new_vis_logit_list[top_i][-1]).rjust(12,'0')+'.jpg')):   
                #         shutil.copy(os.path.join(img_folder, new_vis_logit_list[top_i][-1]), os.path.join(vis_dir, 'txti_' + coco_91[cat_id] + '-after-top_' + str(top_i) + '-csvid_' + coco_91[str(new_vis_logit_list[top_i][-1]).split('/')[0]] + '_' + str(new_vis_logit_list[top_i][-1]).split('/')[1] + '-label-' + str(new_vis_logit_list[top_i][1]) + '-score_' + str(new_vis_logit_list[top_i][0]) + '.png'))
                #         # else:
                #         #     shutil.copy(os.path.join(img_folder2, str(new_vis_logit_list[top_i][-1]).rjust(12,'0')+'.jpg'), os.path.join(vis_dir, 'txti_' + coco_91[int(cat_id)] + '-after-top_' + str(top_i) + '-csvid_' + str(new_vis_logit_list[top_i][-1]) + '-label-' + str(new_vis_logit_list[top_i][1]) + '-score_' + str(new_vis_logit_list[top_i][0]) + '.png'))
                cur_ap, cur_precisions = calculate_ap_new(new_logit_list)
                all_precisions.append(cur_precisions)
                print(cur_ap)
                ap_sum += cur_ap
                recall_cnt += 1

        # print('initial recall@topk: ', recall_topk_initial / recall_cnt)
        # print(recall_cnt)
        
        print('mAP: ', ap_sum / recall_cnt)
        ipdb.set_trace()
        from scipy.interpolate import interp1d
        updated_precisions = []
        target_length = max(len(p) for p in all_precisions)
        for precisions in all_precisions:
            if len(precisions) == 1:
                continue
            x_original = np.linspace(0, 1, len(precisions))
            x_target = np.linspace(0, 1, target_length)
            interpolation_function = interp1d(x_original, precisions, kind='linear', fill_value="extrapolate")
            interpolated_precisions = interpolation_function(x_target)
            updated_precisions.append(interpolated_precisions)
        recall_target = np.linspace(0, 1, target_length)
        avg_precisions = np.mean(updated_precisions, axis=0)
        # plot_average_precision_curve(avg_precisions, recall_target, 'occluded_coco_mAP_hard_sample.png')
        np_save_folder = 'imagenet_r_mAP_curve_10.29'
        np.save(os.path.join(np_save_folder, 'v4_ours_clip_recall_target.npy'), recall_target)
        np.save(os.path.join(np_save_folder, 'v4_ours_clip_avg_precisions.npy'), avg_precisions)
        

        ipdb.set_trace()

    return 0


def evaluate_ilias_tgvpt3(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    logging.info('start evaluation')
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        logging.info('start evaluation')

        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []


        from torchvision import transforms as pth_transforms
        from PIL import Image
        occluded_coco_retrieval_ann_file = json.load(open('DATASET_PATH/ilias-annfile.json'))
        cat_id_2_occ_img = occluded_coco_retrieval_ann_file['cat_id_2_occ_img']
        cat_id_2_negative_img = occluded_coco_retrieval_ann_file['cat_id_2_negative_img']
        img_id_2_img_pth = occluded_coco_retrieval_ann_file['img_id_2_img_pth']
        img_preprocess = pth_transforms.Compose([
            pth_transforms.Resize([224, 224], pth_transforms.InterpolationMode.BICUBIC),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])


        with torch.no_grad():

            # ours_cat_id_2_txt_feat = {}
            # iter_txt = 0
            # for cat_id in cat_id_2_occ_img.keys():
            #     print(iter_txt)
            #     iter_txt += 1
            #     cur_txt = coco_91[cat_id]
            #     text = tokenizer([cur_txt]).to(device=device, non_blocking=True)
            #     text_features = model.encode_text(text)
            #     text_features /= text_features.norm(dim=-1, keepdim=True)
            #     ours_cat_id_2_txt_feat[cat_id] = text_features

            img_folder = 'DATASET_PATH/imagenet-r'

            cur_pkl = pickle.load(open('DATASET_PATH/clip_ilias_infer_img_txt_features_5.11.pkl', 'rb'))
            cat_id_2_txt_feat = cur_pkl['text_features']
            img_id_2_img_feat = cur_pkl['image_features']
            recall_cnt = 0
            ap_sum = 0
            topk = 1000 # topK in reranking
            all_precisions = []

            # recall_topk_list = [100, 200, 500, 1000, 2000]
            # recall_topk_initial = {}
            # for topk_i in recall_topk_list:
            #     recall_topk_initial[topk_i] = 0

            neg_img_id_list = [v for v in range(4715, img_id_2_img_feat.shape[0])]
            for cat_id in range(cat_id_2_txt_feat.shape[0]):
                # print(coco_91[cat_id])
                print(cat_id)
                cur_txt_feat = cat_id_2_txt_feat[cat_id:cat_id+1]
                # ours_cur_txt_feat = ours_cat_id_2_txt_feat[cat_id]
                logit_list = []
                occ_img_id_list = cat_id_2_occ_img[str(cat_id)]
                # neg_img_id_list = cat_id_2_negative_img[str(cat_id)]
                
                for img_id in occ_img_id_list:
                    # ipdb.set_trace()
                    cur_img_feat = img_id_2_img_feat[img_id:img_id+1]
                    # ipdb.set_trace()
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0]#.cpu().numpy().tolist()
                    logit_list.append([cur_logit, 1, img_id])

                # # old implementation - slow for 5M negative images
                # for img_id in neg_img_id_list:
                #     print(img_id)
                #     cur_img_feat = img_id_2_img_feat[img_id:img_id+1]
                #     # ipdb.set_trace()
                #     cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0]#.cpu().numpy().tolist()
                #     logit_list.append([cur_logit, 0, img_id])
                # logit_list.sort(key=lambda x: -x[0])

                # dummpy implementation - for test speed
                # Assuming continuous IDs from neg_img_id_list[0] to neg_img_id_list[-1]
                start_idx = neg_img_id_list[0]
                end_idx = neg_img_id_list[-1] + 1
                total_images = end_idx - start_idx

                # Print info
                print(f"Processing all {total_images} images from {start_idx} to {end_idx-1}")

                # Create dummy values for all images at once
                zeros = torch.zeros(total_images, device=img_id_2_img_feat.device)

                print(f"1. Processing all {total_images} images from {start_idx} to {end_idx-1}")

                img_ids = torch.arange(start_idx, end_idx, device=img_id_2_img_feat.device)

                print(f"2. Processing all {total_images} images from {start_idx} to {end_idx-1}")

                # Create the entire result list in one operation
                batch_results = torch.stack([zeros, zeros, img_ids], dim=1).tolist()

                print(f"3. Processing all {total_images} images from {start_idx} to {end_idx-1}")

                logit_list.extend(batch_results)

                print(f"4. Processing all {total_images} images from {start_idx} to {end_idx-1}")

                logit_list.sort(key=lambda x: -x[0])

                print(f"5. Processing all {total_images} images from {start_idx} to {end_idx-1}")

                
                # # new implementation - speed up for loop
                # batch_size = 10000
                # for i in range(0, len(neg_img_id_list), batch_size):
                #     # Get start and end indices for this batch
                #     start_idx = neg_img_id_list[i]
                #     # Make sure we don't go beyond the list length
                #     end_idx = neg_img_id_list[min(i+batch_size-1, len(neg_img_id_list)-1)] + 1
                    
                #     # Print progress
                #     print(f"Processing batch {i//batch_size + 1}, images {start_idx} to {end_idx-1}")
                    
                #     # Get entire batch of image features directly - much faster than individual access
                #     batch_img_feat = img_id_2_img_feat[start_idx:end_idx]
                    
                #     # Perform batched matrix multiplication
                #     batch_logits = (batch_img_feat @ cur_txt_feat.T).squeeze(1)  # Shape: [batch_size]
                    
                #     # Create tensors for the 0 value and image IDs
                #     zeros = torch.zeros_like(batch_logits)
                #     img_ids = torch.arange(start_idx, end_idx, device=batch_logits.device)
                    
                #     # Stack the tensors and convert to list - no Python loop needed
                #     batch_results = torch.stack([batch_logits, zeros, img_ids], dim=1).tolist()
                #     logit_list.extend(batch_results)

                # array_logit_list = np.array(logit_list)[:,1].astype(np.float32)
                # if len(occ_img_id_list) > 0:
                #     for key_i in recall_topk_initial.keys():
                #         # ipdb.set_trace()
                #         recall_topk_initial[key_i] += array_logit_list[:int(key_i)].sum() / array_logit_list.sum()
                #     recall_cnt += 1
                #     print('initial recalled within topk: ', array_logit_list[:topk].sum() / array_logit_list.sum())
                # continue

                # ipdb.set_trace()
                new_image_list = [] # expected BS=100
                new_logit_list = []
                new_vis_logit_list = []
                for top_i in range(min(topk, len(logit_list))):
                    print(top_i)
                    cur_image = img_preprocess(Image.open(os.path.join(img_id_2_img_pth[str(int(logit_list[top_i][-1]))])).convert('RGB'))
                    cur_image = cur_image.to(device=device, dtype=input_dtype, non_blocking=True).unsqueeze(0)
                    new_image_list.append(cur_image)
                new_image_list = torch.cat(new_image_list, dim=0) # 100 x 3 x 224 x 224
                # ipdb.set_trace()
                new_image_features = model(new_image_list, None, cur_txt_feat.to(device).repeat(new_image_list.shape[0], 1))["image_features"].detach().cpu().numpy()
                new_logit = (new_image_features @ cur_txt_feat.T)
                for top_i in range(new_logit.shape[0]):
                    new_logit_list.append([new_logit[top_i][0], logit_list[top_i][1]])
                    new_vis_logit_list.append([new_logit[top_i][0], logit_list[top_i][1], logit_list[top_i][2]])
                # for top_i in range(min(topk, len(logit_list)), len(logit_list)):
                # # for top_i in range(len(logit_list)):
                #     new_logit_list.append([logit_list[top_i][0]-1000, logit_list[top_i][1]])
                #     new_vis_logit_list.append([logit_list[top_i][0]-1000, logit_list[top_i][1], logit_list[top_i][2]])
                # ipdb.set_trace()
                # cur_ap = calculate_ap(new_logit_list)

                # # for vis 11.9 
                # new_vis_logit_list.sort(key=lambda x: -x[0])
                # # ipdb.set_trace()
                # sum_org = 0
                # sum_ours = 0
                # for top_i in range(100):
                #     sum_org += logit_list[top_i][1]
                #     sum_ours += new_vis_logit_list[top_i][1]
                # print(sum_ours - sum_org)
                # if sum_ours >= sum_org + 5:
                #     # ipdb.set_trace()
                #     for top_i in range(100):
                #         print(top_i)
                #         # if os.path.exists(os.path.join(img_folder, str(logit_list[top_i][-1]).rjust(12,'0')+'.jpg')):
                #         shutil.copy(os.path.join(img_folder, logit_list[top_i][-1]), os.path.join(vis_dir, 'txti_' + coco_91[cat_id] + '-before-top_' + str(top_i) + '-csvid_' + str(logit_list[top_i][-1]).split('/')[0] + '_' + str(logit_list[top_i][-1]).split('/')[1] + '-label-' + str(logit_list[top_i][1]) + '-score_' + str(logit_list[top_i][0]) + '.png'))
                #         # else:
                #         #     shutil.copy(os.path.join(img_folder2, str(logit_list[top_i][-1]).rjust(12,'0')+'.jpg'), os.path.join(vis_dir, 'txti_' + coco_91[int(cat_id)] + '-before-top_' + str(top_i) + '-csvid_' + str(logit_list[top_i][-1]) + '-label-' + str(logit_list[top_i][1]) + '-score_' + str(logit_list[top_i][0]) + '.png'))
                #         # if os.path.exists(os.path.join(img_folder, str(new_vis_logit_list[top_i][-1]).rjust(12,'0')+'.jpg')):   
                #         shutil.copy(os.path.join(img_folder, new_vis_logit_list[top_i][-1]), os.path.join(vis_dir, 'txti_' + coco_91[cat_id] + '-after-top_' + str(top_i) + '-csvid_' + str(new_vis_logit_list[top_i][-1]).split('/')[0] + '_' + str(new_vis_logit_list[top_i][-1]).split('/')[1] + '-label-' + str(new_vis_logit_list[top_i][1]) + '-score_' + str(new_vis_logit_list[top_i][0]) + '.png'))
                #         # else:
                #         #     shutil.copy(os.path.join(img_folder2, str(new_vis_logit_list[top_i][-1]).rjust(12,'0')+'.jpg'), os.path.join(vis_dir, 'txti_' + coco_91[int(cat_id)] + '-after-top_' + str(top_i) + '-csvid_' + str(new_vis_logit_list[top_i][-1]) + '-label-' + str(new_vis_logit_list[top_i][1]) + '-score_' + str(new_vis_logit_list[top_i][0]) + '.png'))
                cur_ap, cur_precisions = calculate_ap_new(new_logit_list)
                all_precisions.append(cur_precisions)
                print(cur_ap)
                ap_sum += cur_ap
                recall_cnt += 1

        # for key_i in recall_topk_initial.keys():
        #     print('initial recall@top' + str(int(key_i)) + ': ', recall_topk_initial[key_i] / recall_cnt)
        # print(recall_cnt)

        
        print('mAP: ', ap_sum / recall_cnt)
        ipdb.set_trace()
        from scipy.interpolate import interp1d
        updated_precisions = []
        target_length = max(len(p) for p in all_precisions)
        for precisions in all_precisions:
            if len(precisions) == 1:
                continue
            x_original = np.linspace(0, 1, len(precisions))
            x_target = np.linspace(0, 1, target_length)
            interpolation_function = interp1d(x_original, precisions, kind='linear', fill_value="extrapolate")
            interpolated_precisions = interpolation_function(x_target)
            updated_precisions.append(interpolated_precisions)
        recall_target = np.linspace(0, 1, target_length)
        avg_precisions = np.mean(updated_precisions, axis=0)
        # plot_average_precision_curve(avg_precisions, recall_target, 'occluded_coco_mAP_hard_sample.png')
        np_save_folder = 'imagenet_r_mAP_curve_10.29'
        np.save(os.path.join(np_save_folder, 'iccv_3.3_ours_clip_recall_target.npy'), recall_target)
        np.save(os.path.join(np_save_folder, 'iccv_3.3_ours_clip_avg_precisions.npy'), avg_precisions)
        

        ipdb.set_trace()

    return 0



def evaluate_imagenet_r_tgvpt3(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    logging.info('start evaluation')
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        logging.info('start evaluation')

        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []


        from torchvision import transforms as pth_transforms
        from PIL import Image
        occluded_coco_retrieval_ann_file = json.load(open('standard_benchmarks/imagenet_r/imagenet-r-annfile.json'))
        cat_id_2_occ_img = occluded_coco_retrieval_ann_file['cat_id_2_occ_img']
        cat_id_2_negative_img = occluded_coco_retrieval_ann_file['cat_id_2_negative_img']
        img_id_2_all_ins_id = occluded_coco_retrieval_ann_file['img_id_2_all_ins_id']
        coco_91 = json.load(open('standard_benchmarks/imagenet_r/imagenet-r-cat2name.json'))
        img_preprocess = pth_transforms.Compose([
            pth_transforms.Resize([224, 224], pth_transforms.InterpolationMode.BICUBIC),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])


        with torch.no_grad():

            # ours_cat_id_2_txt_feat = {}
            # iter_txt = 0
            # for cat_id in cat_id_2_occ_img.keys():
            #     print(iter_txt)
            #     iter_txt += 1
            #     cur_txt = coco_91[cat_id]
            #     text = tokenizer([cur_txt]).to(device=device, non_blocking=True)
            #     text_features = model.encode_text(text)
            #     text_features /= text_features.norm(dim=-1, keepdim=True)
            #     ours_cat_id_2_txt_feat[cat_id] = text_features

            img_folder = '/disk1/work/ypliu/imagenet-r'

            cur_pkl = pickle.load(open('standard_benchmarks/imagenet_r/imagenet-r_clip_feat.pkl', 'rb'))
            cat_id_2_txt_feat = cur_pkl['cat_id_2_txt_feat']
            img_id_2_img_feat = cur_pkl['img_id_2_img_feat']
            recall_cnt = 0
            ap_sum = 0
            topk = 1000 # topK in reranking
            all_precisions = []

            # recall_topk_list = [100, 200, 500, 1000, 2000]
            # recall_topk_initial = {}
            # for topk_i in recall_topk_list:
            #     recall_topk_initial[topk_i] = 0

            for cat_id in cat_id_2_txt_feat.keys():
                print(coco_91[cat_id])
                cur_txt_feat = cat_id_2_txt_feat[cat_id]
                # ours_cur_txt_feat = ours_cat_id_2_txt_feat[cat_id]
                logit_list = []
                occ_img_id_list = cat_id_2_occ_img[cat_id]
                neg_img_id_list = cat_id_2_negative_img[cat_id]
                for img_id in occ_img_id_list:
                    cur_img_feat = img_id_2_img_feat[str(img_id)]
                    # ipdb.set_trace()
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0]#.cpu().numpy().tolist()
                    logit_list.append([cur_logit, 1, img_id])
                for img_id in neg_img_id_list:
                    cur_img_feat = img_id_2_img_feat[str(img_id)]
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0]#.cpu().numpy().tolist()
                    logit_list.append([cur_logit, 0, img_id])
                logit_list.sort(key=lambda x: -x[0])

                # array_logit_list = np.array(logit_list)[:,1].astype(np.float32)
                # if len(occ_img_id_list) > 0:
                #     for key_i in recall_topk_initial.keys():
                #         # ipdb.set_trace()
                #         recall_topk_initial[key_i] += array_logit_list[:int(key_i)].sum() / array_logit_list.sum()
                #     recall_cnt += 1
                #     print('initial recalled within topk: ', array_logit_list[:topk].sum() / array_logit_list.sum())
                # continue

                # ipdb.set_trace()
                new_image_list = [] # expected BS=100
                new_logit_list = []
                new_vis_logit_list = []
                for top_i in range(min(topk, len(logit_list))):
                    # print(top_i)
                    cur_image = img_preprocess(Image.open(os.path.join(img_folder, logit_list[top_i][-1])).convert('RGB'))
                    cur_image = cur_image.to(device=device, dtype=input_dtype, non_blocking=True).unsqueeze(0)
                    new_image_list.append(cur_image)
                new_image_list = torch.cat(new_image_list, dim=0) # 100 x 3 x 224 x 224
                # ipdb.set_trace()
                new_image_features = model(new_image_list, None, torch.from_numpy(cur_txt_feat).to(device).repeat(new_image_list.shape[0], 1))["image_features"].detach().cpu().numpy()
                new_logit = (new_image_features @ cur_txt_feat.T)
                for top_i in range(new_logit.shape[0]):
                    new_logit_list.append([new_logit[top_i][0], logit_list[top_i][1]])
                    new_vis_logit_list.append([new_logit[top_i][0], logit_list[top_i][1], logit_list[top_i][2]])
                for top_i in range(min(topk, len(logit_list)), len(logit_list)):
                # for top_i in range(len(logit_list)):
                    new_logit_list.append([logit_list[top_i][0]-1000, logit_list[top_i][1]])
                    new_vis_logit_list.append([logit_list[top_i][0]-1000, logit_list[top_i][1], logit_list[top_i][2]])
                # ipdb.set_trace()
                # cur_ap = calculate_ap(new_logit_list)

                # # for vis 11.9 
                # new_vis_logit_list.sort(key=lambda x: -x[0])
                # # ipdb.set_trace()
                # sum_org = 0
                # sum_ours = 0
                # for top_i in range(100):
                #     sum_org += logit_list[top_i][1]
                #     sum_ours += new_vis_logit_list[top_i][1]
                # print(sum_ours - sum_org)
                # if sum_ours >= sum_org + 5:
                #     # ipdb.set_trace()
                #     for top_i in range(100):
                #         print(top_i)
                #         # if os.path.exists(os.path.join(img_folder, str(logit_list[top_i][-1]).rjust(12,'0')+'.jpg')):
                #         shutil.copy(os.path.join(img_folder, logit_list[top_i][-1]), os.path.join(vis_dir, 'txti_' + coco_91[cat_id] + '-before-top_' + str(top_i) + '-csvid_' + str(logit_list[top_i][-1]).split('/')[0] + '_' + str(logit_list[top_i][-1]).split('/')[1] + '-label-' + str(logit_list[top_i][1]) + '-score_' + str(logit_list[top_i][0]) + '.png'))
                #         # else:
                #         #     shutil.copy(os.path.join(img_folder2, str(logit_list[top_i][-1]).rjust(12,'0')+'.jpg'), os.path.join(vis_dir, 'txti_' + coco_91[int(cat_id)] + '-before-top_' + str(top_i) + '-csvid_' + str(logit_list[top_i][-1]) + '-label-' + str(logit_list[top_i][1]) + '-score_' + str(logit_list[top_i][0]) + '.png'))
                #         # if os.path.exists(os.path.join(img_folder, str(new_vis_logit_list[top_i][-1]).rjust(12,'0')+'.jpg')):   
                #         shutil.copy(os.path.join(img_folder, new_vis_logit_list[top_i][-1]), os.path.join(vis_dir, 'txti_' + coco_91[cat_id] + '-after-top_' + str(top_i) + '-csvid_' + str(new_vis_logit_list[top_i][-1]).split('/')[0] + '_' + str(new_vis_logit_list[top_i][-1]).split('/')[1] + '-label-' + str(new_vis_logit_list[top_i][1]) + '-score_' + str(new_vis_logit_list[top_i][0]) + '.png'))
                #         # else:
                #         #     shutil.copy(os.path.join(img_folder2, str(new_vis_logit_list[top_i][-1]).rjust(12,'0')+'.jpg'), os.path.join(vis_dir, 'txti_' + coco_91[int(cat_id)] + '-after-top_' + str(top_i) + '-csvid_' + str(new_vis_logit_list[top_i][-1]) + '-label-' + str(new_vis_logit_list[top_i][1]) + '-score_' + str(new_vis_logit_list[top_i][0]) + '.png'))
                cur_ap, cur_precisions = calculate_ap_new(new_logit_list)
                all_precisions.append(cur_precisions)
                print(cur_ap)
                ap_sum += cur_ap
                recall_cnt += 1

        # for key_i in recall_topk_initial.keys():
        #     print('initial recall@top' + str(int(key_i)) + ': ', recall_topk_initial[key_i] / recall_cnt)
        # print(recall_cnt)

        
        print('mAP: ', ap_sum / recall_cnt)
        ipdb.set_trace()
        from scipy.interpolate import interp1d
        updated_precisions = []
        target_length = max(len(p) for p in all_precisions)
        for precisions in all_precisions:
            if len(precisions) == 1:
                continue
            x_original = np.linspace(0, 1, len(precisions))
            x_target = np.linspace(0, 1, target_length)
            interpolation_function = interp1d(x_original, precisions, kind='linear', fill_value="extrapolate")
            interpolated_precisions = interpolation_function(x_target)
            updated_precisions.append(interpolated_precisions)
        recall_target = np.linspace(0, 1, target_length)
        avg_precisions = np.mean(updated_precisions, axis=0)
        # plot_average_precision_curve(avg_precisions, recall_target, 'occluded_coco_mAP_hard_sample.png')
        np_save_folder = 'imagenet_r_mAP_curve_10.29'
        np.save(os.path.join(np_save_folder, 'iccv_3.3_ours_clip_recall_target.npy'), recall_target)
        np.save(os.path.join(np_save_folder, 'iccv_3.3_ours_clip_avg_precisions.npy'), avg_precisions)
        

        ipdb.set_trace()

    return 0



def evaluate_imagenet_r_tgvpt3_siglip(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    logging.info('start evaluation')
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        logging.info('start evaluation')

        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []


        from torchvision import transforms as pth_transforms
        from PIL import Image
        occluded_coco_retrieval_ann_file = json.load(open('standard_benchmarks/imagenet_r/imagenet-r-annfile.json'))
        cat_id_2_occ_img = occluded_coco_retrieval_ann_file['cat_id_2_occ_img']
        cat_id_2_negative_img = occluded_coco_retrieval_ann_file['cat_id_2_negative_img']
        img_id_2_all_ins_id = occluded_coco_retrieval_ann_file['img_id_2_all_ins_id']
        coco_91 = json.load(open('standard_benchmarks/imagenet_r/imagenet-r-cat2name.json'))
        img_preprocess = pth_transforms.Compose([
            # pth_transforms.Resize([224, 224], pth_transforms.InterpolationMode.BICUBIC),
            pth_transforms.Resize([384, 384], pth_transforms.InterpolationMode.BICUBIC), # for resolution 384
            pth_transforms.ToTensor(),
            # pth_transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # For SigLIP
        ])


        with torch.no_grad():

            # ours_cat_id_2_txt_feat = {}
            # iter_txt = 0
            # for cat_id in cat_id_2_occ_img.keys():
            #     print(iter_txt)
            #     iter_txt += 1
            #     cur_txt = coco_91[cat_id]
            #     text = tokenizer([cur_txt]).to(device=device, non_blocking=True)
            #     text_features = model.encode_text(text)
            #     text_features /= text_features.norm(dim=-1, keepdim=True)
            #     ours_cat_id_2_txt_feat[cat_id] = text_features

            img_folder = '/disk1/work/ypliu/imagenet-r'

            cur_pkl = pickle.load(open('standard_benchmarks/imagenet_r/imagenet-r_siglipSO_feat.pkl', 'rb'))
            # cur_pkl = pickle.load(open('standard_benchmarks/imagenet_r/imagenet-r_siglip2G_feat.pkl', 'rb'))
            cat_id_2_txt_feat = cur_pkl['cat_id_2_txt_feat']
            img_id_2_img_feat = cur_pkl['img_id_2_img_feat']
            recall_cnt = 0
            ap_sum = 0
            topk = 200 # 1000 # topK in reranking
            all_precisions = []

            # recall_topk_list = [50, 100, 200, 500, 1000]
            # recall_topk_initial = {}
            # for topk_i in recall_topk_list:
            #     recall_topk_initial[topk_i] = 0

            for cat_id in cat_id_2_txt_feat.keys():
                print(coco_91[cat_id])
                cur_txt_feat = cat_id_2_txt_feat[cat_id]
                # ours_cur_txt_feat = ours_cat_id_2_txt_feat[cat_id]
                logit_list = []
                occ_img_id_list = cat_id_2_occ_img[cat_id]
                neg_img_id_list = cat_id_2_negative_img[cat_id]
                for img_id in occ_img_id_list:
                    cur_img_feat = img_id_2_img_feat[str(img_id)]
                    # ipdb.set_trace()
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0]#.cpu().numpy().tolist()
                    logit_list.append([cur_logit, 1, img_id])
                for img_id in neg_img_id_list:
                    cur_img_feat = img_id_2_img_feat[str(img_id)]
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0]#.cpu().numpy().tolist()
                    logit_list.append([cur_logit, 0, img_id])
                logit_list.sort(key=lambda x: -x[0])

                # array_logit_list = np.array(logit_list)[:,1].astype(np.float32)
                # if len(occ_img_id_list) > 0:
                #     for key_i in recall_topk_initial.keys():
                #         # ipdb.set_trace()
                #         recall_topk_initial[key_i] += array_logit_list[:int(key_i)].sum() / array_logit_list.sum()
                #     recall_cnt += 1
                #     print('initial recalled within topk: ', array_logit_list[:topk].sum() / array_logit_list.sum())
                # continue

                # ipdb.set_trace()
                new_image_list = [] # expected BS=100
                new_logit_list = []
                new_vis_logit_list = []
                for top_i in range(min(topk, len(logit_list))):
                    # print(top_i)
                    cur_image = img_preprocess(Image.open(os.path.join(img_folder, logit_list[top_i][-1])).convert('RGB'))
                    cur_image = cur_image.to(device=device, dtype=input_dtype, non_blocking=True).unsqueeze(0)
                    new_image_list.append(cur_image)
                new_image_list = torch.cat(new_image_list, dim=0) # 100 x 3 x 224 x 224
                # ipdb.set_trace()
                # new_image_features = model(new_image_list, None, torch.from_numpy(cur_txt_feat).to(device).repeat(new_image_list.shape[0], 1))["image_features"].detach().cpu().numpy()

                # -------------------------------- run the model via batches --------------------------------
                bs = 32  # choose a batch size that fits your GPU
                txt_feat = torch.from_numpy(cur_txt_feat).to(device)  # shape: (D,)
                N = new_image_list.shape[0]

                outs = []
                for i in range(0, N, bs):
                    imgs = new_image_list[i:i + bs].to(device)  # (b, ...)
                    out = model(imgs, None, txt_feat.repeat(imgs.shape[0], 1))["image_features"]  # (b, F)
                    outs.append(out.detach().cpu())

                new_image_features = torch.cat(outs, dim=0).numpy()
                # -------------------------------- run the model via batches --------------------------------

                new_logit = (new_image_features @ cur_txt_feat.T)
                for top_i in range(new_logit.shape[0]):
                    new_logit_list.append([new_logit[top_i][0], logit_list[top_i][1]])
                    new_vis_logit_list.append([new_logit[top_i][0], logit_list[top_i][1], logit_list[top_i][2]])
                for top_i in range(min(topk, len(logit_list)), len(logit_list)):
                # for top_i in range(len(logit_list)):
                    new_logit_list.append([logit_list[top_i][0]-1000, logit_list[top_i][1]])
                    new_vis_logit_list.append([logit_list[top_i][0]-1000, logit_list[top_i][1], logit_list[top_i][2]])
                # ipdb.set_trace()
                # cur_ap = calculate_ap(new_logit_list)

                # # for vis 11.9 
                # new_vis_logit_list.sort(key=lambda x: -x[0])
                # # ipdb.set_trace()
                # sum_org = 0
                # sum_ours = 0
                # for top_i in range(100):
                #     sum_org += logit_list[top_i][1]
                #     sum_ours += new_vis_logit_list[top_i][1]
                # print(sum_ours - sum_org)
                # # if sum_ours >= sum_org + 5:
                # if 1 > 0:
                #     # ipdb.set_trace()
                #     for top_i in range(100):
                #         print(top_i)
                #         # if os.path.exists(os.path.join(img_folder, str(logit_list[top_i][-1]).rjust(12,'0')+'.jpg')):
                #         shutil.copy(os.path.join(img_folder, logit_list[top_i][-1]), os.path.join(vis_dir, 'txti_' + coco_91[cat_id] + '-before-top_' + str(top_i) + '-csvid_' + str(logit_list[top_i][-1]).split('/')[0] + '_' + str(logit_list[top_i][-1]).split('/')[1] + '-label-' + str(logit_list[top_i][1]) + '-score_' + str(logit_list[top_i][0]) + '.png'))
                #         # else:
                #         #     shutil.copy(os.path.join(img_folder2, str(logit_list[top_i][-1]).rjust(12,'0')+'.jpg'), os.path.join(vis_dir, 'txti_' + coco_91[int(cat_id)] + '-before-top_' + str(top_i) + '-csvid_' + str(logit_list[top_i][-1]) + '-label-' + str(logit_list[top_i][1]) + '-score_' + str(logit_list[top_i][0]) + '.png'))
                #         # if os.path.exists(os.path.join(img_folder, str(new_vis_logit_list[top_i][-1]).rjust(12,'0')+'.jpg')):   
                #         shutil.copy(os.path.join(img_folder, new_vis_logit_list[top_i][-1]), os.path.join(vis_dir, 'txti_' + coco_91[cat_id] + '-after-top_' + str(top_i) + '-csvid_' + str(new_vis_logit_list[top_i][-1]).split('/')[0] + '_' + str(new_vis_logit_list[top_i][-1]).split('/')[1] + '-label-' + str(new_vis_logit_list[top_i][1]) + '-score_' + str(new_vis_logit_list[top_i][0]) + '.png'))
                #         # else:
                #         #     shutil.copy(os.path.join(img_folder2, str(new_vis_logit_list[top_i][-1]).rjust(12,'0')+'.jpg'), os.path.join(vis_dir, 'txti_' + coco_91[int(cat_id)] + '-after-top_' + str(top_i) + '-csvid_' + str(new_vis_logit_list[top_i][-1]) + '-label-' + str(new_vis_logit_list[top_i][1]) + '-score_' + str(new_vis_logit_list[top_i][0]) + '.png'))
                cur_ap, cur_precisions = calculate_ap_new(new_logit_list)
                all_precisions.append(cur_precisions)
                print(cur_ap)
                ap_sum += cur_ap
                recall_cnt += 1

        # for key_i in recall_topk_initial.keys():
        #     print('initial recall@top' + str(int(key_i)) + ': ', recall_topk_initial[key_i] / recall_cnt)
        # print(recall_cnt)

        
        print('mAP: ', ap_sum / recall_cnt)
        ipdb.set_trace()
        from scipy.interpolate import interp1d
        updated_precisions = []
        target_length = max(len(p) for p in all_precisions)
        for precisions in all_precisions:
            if len(precisions) == 1:
                continue
            x_original = np.linspace(0, 1, len(precisions))
            x_target = np.linspace(0, 1, target_length)
            interpolation_function = interp1d(x_original, precisions, kind='linear', fill_value="extrapolate")
            interpolated_precisions = interpolation_function(x_target)
            updated_precisions.append(interpolated_precisions)
        recall_target = np.linspace(0, 1, target_length)
        avg_precisions = np.mean(updated_precisions, axis=0)
        # plot_average_precision_curve(avg_precisions, recall_target, 'occluded_coco_mAP_hard_sample.png')
        np_save_folder = 'imagenet_r_mAP_curve_10.29'
        np.save(os.path.join(np_save_folder, 'v4_ours_clip_recall_target.npy'), recall_target)
        np.save(os.path.join(np_save_folder, 'v4_ours_clip_avg_precisions.npy'), avg_precisions)
        

        ipdb.set_trace()

    return 0



def evaluate_imagenet_r_tgvpt3_logo_prompt(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    logging.info('start evaluation')
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        logging.info('start evaluation')

        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []


        from torchvision import transforms as pth_transforms
        from PIL import Image
        from PIL import ImageDraw,ImageFont
        import random
        occluded_coco_retrieval_ann_file = json.load(open('DATASET_PATH/imagenet-r-annfile.json'))
        cat_id_2_occ_img = occluded_coco_retrieval_ann_file['cat_id_2_occ_img']
        cat_id_2_negative_img = occluded_coco_retrieval_ann_file['cat_id_2_negative_img']
        img_id_2_all_ins_id = occluded_coco_retrieval_ann_file['img_id_2_all_ins_id']
        coco_91 = json.load(open('DATASET_PATH/imagenet-r-cat2name.json'))
        img_preprocess = pth_transforms.Compose([
            pth_transforms.Resize([224, 224], pth_transforms.InterpolationMode.BICUBIC),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        logo_prompt_transform = pth_transforms.Compose([
            pth_transforms.Resize((16, 192), interpolation=pth_transforms.InterpolationMode.BICUBIC),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711])
        ])
        character_font = ImageFont.truetype(r'PROJECT_PATH/Arial.ttf', size=16)


        with torch.no_grad():

            img_folder = 'DATASET_PATH/imagenet-r'

            cur_pkl = pickle.load(open('DATASET_PATH/imagenet-r_clip_feat.pkl', 'rb'))
            cat_id_2_txt_feat = cur_pkl['cat_id_2_txt_feat']
            img_id_2_img_feat = cur_pkl['img_id_2_img_feat']
            recall_cnt = 0
            ap_sum = 0
            topk = 1000 # topK in reranking
            for cat_id in cat_id_2_txt_feat.keys():
                # generate logo prompt
                size = character_font.getsize(text=(coco_91[cat_id]))
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                im = Image.new("RGB", size, (0, 0, 0))
                draw = ImageDraw.Draw(im)
                draw.text((0, 0), str(coco_91[cat_id]), fill=(255 - r, 255 - g, 255 - b), font=character_font, align="right")
                logo_prompt_im = logo_prompt_transform(im).to(device)

                print(coco_91[cat_id])
                cur_txt_feat = cat_id_2_txt_feat[cat_id]
                logit_list = []
                occ_img_id_list = cat_id_2_occ_img[cat_id]
                neg_img_id_list = cat_id_2_negative_img[cat_id]
                for img_id in occ_img_id_list:
                    cur_img_feat = img_id_2_img_feat[str(img_id)]
                    # ipdb.set_trace()
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0]#.cpu().numpy().tolist()
                    logit_list.append([cur_logit, 1, img_id])
                for img_id in neg_img_id_list:
                    cur_img_feat = img_id_2_img_feat[str(img_id)]
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0]#.cpu().numpy().tolist()
                    logit_list.append([cur_logit, 0, img_id])
                logit_list.sort(key=lambda x: -x[0])
                # ipdb.set_trace()
                new_image_list = [] # expected BS=100
                new_logit_list = []
                for top_i in range(min(topk, len(logit_list))):
                    # print(top_i)
                    cur_image = img_preprocess(Image.open(os.path.join(img_folder, logit_list[top_i][-1])).convert('RGB'))
                    cur_image = cur_image.to(device=device, dtype=input_dtype, non_blocking=True).unsqueeze(0)
                    
                    # if args.logo_prompt:
                    x_ = 0
                    y_ = 0
                    prompt = torch.zeros(1, 3, 224, 224).cuda()
                    prompt[:, :, x_:x_ + 16, y_:y_ + 192] = logo_prompt_im.unsqueeze(0)
                    cur_image = cur_image + prompt
                    
                    new_image_list.append(cur_image)
                new_image_list = torch.cat(new_image_list, dim=0) # 100 x 3 x 224 x 224
                # ipdb.set_trace()
                new_image_features = model(new_image_list, None, torch.from_numpy(cur_txt_feat).to(device).repeat(new_image_list.shape[0], 1))["image_features"].detach().cpu().numpy()
                new_logit = (new_image_features @ cur_txt_feat.T)
                for top_i in range(new_logit.shape[0]):
                    new_logit_list.append([new_logit[top_i][0], logit_list[top_i][1]])
                for top_i in range(min(topk, len(logit_list)), len(logit_list)):
                # for top_i in range(len(logit_list)):
                    new_logit_list.append([logit_list[top_i][0], logit_list[top_i][1]])
                # ipdb.set_trace()
                cur_ap = calculate_ap(new_logit_list)
                print(cur_ap)
                ap_sum += cur_ap
                recall_cnt += 1

            print('mAP: ', ap_sum / recall_cnt)

    return 0



def evaluate_imagenet_r_tgvpt3_sd(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    logging.info('start evaluation')
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        logging.info('start evaluation')

        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []


        from torchvision import transforms as pth_transforms
        from PIL import Image
        occluded_coco_retrieval_ann_file = json.load(open('DATASET_PATH/imagenet-r-annfile.json'))
        cat_id_2_occ_img = occluded_coco_retrieval_ann_file['cat_id_2_occ_img']
        cat_id_2_negative_img = occluded_coco_retrieval_ann_file['cat_id_2_negative_img']
        img_id_2_all_ins_id = occluded_coco_retrieval_ann_file['img_id_2_all_ins_id']
        coco_91 = json.load(open('DATASET_PATH/imagenet-r-cat2name.json'))
        img_preprocess = pth_transforms.Compose([
            pth_transforms.Resize([224, 224], pth_transforms.InterpolationMode.BICUBIC),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])


        with torch.no_grad():

            img_folder = 'DATASET_PATH/imagenet-r'

            cur_pkl = pickle.load(open('DATASET_PATH/imagenet-r_clip_feat.pkl', 'rb'))
            cat_id_2_txt_feat = cur_pkl['cat_id_2_txt_feat']
            img_id_2_img_feat = cur_pkl['img_id_2_img_feat']
            recall_cnt = 0
            ap_sum = 0
            topk = 1000 # topK in reranking
            sd_img_folder = 'DATASET_PATH/coco_sd_generated_imgs_9.22_4'
            img_preprocess = pth_transforms.Compose([
                pth_transforms.Resize([224, 224], pth_transforms.InterpolationMode.BICUBIC),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
            for cat_id in cat_id_2_txt_feat.keys():
                print(coco_91[cat_id])
                cur_txt_feat = cat_id_2_txt_feat[cat_id]
                cur_txt_sd_image = img_preprocess(Image.open(os.path.join(sd_img_folder, coco_91[cat_id]+'.png')).convert('RGB')).unsqueeze(0)
                # if img_id == '289343':
                cur_txt_sd_image = cur_txt_sd_image.to(device=device, dtype=input_dtype, non_blocking=True)
                cur_txt_sd_feat = model.encode_image(cur_txt_sd_image, normalize=True)
                logit_list = []
                occ_img_id_list = cat_id_2_occ_img[cat_id]
                neg_img_id_list = cat_id_2_negative_img[cat_id]
                for img_id in occ_img_id_list:
                    cur_img_feat = img_id_2_img_feat[str(img_id)]
                    # ipdb.set_trace()
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0]#.cpu().numpy().tolist()
                    logit_list.append([cur_logit, 1, img_id])
                for img_id in neg_img_id_list:
                    cur_img_feat = img_id_2_img_feat[str(img_id)]
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0]#.cpu().numpy().tolist()
                    logit_list.append([cur_logit, 0, img_id])
                logit_list.sort(key=lambda x: -x[0])
                # ipdb.set_trace()
                new_image_list = [] # expected BS=100
                new_logit_list = []
                for top_i in range(min(topk, len(logit_list))):
                    # print(top_i)
                    cur_image = img_preprocess(Image.open(os.path.join(img_folder, logit_list[top_i][-1])).convert('RGB'))
                    cur_image = cur_image.to(device=device, dtype=input_dtype, non_blocking=True).unsqueeze(0)
                    new_image_list.append(cur_image)
                new_image_list = torch.cat(new_image_list, dim=0) # 100 x 3 x 224 x 224
                # ipdb.set_trace()
                cur_txt_feat = (torch.from_numpy(cur_txt_feat) + cur_txt_sd_feat.detach().cpu()).numpy()

                new_image_features = model(new_image_list, None, torch.from_numpy(cur_txt_feat).to(device).repeat(new_image_list.shape[0], 1))["image_features"].detach().cpu().numpy()
                new_logit = (new_image_features @ cur_txt_feat.T)
                for top_i in range(new_logit.shape[0]):
                    new_logit_list.append([new_logit[top_i][0], logit_list[top_i][1]])
                for top_i in range(min(topk, len(logit_list)), len(logit_list)):
                # for top_i in range(len(logit_list)):
                    new_logit_list.append([logit_list[top_i][0], logit_list[top_i][1]])
                # ipdb.set_trace()
                cur_ap = calculate_ap(new_logit_list)
                print(cur_ap)
                ap_sum += cur_ap
                recall_cnt += 1

            print('mAP: ', ap_sum / recall_cnt)

    return 0




def evaluate_occluded_coco_tgvpt(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    logging.info('start evaluation')
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        logging.info('start evaluation')

        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []


        from torchvision import transforms as pth_transforms
        from PIL import Image
        occluded_coco_retrieval_ann_file = json.load(open('DATASET_PATH/coco2017val_cat_id_2_occ_img_negative_img.json'))
        cat_id_2_occ_img = occluded_coco_retrieval_ann_file['cat_id_2_occ_img']
        cat_id_2_negative_img = occluded_coco_retrieval_ann_file['cat_id_2_negative_img']
        img_id_2_all_ins_id = occluded_coco_retrieval_ann_file['img_id_2_all_ins_id']
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
        img_preprocess = pth_transforms.Compose([
            pth_transforms.Resize([224, 224], pth_transforms.InterpolationMode.BICUBIC),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])


        with torch.no_grad():

            cat_id_2_txt_feat = {}
            iter_txt = 0
            all_text_features = []
            for cat_id in cat_id_2_occ_img.keys():
                print(iter_txt)
                iter_txt += 1
                cur_txt = coco_91[int(cat_id)]
                text = tokenizer([cur_txt]).to(device=device, non_blocking=True)
                text_features = model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                cat_id_2_txt_feat[cat_id] = text_features
                all_text_features.append(text_features)
            all_text_features = torch.concat(all_text_features, dim=0)

            img_id_2_img_feat = {}
            img_folder = '/scratch/shared/beegfs/shared-datasets/COCO/COCO2017/val2017/'
            iter_img = 0
            for img_id in tqdm(img_id_2_all_ins_id.keys()):
                # ipdb.set_trace()
                # print(iter_img)
                iter_img += 1
                cur_image = img_preprocess(Image.open(os.path.join(img_folder, str(img_id).rjust(12,'0')+'.jpg')).convert('RGB')).unsqueeze(0)
                # if img_id == '289343':
                cur_image = cur_image.to(device=device, dtype=input_dtype, non_blocking=True)

                # cur_txt_feat = cat_id_2_txt_feat['2']
                # image_features = model.encode_image(cur_image, text_embed=cur_txt_feat)
                # image_features /= image_features.norm(dim=-1, keepdim=True)
                # img_id_2_img_feat[img_id] = image_features

                image_features_i = []
                for cat_id in cat_id_2_txt_feat.keys():
                    cur_txt_feat = cat_id_2_txt_feat[cat_id]
                    image_features = model.encode_image(cur_image, text_embed=cur_txt_feat)
                    image_features_i.append(image_features)
                    continue

                image_features_i = torch.concat(image_features_i, dim=0)
                all_logits = image_features_i @ all_text_features.T
                logits_cate = torch.diagonal(all_logits, dim1=-2, dim2=-1)
                prompt_idx = torch.argmax(logits_cate)
                image_features = image_features_i[prompt_idx].unsqueeze(0)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                img_id_2_img_feat[img_id] = image_features

            top1_recall = 0
            top5_recall = 0
            top10_recall = 0
            recall_cnt = 0
            for cat_id in cat_id_2_txt_feat.keys():
                if cat_id == '1':
                    continue
                print(coco_91[int(cat_id)])
                cur_txt_feat = cat_id_2_txt_feat[cat_id]
                logit_list = []
                occ_img_id_list = cat_id_2_occ_img[cat_id]
                neg_img_id_list = cat_id_2_negative_img[cat_id]
                for img_id in occ_img_id_list:
                    cur_img_feat = img_id_2_img_feat[str(img_id)]
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0].cpu().numpy().tolist()
                    logit_list.append([cur_logit, 1])
                for img_id in neg_img_id_list:
                    cur_img_feat = img_id_2_img_feat[str(img_id)]
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0].cpu().numpy().tolist()
                    logit_list.append([cur_logit, 0])
                logit_list.sort(key=lambda x: -x[0])
                logit_list = np.array(logit_list)[:,1]
                print(logit_list[:len(occ_img_id_list)])
                if logit_list[:len(occ_img_id_list)].sum() == len(occ_img_id_list):
                    top1_recall += 1
                if logit_list[:5*len(occ_img_id_list)].sum() == len(occ_img_id_list):
                    top5_recall += 1
                if logit_list[:10*len(occ_img_id_list)].sum() == len(occ_img_id_list):
                    top10_recall += 1
                recall_cnt += 1

            print('top1 recall: ', top1_recall / recall_cnt)
            print('top5 recall: ', top5_recall / recall_cnt)
            print('top10 recall: ', top10_recall / recall_cnt)
            print('recall cnt: ', recall_cnt)
    return 0


def calculate_ap(logit_prediction_pairs):
    # Sort the predictions by logit (confidence) in descending order
    sorted_pairs = sorted(logit_prediction_pairs, key=lambda x: x[0], reverse=True)

    tp = 0  # True positives
    fp = 0  # False positives
    precisions = []

    for i, (logit, prediction) in enumerate(sorted_pairs):
        if prediction == 1:  # True positive
            tp += 1
            precision = tp / (tp + fp)
            precisions.append(precision)
        else:
            fp += 1  # False positive

    if tp == 0:
        return 0.0  # No true positives, AP is 0

    # Average of the precision values at each true positive
    return np.mean(precisions)


def calculate_ap_new(logit_prediction_pairs):
    # Sort the predictions by logit (confidence) in descending order
    sorted_pairs = sorted(logit_prediction_pairs, key=lambda x: x[0], reverse=True)

    tp = 0  # True positives
    fp = 0  # False positives
    precisions = []

    precisions.append(1.0) 

    for i, (logit, prediction) in enumerate(sorted_pairs):
        if prediction == 1:  # True positive
            tp += 1
            precision = tp / (tp + fp)
            precisions.append(precision)
        else:
            fp += 1  # False positive

    if tp == 0:
        return 0.0  # No true positives, AP is 0

    # Average of the precision values at each true positive
    return np.mean(precisions), precisions


def tgvpt_evaluate_reranking_sbu(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        # with torch.no_grad():
            # all_images, all_texts = [], []
            # for i, batch in enumerate(dataloader):
            #     print(i)
            #     with autocast():
            #         images, texts = batch
            #         images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            #         texts = texts.to(device=device, non_blocking=True)
            #         all_images.append(images)
            #         all_texts.append(texts)

                    # model_out = model(images, texts)
                    # image_features = model_out["image_features"]
                    # text_features = model_out["text_features"]
                    # logit_scale = model_out["logit_scale"]
                    # # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # # however, system RAM is easily exceeded and compute time becomes problematic
                    # all_text_features.append(text_features.cpu())
                    # logit_scale = logit_scale.mean()
                    # logits_per_image = logit_scale * image_features @ text_features.t()
                    # logits_per_text = logits_per_image.t()

                    # batch_size = images.shape[0]
                    # labels = torch.arange(batch_size, device=device).long()
                    # total_loss = (
                    #     F.cross_entropy(logits_per_image, labels) +
                    #     F.cross_entropy(logits_per_text, labels)
                    # ) / 2

                    # gen_loss = maybe_compute_generative_loss(model_out)

                # cumulative_loss += total_loss * batch_size
                # num_samples += batch_size
                # if is_master(args) and (i % 100) == 0:
                #     logging.info(
                #         f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                #         f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                #     if gen_loss is not None:
                #         cumulative_gen_loss += gen_loss * batch_size
                #         logging.info(
                #             f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")
            # all_text_features = torch.cat(all_text_features).to(device=device, non_blocking=True)

            # all_images = torch.cat(all_images, dim=0)
            # all_texts = torch.cat(all_texts, dim=0)

        import pickle
        import json
        txt_indices = json.load(open('DATASET_PATH/sbu_randomly_sampled_indices_9.17.json'))[:100]
        topk = 5000
        org_clip_features = pickle.load(open('DATASET_PATH/clip_vitb16_sbu_infer_img_txt_features_9.4.pkl', 'rb'))
        org_image_features = org_clip_features['image_features']
        org_text_features = org_clip_features['text_features']
        logits_per_image = (org_image_features @ org_text_features[txt_indices].t()).detach().cpu()
        logits_per_text = logits_per_image.t().detach().cpu()
        # logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
        ranking = torch.argsort(logits_per_text, descending=True)
        top1_recall = 0
        top5_recall = 0
        top10_recall = 0
        recall_cnt = 0
        recall_topk_curve = {}
        name = 'text_to_image'
        for topi in range(1, 5000):
            recall_topk_curve[f"{name}_R@{topi}"] = 0
        # for text_i in range(ranking.shape[0]):
        for text_idx in range(len(txt_indices)):
            text_i = txt_indices[text_idx]
            print('text_idx: ', text_idx)
            print('text_i: ', text_i)
            topk_img_indices = ranking[text_idx][:topk]
            # image_i_2_tg_img_feat = {}
            logit_imgi_list = []
            # ipdb.set_trace()
            # tg_img_feat = model.encode_image(all_images[topk_img_indices], text_embed=org_text_features[text_i].unsqueeze(0), normalize=True)
            for image_i in topk_img_indices.detach().cpu().tolist():
                # print(image_i)
                # ipdb.set_trace()
                tg_img_feat = model.encode_image(dataloader.dataset[image_i][0].unsqueeze(0).to(device), text_embed=org_text_features[text_i].unsqueeze(0).to(device), normalize=True)
                tg_img_feat = tg_img_feat.squeeze()
                # image_i_2_tg_img_feat[image_i] = tg_img_feat
                logit_imgi_list.append([(org_text_features[text_i].to(device) * tg_img_feat).sum().detach().cpu().tolist(), image_i])
            logit_imgi_list.sort(key=lambda x: -x[0])
            # ipdb.set_trace()
            logit_list = np.array(logit_imgi_list)[:,1].tolist()
            if text_i in logit_list[:1]:
                print('top1')
                top1_recall += 1
            if text_i in logit_list[:5]:
                print('top5')
                top5_recall += 1
            if text_i in logit_list[:10]:
                print('top10')
                top10_recall += 1
            recall_cnt += 1

            for topi in range(1, 5000):
                recall_topk_curve[f"{name}_R@{topi}"] += text_i in logit_list[:topi]
                
        print('top1_recall: ', top1_recall / recall_cnt)
        print('top5_recall: ', top5_recall / recall_cnt)
        print('top10_recall: ', top10_recall / recall_cnt)

        for topi in range(1, 5000):
            recall_topk_curve[f"{name}_R@{topi}"] =  recall_topk_curve[f"{name}_R@{topi}"] / recall_cnt
        ipdb.set_trace()
        with open('PROJECT_PATH/recall_k_sbu_data_after_reranking_9.26_datacompDR12M_bs56_ep1_100samples.json', 'w') as dump_f:
            json.dump(recall_topk_curve, dump_f)
                
                
    return 0


def evaluate_occluded_coco_tgvpt2(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    logging.info('start evaluation')
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        logging.info('start evaluation')

        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []


        from torchvision import transforms as pth_transforms
        from PIL import Image
        occluded_coco_retrieval_ann_file = json.load(open('DATASET_PATH/coco2017val_cat_id_2_occ_img_negative_img.json'))
        cat_id_2_occ_img = occluded_coco_retrieval_ann_file['cat_id_2_occ_img']
        cat_id_2_negative_img = occluded_coco_retrieval_ann_file['cat_id_2_negative_img']
        img_id_2_all_ins_id = occluded_coco_retrieval_ann_file['img_id_2_all_ins_id']
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
        img_preprocess = pth_transforms.Compose([
            pth_transforms.Resize([224, 224], pth_transforms.InterpolationMode.BICUBIC),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])


        with torch.no_grad():

            cat_id_2_txt_feat = {}
            iter_txt = 0
            all_text_features = []
            for cat_id in cat_id_2_occ_img.keys():
                print(iter_txt)
                iter_txt += 1
                cur_txt = coco_91[int(cat_id)]
                text = tokenizer([cur_txt]).to(device=device, non_blocking=True)
                text_features = model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                cat_id_2_txt_feat[cat_id] = text_features
                all_text_features.append(text_features)
            all_text_features = torch.concat(all_text_features, dim=0)

            img_id_2_img_feat = {}
            img_folder = '/scratch/shared/beegfs/shared-datasets/COCO/COCO2017/val2017/'
            iter_img = 0
            for img_id in tqdm(img_id_2_all_ins_id.keys()):
                # ipdb.set_trace()
                # print(iter_img)
                iter_img += 1
                cur_image = img_preprocess(Image.open(os.path.join(img_folder, str(img_id).rjust(12,'0')+'.jpg')).convert('RGB')).unsqueeze(0)
                # if img_id == '289343':
                cur_image = cur_image.to(device=device, dtype=input_dtype, non_blocking=True)

                # cur_txt_feat = cat_id_2_txt_feat['2']
                # image_features = model.encode_image(cur_image, text_embed=cur_txt_feat)
                # image_features /= image_features.norm(dim=-1, keepdim=True)
                # img_id_2_img_feat[img_id] = image_features

                image_features_i = []
                for cat_id in cat_id_2_txt_feat.keys():
                    cur_txt_feat = cat_id_2_txt_feat[cat_id]
                    image_features = model.encode_image(cur_image, text_embed=cur_txt_feat)
                    image_features_i.append(image_features)
                    continue

                image_features_i = torch.concat(image_features_i, dim=0)
                all_logits = image_features_i @ all_text_features.T
                logits_cate = torch.diagonal(all_logits, dim1=-2, dim2=-1)
                prompt_idx = torch.argmax(logits_cate)
                image_features = image_features_i[prompt_idx].unsqueeze(0)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                img_id_2_img_feat[img_id] = image_features

            top1_recall = 0
            top5_recall = 0
            top10_recall = 0
            ap_sum = 0
            recall_cnt = 0
            for cat_id in cat_id_2_txt_feat.keys():
                if cat_id == '1':
                    continue
                print(coco_91[int(cat_id)])
                cur_txt_feat = cat_id_2_txt_feat[cat_id]
                logit_list = []
                occ_img_id_list = cat_id_2_occ_img[cat_id]
                neg_img_id_list = cat_id_2_negative_img[cat_id]
                for img_id in occ_img_id_list:
                    cur_img_feat = img_id_2_img_feat[str(img_id)]
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0].cpu().numpy().tolist()
                    logit_list.append([cur_logit, 1])
                for img_id in neg_img_id_list:
                    cur_img_feat = img_id_2_img_feat[str(img_id)]
                    cur_logit = (cur_img_feat @ cur_txt_feat.T)[0][0].cpu().numpy().tolist()
                    logit_list.append([cur_logit, 0])
                logit_list.sort(key=lambda x: -x[0])
                cur_ap = calculate_ap(logit_list)
                ap_sum += cur_ap
                logit_list = np.array(logit_list)[:,1]
                print(logit_list[:len(occ_img_id_list)])
                if logit_list[:len(occ_img_id_list)].sum() == len(occ_img_id_list):
                    top1_recall += 1
                if logit_list[:5*len(occ_img_id_list)].sum() == len(occ_img_id_list):
                    top5_recall += 1
                if logit_list[:10*len(occ_img_id_list)].sum() == len(occ_img_id_list):
                    top10_recall += 1
                recall_cnt += 1

            print('top1 recall: ', top1_recall / recall_cnt)
            print('top5 recall: ', top5_recall / recall_cnt)
            print('top10 recall: ', top10_recall / recall_cnt)
            print('mAP: ', ap_sum / recall_cnt)
            print('recall cnt: ', recall_cnt)
    return 0



def tgvpt_evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            all_images, all_texts = [], []
            for i, batch in enumerate(dataloader):
                with autocast():
                    images, texts = batch
                    images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                    texts = texts.to(device=device, non_blocking=True)
                    all_images.append(images)
                    all_texts.append(texts)

                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")
            all_text_features = torch.cat(all_text_features).to(device=device, non_blocking=True)

            all_images = torch.cat(all_images, dim=0)
            all_texts = torch.cat(all_texts, dim=0)
            for i in tqdm(range(num_samples)):
                image_features_i = []
                for j in range(num_samples):
                    with autocast():
                        # model_out = model(all_images[i].unsqueeze(0), all_texts[j].unsqueeze(0))
                        # image_features = model_out["image_features"]
                        image_features = model.encode_image(all_images[i].unsqueeze(0), text_embed=all_text_features[j].unsqueeze(0), normalize=True)
                        image_features_i.append(image_features)
                image_features_i = torch.cat(image_features_i).to(device=device, non_blocking=True)
                all_logits = image_features_i @ all_text_features.T
                logits_cate = torch.diagonal(all_logits, dim1=-2, dim2=-1)
                prompt_idx = torch.argmax(logits_cate)
                all_image_features.append(image_features_i[prompt_idx].unsqueeze(0))

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features,dim=0),
                text_features=all_text_features,
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def mp_tgvpt_evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    from tqdm import tqdm

    metrics = {}
    # ----------- multi-GPU running 2024.9.8 -----------
    # if not is_master(args):
    #     return metrics
    # ----------- multi-GPU running 2024.9.8 -----------
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            all_images, all_texts = [], []
            for i, batch in enumerate(dataloader):
                # dist.barrier()
                with autocast():
                    images, texts = batch
                    images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                    texts = texts.to(device=device, non_blocking=True)
                    all_images.append(images)
                    all_texts.append(texts)

                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i%100==0):
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")
            all_text_features = torch.cat(all_text_features).to(device=device, non_blocking=True)

            all_images = torch.cat(all_images, dim=0)
            all_texts = torch.cat(all_texts, dim=0)
            for i in tqdm(range(num_samples)):
                image_features_i = []
                for j in range(num_samples):
                    with autocast():
                        # model_out = model(all_images[i].unsqueeze(0), all_texts[j].unsqueeze(0))
                        # image_features = model_out["image_features"]
                        image_features = model.module.encode_image(all_images[i].unsqueeze(0), text_embed=all_text_features[j].unsqueeze(0), normalize=True)
                        image_features_i.append(image_features)
                image_features_i = torch.cat(image_features_i).to(device=device, non_blocking=True)
                all_logits = image_features_i @ all_text_features.T
                logits_cate = torch.diagonal(all_logits, dim1=-2, dim2=-1)
                prompt_idx = torch.argmax(logits_cate)
                all_image_features.append(image_features_i[prompt_idx].unsqueeze(0))

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features,dim=0),
                text_features=all_text_features,
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    if is_master(args):
        logging.info(
            f"Eval Epoch: {epoch} "
            + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
        )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)

