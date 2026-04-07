import logging

import numpy as np
import torch
from tqdm import tqdm

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES, COCO_CLASSNAMES, SIMPLE_COCO_TEMPLATES, \
    UCF101_NAMES, SIMPLE_ACTION_TEMPLATES, REID_NAMES, SIMPLE_REID_TEMPLATES, CUB_NAMES, SIMPLE_CUB_TEMPLATES, \
    COCO_CLASSNAMES_KNOWN, COCO_CLASSNAMES_NOVEL
from .precision import get_autocast

from open_clip.transformer import PerCatePromptedTransformer

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    percate = False
    if isinstance(model.visual.transformer, PerCatePromptedTransformer):
        percate = True
        cate_num = model.visual.transformer.cate_num

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        logging.info(f'Evaluating on {len(dataloader)} samples')
        for images, target in tqdm(dataloader, unit_scale=dataloader.batch_size):
            # print(target)
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                if percate:
                    all_image_features = []
                    B = images.shape[0]
                    for i in range(cate_num):
                        cate_label = torch.ones(B).type(torch.long).cuda() * i
                        output = model(image=images, label=cate_label)
                        image_features = output['image_features'] if isinstance(output, dict) else output[0]
                        all_image_features.append(image_features)
                    all_image_features = torch.cat([a.unsqueeze(1) for a in all_image_features], dim=1)
                    # using target to get prompt idx
                    # prompt_idx = target
                    # method1： max on 2D matrix
                    # all_logits = all_image_features @ classifier
                    # all_logits_flatten = all_logits.view(all_logits.shape[0], -1)
                    # _, prompt_idx = all_logits_flatten.max(dim=1)
                    # prompt_idx = prompt_idx // cate_num
                    # method2: max on diagonal
                    all_logits = all_image_features @ classifier
                    logits_cate = torch.diagonal(all_logits, dim1=-2, dim2=-1)
                    prompt_idx = torch.argmax(logits_cate, dim=1)

                    final_image_features = torch.cat([all_image_features[i, pid, :].unsqueeze(0) for i, pid in enumerate(prompt_idx)], dim=0)
                    image_features = final_image_features
                else:
                    output = model(image=images)
                    image_features = output['image_features'] if isinstance(output, dict) else output[0]
                    if len(image_features.shape) == 3:
                        if args.res_token:
                            image_features = torch.add(image_features[:,1:,:], image_features[:, 0, :].unsqueeze(1))
                        else:
                            image_features = image_features[:, 1:]

                        if args.token == 'diag' or args.token == 'all':
                            all_logits = image_features @ classifier
                            logits = 100. * torch.diagonal(all_logits, dim1=-2, dim2=-1)
                        elif args.token == 'first':
                            logits = 100. * image_features[:,0,:] @ classifier
                        else:
                            raise NotImplementedError('Not supported token type!')
                    else:
                        logits = 100. * image_features @ classifier

                # res = logits[:, 0]
                # print("target", target)
                # print("res", res)

                # if percate:
                #     #[NxC,..., NxC]
                #     print("image_features.shape",image_features.shape)
                #     image_features_of_all_categories = torch.split(image_features, cate_num, dim=0)
                #     # print("image_features_of_all_categories.shape", image_features_of_all_categories.shape)
                #     # NxNcxC
                #     res_features = torch.cat([t.unsqueeze(1) for t in image_features_of_all_categories], dim=1).permute(1,0,2)
                #     print("res_features.shape", res_features.shape)
                #     # (NxNcxC, CxT) -> NxNcxT
                #     all_logits = res_features @ classifier
                #     print("all_logits.shape", all_logits.shape)
                #     # method1： max on 2D matrix
                #     # all_logits_flatten = all_logits.view(all_logits.shape[0], -1)
                #     # _, prompt_idx = all_logits_flatten.max(dim=1)
                #     # prompt_idx = prompt_idx // cate_num
                #     # method2: max on diagonal
                #     logits_cate = torch.diagonal(all_logits, dim1=-2, dim2=-1)
                #     print("logits_cate.shape", logits_cate.shape)
                #     prompt_idx = torch.argmax(logits_cate, dim=1)
                #     print("prompt_idx.shape", prompt_idx.shape)
                #     prompt_idx = target
                #     logits = 100. * torch.cat([all_logits[i,pid,:].unsqueeze(0) for i, pid in enumerate(prompt_idx)], dim=0)
                # else:
                #     logits = 100. * image_features @ classifier
            # print('image_features.shape', image_features.shape)
            # print('logits.shape', logits.shape)

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def run_reid(model, dataloader, args):
    from .reid_utils import cal_metrics

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    with torch.no_grad():
        logging.info(f'Evaluating on {len(dataloader)} samples')
        feats = []
        labels = []
        query_flag = []
        for images, target, is_query in tqdm(dataloader, unit_scale=dataloader.batch_size):
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)
            is_query = is_query.to(args.device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]

            feats.append(image_features)
            labels.append(target)
            query_flag.append(is_query)

    feats = torch.cat(feats, dim=0).cpu()
    labels = torch.cat(labels, dim=0).cpu()
    query_flag = torch.cat(query_flag, dim=0).cpu()

    print(feats.shape)
    print(query_flag.shape)
    print(torch.sum(query_flag == 1))
    print(torch.sum(query_flag == 0))

    feats_q = feats[query_flag == 1]
    feats_g = feats[query_flag == 0]
    labels_q = labels[query_flag == 1]
    labels_g = labels[query_flag == 0]

    r1,r5,r10,map = cal_metrics(feats_q, labels_q, feats_g, labels_g)
    return r1,r5,r10,map


def run_ir(model, dataloader, args):
    from .reid_utils import cal_metrics

    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    with torch.no_grad():
        logging.info(f'Evaluating on {len(dataloader)} samples')
        feats = []
        labels = []
        for images, target in tqdm(dataloader, unit_scale=dataloader.batch_size):
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]

            feats.append(image_features)
            labels.append(target)

    feats = torch.cat(feats, dim=0).cpu()
    labels = torch.cat(labels, dim=0).cpu()

    r1, r5, r10, map = cal_metrics(feats, labels, feats, labels)
    return r1, r5, r10, map

    # names, accs = evaluate_emb(feats, labels)
    # return accs[0], accs[1], accs[2], accs[3], accs[4]


def run_withclass(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        logging.info(f'Evaluating on {len(dataloader)} samples')
        for images, texts, target in tqdm(dataloader, unit_scale=dataloader.batch_size):
            # print(target)
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)

            with autocast():
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]

                if len(image_features.shape) == 3:
                    if args.res_token:
                        image_features = torch.add(image_features[:, 1:, :], image_features[:, 0, :].unsqueeze(1))
                    else:
                        image_features = image_features[:, 1:]

                    if args.token == 'diag' or args.token == 'all':
                        all_logits = image_features @ classifier
                        logits = 100. * torch.diagonal(all_logits, dim1=-2, dim2=-1)
                    elif args.token == 'first':
                        logits = 100. * image_features[:, 0, :] @ classifier
                    else:
                        raise NotImplementedError('Not supported token type!')
                else:
                    logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def run_withmask(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        logging.info(f'Evaluating on {len(dataloader)} samples')
        for images, texts, target, objects, occluders, occludees in tqdm(dataloader, unit_scale=dataloader.batch_size):
        # for images, target in tqdm(dataloader, unit_scale=dataloader.batch_size):
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)
            objects = objects.to(device=args.device, dtype=input_dtype)
            occluders = occluders.to(device=args.device, dtype=input_dtype)
            occludees = occludees.to(device=args.device, dtype=input_dtype)

            with autocast():
                output = model(image=[images, objects, occluders, occludees])
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def zero_shot_eval(model, data, epoch, args, tokenizer=None):
    if ('imagenet-val' not in data and 'imagenet-v2' not in data and 'coco-val' not in data
            and 'ucf101-val' not in data and 'reid-val' not in data and 'cub-val' not in data):
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info('Starting zero-shot imagenet.')
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    classnames = IMAGENET_CLASSNAMES
    templates = OPENAI_IMAGENET_TEMPLATES
    if 'coco-val' in data:
        classnames = COCO_CLASSNAMES
        templates = SIMPLE_COCO_TEMPLATES

    if 'ucf101-val' in data:
        classnames = UCF101_NAMES
        templates = SIMPLE_ACTION_TEMPLATES

    if 'reid-val' in data:
        classnames = REID_NAMES
        templates = SIMPLE_REID_TEMPLATES

    if 'cub-val' in data:
        classnames = CUB_NAMES
        templates = SIMPLE_CUB_TEMPLATES

    logging.info('Building zero-shot classifier')
    autocast = get_autocast(args.precision)

    with autocast():
        if 'reid-val' not in data:
            classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=classnames,
                templates=templates,
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            )

        if 'val_novel' in data:
            classnames = COCO_CLASSNAMES_KNOWN
            classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=classnames,
                templates=templates,
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            )
            classnames = COCO_CLASSNAMES_NOVEL
            classifier_novel = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=classnames,
                templates=templates,
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            )
            # save the text embedding
            # np.save('/home/ypliu/projects/OccludedCLIP/open_clip-main/embeddings/known_txt_embedd.npy', classifier.cpu().detach().numpy())
            # np.save('/home/ypliu/projects/OccludedCLIP/open_clip-main/embeddings/novel_txt_embedd.npy', classifier_novel.cpu().detach().numpy())
            # print('-------------------------------- save embeddings --------------------------------')
            
    logging.info('Using classifier')
    results = {}
    if 'imagenet-val' in data:
        top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5
    if 'coco-val' in data:
        if 'val_novel' in data:
            top1, top5 = run_withclass(model, classifier, data['val'].dataloader, args)
            f = open('/home/ypliu/projects/OccludedCLIP/open_clip-main/embeddings/known_flag.txt', 'w')
            f.write('novel')
            f.close()
            top1_novel, top5_novel = run_withclass(model, classifier_novel, data['val_novel'].dataloader, args)
            f = open('/home/ypliu/projects/OccludedCLIP/open_clip-main/embeddings/known_flag.txt', 'w')
            f.write('known')
            f.close()
            results['coco-zeroshot-val-top1_novel'] = top1_novel
            results['coco-zeroshot-val-top5_novel'] = top5_novel
        else:
            top1, top5 = run(model, classifier, data['coco-val'].dataloader, args)
        results['coco-zeroshot-val-top1'] = top1
        results['coco-zeroshot-val-top5'] = top5
    if 'ucf101-val' in data:
        top1, top5 = run(model, classifier, data['ucf101-val'].dataloader, args)
        results['ucf101-zeroshot-val-top1'] = top1
        results['ucf101-zeroshot-val-top5'] = top5

    if 'reid-val' in data:
        # print(len(data['val'].dataloader.dataset), ' dataset length')
        r1,r5,r10,map = run_reid(model, data['val'].dataloader, args)
        results['reid-zeroshot-map'] = map

    if 'cub-val' in data:
        top1, top5 = run(model, classifier, data['cub-val'].dataloader, args)
        results['cub-zeroshot-val-top1'] = top1
        results['cub-zeroshot-val-top5'] = top5

        r1,r5,r10,map  = run_ir(model, data['cub-val'].dataloader, args)
        results['cub-zeroshot-map'] = map
    # if 'train' in data:
    #     if args.csv_mask_key != "none":
    #         top1, top5 = run_withmask(model, classifier, data['train_val'].dataloader, args)
    #     else:
    #         top1, top5 = run_withclass(model, classifier, data['train_val'].dataloader, args)
    #     results['coco-train-top1'] = top1
    #     results['coco-train-top5'] = top5
    # if 'val' in data:
    #     if args.csv_mask_key != "none":
    #         top1, top5 = run_withmask(model, classifier, data['val'].dataloader, args)
    #     else:
    #         top1, top5 = run_withclass(model, classifier, data['val'].dataloader, args)
    #     results['coco-val-top1'] = top1
    #     results['coco-val-top5'] = top5

    logging.info('Finished zero-shot imagenet.')

    return results
