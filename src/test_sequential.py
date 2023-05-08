import sys
import os
import argparse
from pathlib import Path
import logging
from datetime import datetime
import json
import glob
import time
import random

import torch_npu
from torch_npu.contrib import transfer_to_npu

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pretrain_data import get_loader
from utils import LossMeter, load_state_dict
from pretrain_model import P5Pretraining
from notebooks.evaluate.metrics4rec import evaluate_all

def convert_json_key(param_dict):
    """
    json.dump不支持key是int的dict，在编码存储的时候会把所有的int型key写成str类型的
    所以在读取json文件后，用本方法将所有的被解码成str的int型key还原成int
    """
    new_dict = dict()
    for key, value in param_dict.items():
        if isinstance(value, (dict,)):
            res_dict = convert_json_key(value)
            try:
                new_key = int(key)
                new_dict[new_key] = res_dict
            except:
                new_dict[key] = res_dict
        else:
            try:
                new_key = int(key)
                new_dict[new_key] = value
            except:
                new_dict[key] = value

    return new_dict


def create_config(args):
    from transformers import T5Config, BartConfig

    if 't5' in args.backbone:
        config_class = T5Config
    else:
        return None

    config = config_class.from_pretrained(args.backbone)
    config.backbone = args.backbone
    config.losses = args.losses

    return config


def create_tokenizer(args):
    from transformers import T5Tokenizer, T5TokenizerFast
    from src.tokenization import P5Tokenizer, P5TokenizerFast

    if 'p5' in args.tokenizer:
        tokenizer_class = P5Tokenizer

    tokenizer_name = args.backbone
    
    tokenizer = tokenizer_class.from_pretrained(
        tokenizer_name,
        max_length=args.max_text_length,
        do_lower_case=args.do_lower_case,
    )
    
    return tokenizer


def create_model(model_class, config=None):

    model_name = config.backbone

    model = model_class.from_pretrained(
        model_name,
        config=config
    )
    return model

def now_time():
    return '[' + datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_url", default='/home/ma-user/work/datasets/KE-P5/data')
    parser.add_argument("--test", default='books')
    parser.add_argument('--load', type=str, default='/home/ma-user/work/obs_output/P5_mask_seq/books_t5-small_v3/BEST_EVAL_LOSS.pth', 
                        help='Load the model')
    parser.add_argument('--train_url', type=str, default='/home/ma-user/work/obs_output/P5_mask_seq/books_t5-small_v3', 
                        help='Load the model')

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument("--num_workers", default=160, type=int)
    parser.add_argument('--seed', type=int, default=2023)

    # Model Config
    parser.add_argument('--backbone', type=str, default='t5-small')
    parser.add_argument('--tokenizer', type=str, default='p5')
    parser.add_argument('--whole_word_embed', action='store_false')

    parser.add_argument('--max_text_length', type=int, default=512)
    parser.add_argument('--gen_max_length', type=int, default=64)
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument("--losses", default='sequential', type=str)
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--degree', type=int, default=20, help='relation number of kg')
    parser.add_argument('--hop_num', type=int, default=2, help='number of kg hop')
    parser.add_argument('--max_items', type=int, default=5, help='number of items')

    parser.add_argument("--tree_mask", action='store_true')
    parser.add_argument("--cross_mask", action='store_true')
    parser.add_argument("--test_random", action='store_true')

    parser.add_argument("--test_prompt", default='2-11', type=str)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    _, name = os.path.split(args.load)
    if args.local_rank == 0:
        logging.basicConfig(filename = os.path.join(args.train_url, name.split('.')[0] + '_' + args.test_prompt + ".log"), filemode = 'w', level = logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler())
        logging.info(now_time() + 'begin test')
        logging.info(args)

    main_worker(args.local_rank, args)

def main_worker(local_rank, args):

    args.local_rank = local_rank
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')

    config = create_config(args)
    tokenizer = create_tokenizer(args)

    model_class = P5Pretraining
    model = create_model(model_class, config)

    if 'p5' in args.tokenizer:
        model.resize_token_embeddings(tokenizer.vocab_size)
        
    model.tokenizer = tokenizer

    state_dict = load_state_dict(args.load, 'cpu')
    results = model.load_state_dict(state_dict, strict=False)
    if args.local_rank == 0:
        logging.info('Model loaded from {}'.format(args.load))
        logging.info(results)

    model = model.cuda()
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank])

    test_task_list = {'sequential': [args.test_prompt]}
    test_sample_numbers = {'rating': 1, 'sequential': 1, 'explanation': 1, 'review': 1, 'traditional': (1, 1)}

    test_loader = get_loader(
            args,
            test_task_list,
            test_sample_numbers,
            split=args.test, 
            mode='test', 
            batch_size=args.batch_size,
            workers=args.num_workers,
            distributed=args.distributed
    )

    if args.local_rank == 0:
        logging.info(len(test_loader))
        for infile in glob.glob(os.path.join(args.train_url, '*.json')):
            os.remove(infile)

    all_info = []
    model.eval()
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        with torch.no_grad():
            # results = model.generate_step(batch)
            cross_mask = batch['cross_mask'].repeat_interleave(20, dim=0)

            if args.distributed:
                beam_outputs = model.module.generate(
                        batch['input_ids'].to('cuda'), 
                        attention_mask = batch['visible_matrix'].to('cuda'),   #######
                        max_length=50, 
                        num_beams=20,
                        no_repeat_ngram_size=0, 
                        num_return_sequences=20,
                        early_stopping=True,
                        cross_mask=cross_mask.to('cuda'),
                        # **kwargs
                )
                generated_sents = model.module.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)

            else:
                beam_outputs = model.generate(
                        batch['input_ids'].to('cuda'), 
                        attention_mask = batch['visible_matrix'].to('cuda'), #########
                        max_length=50, 
                        num_beams=20,
                        no_repeat_ngram_size=0, 
                        num_return_sequences=20,
                        early_stopping=True,
                        cross_mask=cross_mask.to('cuda'),
                        # **kwargs
                )
                generated_sents = model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)

            for j, item in enumerate(batch['target_text']):
                new_info = {}
                new_info['target_item'] = item.split(' ')[1]
                new_info['gen_item_list'] = generated_sents[j*20: (j+1)*20]
                all_info.append(new_info)
 
    gt = {}
    ui_scores = {}
    for i, info in enumerate(all_info):
        gt[i] = [int(info['target_item'])]
        pred_dict = {}
        for j in range(len(info['gen_item_list'])):
            try:
                pred_dict[int(info['gen_item_list'][j])] = -(j+1)
            except:
                pass
        ui_scores[i] = pred_dict

    if args.distributed:

        save_path = os.path.join(args.train_url, 'temp_{}.json'.format(args.local_rank))
        with open(save_path, "w") as f:
            json.dump({'ui_scores': ui_scores, 'gt': gt}, f)
        
        # dist.barrier()

        if args.local_rank == 0:
            time.sleep(300)
            temps = glob.glob(os.path.join(args.train_url, '*.json'))
            ui_scores_temp = {}
            gt_temp = {}
            num = 0
            num_1 = 0
            for temp_path in temps:
                logging.info(temp_path)
                with open(temp_path, "r") as f:
                    temp = json.load(f)

                for idx, item in temp['ui_scores'].items():
                    ui_scores_temp[num] = convert_json_key(item)
                    num += 1
                for idx, item in temp['gt'].items():
                    gt_temp[num_1] = item
                    num_1 += 1
            # logging.info(gt_temp)
            logging.info('num: {}'.format(num))
            msg, res = evaluate_all(ui_scores_temp, gt_temp, 5)
            logging.info(msg)
            msg, res = evaluate_all(ui_scores_temp, gt_temp, 10)
            logging.info(msg)
    else:
        # logging.info(gt)
        msg, res = evaluate_all(ui_scores, gt, 5)
        logging.info(msg)
        msg, res = evaluate_all(ui_scores, gt, 10)
        logging.info(msg)


if __name__ == "__main__":
    torch_npu.npu.set_compile_mode(jit_compile=False)
    main()
