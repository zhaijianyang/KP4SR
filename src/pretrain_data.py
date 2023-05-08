from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import gzip
import random
from multiprocessing import Pool
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
import os
from torch.utils.data.distributed import DistributedSampler
from copy import deepcopy

from transformers import T5Tokenizer, T5TokenizerFast
from tokenization import P5Tokenizer, P5TokenizerFast

import matplotlib.pyplot as plt

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def ReadLineFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

class TreeNode:
    def __init__(self, val, token_ids, tokenized_text, token_obs_pos, is_id=False):
        self.val = val
        self.is_id = is_id
        self.token_ids = token_ids
        self.tokenized_text = tokenized_text
        self.token_obs_pos = token_obs_pos
        self.tail_entity = None
        self.children = []
        self.parent = None
        self.level = -1

    def add_child(self, child):
        child.parent = self
        child.level = self.level + 1
        self.children.append(child)

    def get_parent_value(self):
        if self.parent:
            return self.parent.val
        else:
            return None

    def get_parent_children_values(self):
        if self.parent:
            return [child.val for child in self.parent.children if child != self]
        else:
            return None


class KG_Dataset:
    def __init__(self, args, split):
        self.relation2template = load_json(os.path.join(args.data_url, split, 'relation2template.json'))
        self.triple2dict = load_json(os.path.join(args.data_url, split, 'triple2dict.json'))
        self.item2entity = load_json(os.path.join(args.data_url, split, 'item2entity.json'))
        self.entity2text = load_json(os.path.join(args.data_url, split, 'entity2text.json'))
    
class P5_Amazon_Dataset(Dataset):
    def __init__(self, all_tasks, task_list, tokenizer, args, sample_numbers, mode='train', split='toys'): 
        self.all_tasks = all_tasks
        self.task_list = task_list
        self.tokenizer = tokenizer
        self.args = args
        self.sample_numbers = sample_numbers
        self.split = split

        self.degree  = args.degree
        self.hop_num = args.hop_num
        self.max_items = args.max_items
        
        print('Data sources: ', split.split(','))

        kg_data = KG_Dataset(args, split)
        self.relation2template = kg_data.relation2template
        self.triple2dict = kg_data.triple2dict
        self.item2entity = kg_data.item2entity
        self.entity2text = kg_data.entity2text

        self.mode = mode
            
        self.sequential_data = ReadLineFromFile(os.path.join(args.data_url, split, 'sequential_data.txt'))
            
        datamaps = load_json(os.path.join(args.data_url, split, 'datamaps.json'))
        self.user2id = datamaps['user2id']
        self.item2id = datamaps['item2id']
        self.user_list = list(datamaps['user2id'].keys())
        self.item_list = list(datamaps['item2id'].keys())
        self.id2item = datamaps['id2item']

        self.entity2item_id = {}
        for item, entity in self.item2entity.items():
            self.entity2item_id[entity] = self.item2id[item]
        
        self.user_id2name = load_pickle(os.path.join(args.data_url, split, 'user_id2name.pkl'))
            
        print('compute_datum_info')
        self.total_length = 0
        self.datum_info = []
        self.compute_datum_info()
        
    def compute_datum_info(self):
        curr = 0
        for key in list(self.task_list.keys()):
            if key == 'rating':
                self.total_length += len(self.rating_data) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            elif key == 'sequential':
                self.total_length += len(self.sequential_data) * self.sample_numbers[key]
                for i in range(self.total_length - curr):
                    self.datum_info.append((i + curr, key, i // self.sample_numbers[key]))
                curr = self.total_length
            else:
                raise NotImplementedError
    
    def get_root(self, source_text, purchase_history):
        tokens = source_text.split()

        source_ids = []
        source_tokenized_text = []
        abs_pos = 0

        child_node = []

        for i, ele in enumerate(tokens):
            tokenized_text = self.tokenizer.tokenize(ele)
            ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)

            abs_pos_i = [pos + abs_pos for pos in range(len(ids))]

            if ele in purchase_history:
                node = TreeNode(val=ele, token_ids=ids, tokenized_text=tokenized_text, token_obs_pos=abs_pos_i, is_id=True)
                node.tail_entity = ele
                child_node.append(node)

            abs_pos += len(ids)
            source_ids.extend(ids)
            source_tokenized_text.extend(tokenized_text)
        
        root_node = TreeNode(val=source_text, token_ids=source_ids, tokenized_text=source_tokenized_text, token_obs_pos=list(range(abs_pos)))
        for node in child_node:
            root_node.add_child(node)

        return root_node, abs_pos
    
    def get_next_level_nodes(self, nodes, obs_pos, flag=True):
        next_nodes = []
        for node in nodes:
            children = self.get_kg_text(node.tail_entity, node.is_id)
            for child in children:
                obs_pos_list = [obs_pos + idx for idx in range(len(child.token_ids))]
                obs_pos += len(child.token_ids)
                child.token_obs_pos = obs_pos_list
                node.add_child(child)
                next_nodes.append(child)

                if obs_pos > self.args.max_text_length:
                    flag = False
                    break
            if not flag:
                break

        return next_nodes, obs_pos, flag
    
    def get_input(self, all_nodes, abs_pos):
        input_ids = []
        tokenized_texts = []
        all_texts = ''
        visible_matrix = np.zeros((abs_pos, abs_pos))
        for node in all_nodes:
            if node.level != 0:   
                input_ids.extend(node.token_ids)
                tokenized_texts.extend(node.tokenized_text)
                all_texts += node.val
                all_texts += ' '

            visible_pos = []
            if node.parent:
                visible_pos.extend(node.parent.token_obs_pos)
                for child in node.parent.children:
                    visible_pos.extend(child.token_obs_pos)
            else:
                visible_pos = node.token_obs_pos

            if node.children:
                for child in node.children:
                    visible_pos.extend(child.token_obs_pos)

            for id in node.token_obs_pos:
                visible_matrix[id, visible_pos] = 1
            
            # for id in visible_pos:
            #     visible_matrix[id, visible_pos] = 1

        return all_texts, input_ids, tokenized_texts, visible_matrix
    
    def get_kg_text(self, entity, is_id = False):
        if is_id:
            head_text = entity
            item = self.id2item[entity]
            entity = self.item2entity[item]
        else:
            head_text = self.entity2text[entity]

        #### 有可能不存在
        entity2dict = self.triple2dict.get(entity)
        if entity2dict is None:
            return []

        if len(list(entity2dict.keys())) > self.degree:
            if not self.args.test_random:
                if self.mode == 'train':
                    candi_rel = np.random.choice(list(entity2dict.keys()), self.degree, replace=False)
                else:
                    candi_rel = list(entity2dict.keys())[: self.degree]
            else:
                candi_rel = np.random.choice(list(entity2dict.keys()), self.degree, replace=False)
        else:
            candi_rel = list(entity2dict.keys())

        children = []
        for relation in candi_rel:
            relation_template = self.relation2template[relation]
            tail_list = entity2dict[relation]
            if len(tail_list) > 1:
                if not self.args.test_random:
                    if self.mode == 'train':
                        tail_list = np.random.choice(tail_list, 1)
                else:
                    tail_list = np.random.choice(tail_list, 1)

            tail = tail_list[0]

            item_id = self.entity2item_id.get(tail)
            if not item_id:
                tail_text = self.entity2text[tail]
                is_id = False
            else:
                tail_text = item_id
                tail = item_id
                is_id = True

            text = relation_template.replace('[X]', head_text)
            text = text.replace('[Y]', tail_text)

            tokenized_text = self.tokenizer.tokenize(text)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            node = TreeNode(val=text, token_ids=token_ids, tokenized_text=tokenized_text, token_obs_pos=None, is_id=is_id)
            node.tail_entity = tail
            children.append(node)

        return children
    
            
    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        
        out_dict = {}
        out_dict['args'] = self.args
        
        loss_weight = 1.0
        
        datum_info_idx = self.datum_info[idx]
        assert datum_info_idx[0] == idx
        if len(datum_info_idx) == 3:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
        elif len(datum_info_idx) == 4:
            task_name = datum_info_idx[1]
            datum_idx = datum_info_idx[2]
            task_idx = datum_info_idx[3]
        else:
            raise NotImplementedError
            
        if task_name == 'sequential':
            sequential_datum = self.sequential_data[datum_idx]
            sequence = sequential_datum.split()
            user_id = sequence[0]
            user_desc = self.user_id2name[user_id]
            if self.mode == 'train':
                end_candidates = [_ for _ in range(min(self.max_items, len(sequence) - 4), len(sequence) - 3)]
                end_index = random.randint(0, len(end_candidates)-1)
                end_pos = end_candidates[end_index]
                start_pos = max(1, end_pos - self.max_items + 1)
                purchase_history = sequence[start_pos:end_pos+1]
                target_item = sequence[end_pos+1]
            elif self.mode == 'val':
                start = max(1, len(sequence) - self.max_items - 2)
                purchase_history = sequence[start:-2]
                target_item = sequence[-2]
            elif self.mode == 'test':
                start = max(1, len(sequence) - self.max_items - 1)
                purchase_history = sequence[start:-1]
                target_item = sequence[-1]
            else:
                raise NotImplementedError
            
            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates)-1) # random choose the task index for task_candidates
            task_template = self.all_tasks['sequential'][task_candidates[task_idx]]
            assert task_template['task'] == 'sequential'
            
            if self.split == 'books':
                task_temp = ['2-1', '2-3', '2-4', '2-6', '2-9', '2-10', '2-11']

                if task_template['id'] in task_temp:
                    source_text = task_template['source'].format(user_id, ' , '.join(purchase_history))
                else:
                    source_text = task_template['source'].format(user_desc, ' , '.join(purchase_history))

            elif self.split == 'movielens' or self.split == 'ml-10m':
                task_temp = ['2-4', '2-9', '2-10']
                task_temp_seq = ['2-3', '2-5', '2-7']

                if task_template['id'] in task_temp :
                    user_info = user_id
                else:
                    user_info = user_desc

                if task_template['id'] in task_temp_seq:
                    source_text = task_template['source'].format(' , '.join(purchase_history), user_info)
                else:
                    source_text = task_template['source'].format(user_info, ' , '.join(purchase_history))

            elif self.split == 'flm-1b':
                task_temp = ['2-2', '2-5', '2-10']
                task_temp_seq = ['2-3', '2-4', '2-6', '2-7', '2-8']

                if task_template['id'] in task_temp :
                    user_info = user_id
                else:
                    user_info = user_desc

                if task_template['id'] in task_temp_seq:
                    source_text = task_template['source'].format(' , '.join(purchase_history), user_info)
                else:
                    source_text = task_template['source'].format(user_info, ' , '.join(purchase_history))

            target_text = task_template['target'].format(target_item)

        root_node, abs_pos = self.get_root(source_text, purchase_history)
        all_nodes = [root_node]
        nodes = root_node.children
        all_nodes.extend(nodes)

        flag = True
        for i in range(self.hop_num):
            nodes, abs_pos, flag = self.get_next_level_nodes(nodes, abs_pos, flag=flag)
            all_nodes.extend(nodes)
            if not flag:
                break

        all_texts, input_ids, tokenized_texts, visible_matrix = self.get_input(all_nodes, abs_pos)
        # plt.imshow(visible_matrix, cmap='gray', interpolation='nearest')
        # plt.savefig('filename.png')
        source_ids = root_node.token_ids

        assert len(tokenized_texts) == len(input_ids)

        input_ids = input_ids[:self.args.max_text_length]
        tokenized_texts = tokenized_texts[:self.args.max_text_length]
        whole_word_ids = self.calculate_whole_word_ids(tokenized_texts)
        assert len(whole_word_ids) == len(input_ids)

        
        target_ids = self.tokenizer.encode(
                target_text, padding=True, truncation=True, max_length=self.args.gen_max_length)

        out_dict['visible_matrix'] = torch.LongTensor(visible_matrix)
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        out_dict['whole_word_ids'] = torch.LongTensor(whole_word_ids)
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['source_ids'] = torch.LongTensor(source_ids)
        out_dict['target_length'] = len(target_ids)
        out_dict['source_length'] = len(source_ids)

        out_dict['source_text'] = source_text
        out_dict['tokenized_text'] = tokenized_texts
        out_dict['target_text'] = target_text

        out_dict['task'] = task_template['task']

        out_dict['loss_weight'] = loss_weight

        return out_dict
    
    def calculate_whole_word_ids(self, tokenized_text):
        whole_word_ids = []
        curr = 0
        for i in range(len(tokenized_text)):
            if tokenized_text[i].startswith('▁'):
                curr += 1
                whole_word_ids.append(curr)
            else:
                whole_word_ids.append(curr)
        # last_item = whole_word_ids[len(input_ids) - 2]
        return whole_word_ids ## the added [0] is for </s>
    
    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        args = self.args

        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)

        visible_matrix = torch.zeros(B, S_W_L, S_W_L, dtype=torch.long)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        whole_word_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        cross_mask = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        loss_weights = torch.ones(B, dtype=torch.float)

        tasks = []
        source_text = []
        tokenized_text = []
        target_text = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            whole_word_ids[i, :entry['input_length']] = entry['whole_word_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']

            if self.args.cross_mask:
                cross_mask[i, :entry['source_length']] = 1
            else:
                cross_mask[i, :entry['input_length']] = 1

            if entry['visible_matrix'].shape[0] > S_W_L:
                visible_matrix[i] = entry['visible_matrix'][:S_W_L, :S_W_L]
            else:
                visible_matrix[i, :entry['input_length'], :entry['input_length']] = entry['visible_matrix']

            if 'task' in entry:
                tasks.append(entry['task'])

            if 'source_text' in entry:
                source_text.append(entry['source_text'])
                
            if 'tokenized_text' in entry:
                tokenized_text.append(entry['tokenized_text'])
                
            if 'target_text' in entry:
                target_text.append(entry['target_text'])

            if 'loss_weight' in entry:
                ## length-aware loss normalization
                if entry['target_length'] > 0:
                    loss_weights[i] = entry['loss_weight'] / entry['target_length']
                else:
                    loss_weights[i] = entry['loss_weight']

        if not self.args.tree_mask:
            visible_matrix = input_ids.ne(self.tokenizer.pad_token_id)

        assert 't5' in args.backbone
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100
        batch_entry['task'] = tasks

        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text

        batch_entry['input_ids'] = input_ids
        batch_entry['whole_word_ids'] = whole_word_ids
        batch_entry['target_ids'] = target_ids

        batch_entry['loss_weights'] = loss_weights
        batch_entry['visible_matrix'] = visible_matrix
        batch_entry['cross_mask'] = cross_mask

        return batch_entry
    

def get_loader(args, task_list, sample_numbers, split='toys', mode='train', 
               batch_size=16, workers=4, distributed=False):

    if 't5' in args.backbone:
        tokenizer = P5Tokenizer.from_pretrained(
            args.backbone, 
            max_length=args.max_text_length, 
            do_lower_case=args.do_lower_case)

    if split == 'books':
        from mask_amazon_templates import all_tasks as task_templates
    elif split == 'flm-1b':
        from mask_flm_1b_templates import all_tasks as task_templates
    else:
        from mask_movielens_templates import all_tasks as task_templates

    dataset = P5_Amazon_Dataset(
        task_templates,
        task_list,
        tokenizer,
        args,
        sample_numbers,
        mode=mode,
        split=split,
    )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)
        
    return loader
