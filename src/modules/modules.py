import copy
import datetime
import numpy as np
import sys
import time
import torch
import torch.nn.functional as F
import models
from itertools import compress
from config import cfg
from data import make_data_loader, make_batchnorm_stats, FixTransform, MixDataset
from utils import to_device, make_optimizer, collate, to_device, calc_kld_from_logits
from metrics import Accuracy
import sqlite3
import json
import ast
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from agent.server import ServerAgentDQN
import wandb

class Server:
    def __init__(self, model):
        self.model_state_dict = save_model_state_dict(model.state_dict())
        self.server_epoch = cfg['server']['num_epochs']
        if 'fmatch' in cfg['loss_mode']:
            optimizer = make_optimizer(model.make_sigma_parameters(), 'local')
            global_optimizer = make_optimizer(model.make_phi_parameters(), 'global')
        else:
            optimizer = make_optimizer(model.parameters(), 'local')
            global_optimizer = make_optimizer(model.parameters(), 'global')
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
        
    def distribute(self, client, batchnorm_dataset=None):
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model.load_state_dict(self.model_state_dict)
        if batchnorm_dataset is not None:
            model = make_batchnorm_stats(batchnorm_dataset, model, 'global')
        model_state_dict = save_model_state_dict(model.state_dict())
        for m in range(len(client)):
            if client[m].active:
                client[m].model_state_dict = copy.deepcopy(model_state_dict)
        return

    def update(self, client):  # aggregation
        if 'fmatch' not in cfg['loss_mode']:
            with torch.no_grad():
                valid_client = [client[i] for i in range(len(client)) if client[i].active]
                if len(valid_client) > 0:
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model.load_state_dict(self.model_state_dict)
                    global_optimizer = make_optimizer(model.parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client))
                    weight = weight / weight.sum()
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v = v.data.new_zeros(v.size())
                            for m in range(len(valid_client)):
                                tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                            v.grad = (v.data - tmp_v).detach()
                    global_optimizer.step()
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())
        elif 'fmatch' in cfg['loss_mode']:
            with torch.no_grad():
                valid_client = [client[i] for i in range(len(client)) if client[i].active]
                if len(valid_client) > 0:
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model.load_state_dict(self.model_state_dict)
                    global_optimizer = make_optimizer(model.make_phi_parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client))
                    weight = weight / weight.sum() 
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v = v.data.new_zeros(v.size())
                            for m in range(len(valid_client)):
                                tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                            v.grad = (v.data - tmp_v).detach()
                    global_optimizer.step()
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())
        else:
            raise ValueError('Not valid loss mode')
        for i in range(len(client)):
            client[i].active = False

        return

    def update_parallel(self, client):
        if 'frgd' not in cfg['loss_mode']:
            with torch.no_grad():
                valid_client_server = [self] + [client[i] for i in range(len(client)) if client[i].active]
                if len(valid_client_server) > 0:
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model.load_state_dict(self.model_state_dict)
                    global_optimizer = make_optimizer(model.parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client_server))
                    weight = weight / (2 * (weight.sum() - 1))
                    weight[0] = 1 / 2 if len(valid_client_server) > 1 else 1
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v = v.data.new_zeros(v.size())
                            for m in range(len(valid_client_server)):
                                tmp_v += weight[m] * valid_client_server[m].model_state_dict[k]
                            v.grad = (v.data - tmp_v).detach()
                    global_optimizer.step()
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())
        elif 'frgd' in cfg['loss_mode']:
            with torch.no_grad():
                valid_client_server = [self] + [client[i] for i in range(len(client)) if client[i].active]
                num_valid_client = len(valid_client_server) - 1
                if len(valid_client_server) > 0:
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model.load_state_dict(self.model_state_dict)
                    global_optimizer = make_optimizer(model.parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client_server)) / (num_valid_client // 2 + 1)
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v_1 = v.data.new_zeros(v.size())
                            tmp_v_1 += weight[0] * valid_client_server[0].model_state_dict[k]
                            for m in range(1, num_valid_client // 2 + 1):
                                tmp_v_1 += weight[m] * valid_client_server[m].model_state_dict[k]
                            tmp_v_2 = v.data.new_zeros(v.size())
                            tmp_v_2 += weight[0] * valid_client_server[0].model_state_dict[k]
                            for m in range(num_valid_client // 2 + 1, len(valid_client_server)):
                                tmp_v_2 += weight[m] * valid_client_server[m].model_state_dict[k]
                            tmp_v = (tmp_v_1 + tmp_v_2) / 2
                            v.grad = (v.data - tmp_v).detach()
                    global_optimizer.step()
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())
        else:
            raise ValueError('Not valid loss mode')
        for i in range(len(client)):
            client[i].active = False
        return

    def train(self, dataset, lr, metric, logger):  # Fine-tuning
        if 'fmatch' not in cfg['loss_mode']:
            data_loader = make_data_loader({'train': dataset}, 'server')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['server']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][1])))
            else:
                num_batches = None
            for epoch in range(1, cfg['server']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        else:
            data_loader = make_data_loader({'train': dataset}, 'server')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.make_sigma_parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['server']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][1])))
            else:
                num_batches = None
            for epoch in range(1, cfg['server']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            v.grad[(v.grad.size(0) // 2):] = 0
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break

        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.model_state_dict = save_model_state_dict(model.state_dict())
        return


class Client:
    def __init__(self, client_id, model, data_split):
        self.client_id = client_id
        self.data_split = data_split
        self.model_state_dict = save_model_state_dict(model.state_dict())
        if 'fmatch' in cfg['loss_mode']:
            optimizer = make_optimizer(model.make_phi_parameters(), 'local')
        else:
            optimizer = make_optimizer(model.parameters(), 'local')
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.active = False
        self.beta = torch.distributions.beta.Beta(torch.tensor([cfg['alpha']]), torch.tensor([cfg['alpha']]))
        self.verbose = cfg['verbose']
        
        #self.threshold = np.random.uniform(0.7, 0.95)
        self.threshold = 0.95
        #self.client_epoch = np.random.randint(1, 11)
        self.client_epoch = 5
        
        self.db_data = {}
        self.db_data['round'] = 0
        self.db_data['client_id'] = client_id
        self.db_data['threshold'] = self.threshold 
        self.db_data['client_epoch'] = self.client_epoch
        self.db_data['loss'] = -1
        self.db_data['label_ratio'] = -1
        self.db_data['msp'] = -1
        self.db_data['time'] = -1
    
        
    def evaluate_using_server_data(self, server, dataset, batchnorm_dataset=None):
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model.load_state_dict(server.model_state_dict)
        data_loader = make_data_loader({'train': dataset}, 'server')['train']
        print(dir(model))
        assert False
        
        
    def make_hard_pseudo_label(self, soft_pseudo_label):  # threshold 이상이면 hard label로 변환
        max_p, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        mask = max_p.ge(self.threshold)
        return hard_pseudo_label, mask

    def make_dataset(self, dataset, metric, logger):  # FixMatch 기반 pseudo-label + dataset 생성
        self.dataset = dataset
        if 'sup' in cfg['loss_mode']:
            return dataset
        elif 'fix' in cfg['loss_mode']:
            with torch.no_grad():
                data_loader = make_data_loader({'train': dataset}, 'global', shuffle={'train': False})['train']
                model = eval('models.{}(track=True).to(cfg["device"])'.format(cfg['model_name']))
                model.load_state_dict(self.model_state_dict)
                model.train(False)
                output = []
                target = []
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input = to_device(input, cfg['device'])
                    output_ = model(input)
                    output_i = output_['target']
                    target_i = input['target']
                    output.append(output_i.cpu())
                    target.append(target_i.cpu())
                    
                output_, input_ = {}, {}
                output_['target'] = torch.cat(output, dim=0)
                input_['target'] = torch.cat(target, dim=0)
                
                gmodel_ldata_output_logit = torch.cat(output, dim=0).numpy().tolist()
                output_logit_txt = [[float(f"{x:.4f}") for x in row] for row in gmodel_ldata_output_logit]
                json_gmodel_ldata_output_logit = json.dumps(output_logit_txt)
                
                json_client_label = json.dumps(np.round(input_['target'].cpu().numpy(),4).tolist())
                
                output_['target'] = F.softmax(output_['target'], dim=-1)
                new_target, mask = self.make_hard_pseudo_label(output_['target'])
                output_['mask'] = mask
                evaluation = metric.evaluate(['PAccuracy', 'MAccuracy', 'LabelRatio'], input_, output_)
                msp = output_['target'].max(dim=-1).values
                msp_mean = msp.mean().item()
                self.db_data['msp'] = msp_mean
                self.db_data['label_ratio'] = evaluation['LabelRatio']
        
                self.db_data['gm_ld_client_labels'] = json_client_label
                self.db_data['gm_ld_output_logit'] = json_gmodel_ldata_output_logit
                logger.append(evaluation, 'train', n=len(input_['target']))
                if torch.any(mask):
                    fix_dataset = copy.deepcopy(dataset)
                    fix_dataset.target = new_target.tolist()
                    mask = mask.tolist()
                    fix_dataset.data = list(compress(fix_dataset.data, mask))
                    fix_dataset.target = list(compress(fix_dataset.target, mask))
                    fix_dataset.other = {'id': list(range(len(fix_dataset.data)))}
                    if 'mix' in cfg['loss_mode']:
                        mix_dataset = copy.deepcopy(dataset)
                        mix_dataset.target = new_target.tolist()
                        mix_dataset = MixDataset(len(fix_dataset), mix_dataset)
                    else:
                        mix_dataset = None
                    return fix_dataset, mix_dataset
                else:
                    return None
        else:
            raise ValueError('Not valid client loss mode')

    def train(self, dataset, lr, metric, logger):
        if cfg['loss_mode'] == 'sup':
            data_loader = make_data_loader({'train': dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if self.client_epoch == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            for epoch in range(1, self.client_epoch + 1):
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        elif 'fix' in cfg['loss_mode'] and 'mix' not in cfg['loss_mode'] and 'batch' not in cfg[
            'loss_mode'] and 'frgd' not in cfg['loss_mode'] and 'fmatch' not in cfg['loss_mode']:
            fix_dataset, _ = dataset
            fix_data_loader = make_data_loader({'train': fix_dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if self.client_epoch == 1:
                num_batches = int(np.ceil(len(fix_data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            for epoch in range(1, self.client_epoch + 1):
                for i, input in enumerate(fix_data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        elif 'fix' in cfg['loss_mode'] and 'mix' in cfg['loss_mode'] and 'batch' not in cfg[
            'loss_mode'] and 'frgd' not in cfg['loss_mode'] and 'fmatch' not in cfg['loss_mode']:
            fix_dataset, mix_dataset = dataset
            fix_data_loader = make_data_loader({'train': fix_dataset}, 'client')['train']
            mix_data_loader = make_data_loader({'train': mix_dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if self.client_epoch == 1:
                num_batches = int(np.ceil(len(fix_data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            for epoch in range(1, self.client_epoch + 1):
                for i, (fix_input, mix_input) in enumerate(zip(fix_data_loader, mix_data_loader)):
                    input = {'data': fix_input['data'], 'target': fix_input['target'], 'aug': fix_input['aug'],
                             'mix_data': mix_input['data'], 'mix_target': mix_input['target']}
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input['lam'] = self.beta.sample()[0]
                    input['mix_data'] = (input['lam'] * input['data'] + (1 - input['lam']) * input['mix_data']).detach()
                    input['mix_target'] = torch.stack([input['target'], input['mix_target']], dim=-1)
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        elif 'batch' in cfg['loss_mode'] or 'frgd' in cfg['loss_mode'] or 'fmatch' in cfg['loss_mode']:
            data_loader = make_data_loader({'train': dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            if 'fmatch' in cfg['loss_mode']:
                optimizer = make_optimizer(model.make_phi_parameters(), 'local')
            else:
                optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if self.client_epoch == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            for epoch in range(1, self.client_epoch + 1):
                for i, input in enumerate(data_loader):
                    with torch.no_grad():
                        model.train(False)
                        input_ = collate(input)
                        input_ = to_device(input_, cfg['device'])
                        output_ = model(input_)
                        output_i = output_['target']
                        output_['target'] = F.softmax(output_i, dim=-1)
                        new_target, mask = self.make_hard_pseudo_label(output_['target'])
                        output_['mask'] = mask
                        evaluation = metric.evaluate(['PAccuracy', 'MAccuracy', 'LabelRatio'], input_, output_)
                        logger.append(evaluation, 'train', n=len(input_['target']))
                    if torch.all(~mask):
                        continue
                    model.train(True)
                    input = {'data': input['data'][mask], 'aug': input['aug'][mask], 'target': new_target[mask]}
                    input = to_device(input, cfg['device'])
                    input_size = input['data'].size(0)
                    input['loss_mode'] = 'fix'
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        else:
            raise ValueError('Not valid client loss mode')
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.model_state_dict = save_model_state_dict(model.state_dict())
        self.db_data['loss'] = evaluation['Loss']



        model.load_state_dict(self.model_state_dict)
        model.train(False)        
        output = []
        target = []
        data_loader = make_data_loader({'train': self.dataset}, 'client', shuffle={'train': False})['train']
        for i, input in enumerate(data_loader):
            with torch.no_grad():
                input = collate(input)
                input = to_device(input, cfg['device'])
                output_ = model(input)
                output_i = output_['target']
                target_i = input['target']
                output.append(output_i.cpu())
                target.append(target_i.cpu())
        output_logit = torch.cat(output, dim=0).numpy()
        input_labels = torch.cat(target, dim=0).numpy().tolist()
        output_logit = output_logit.round(4).tolist()
        input_labels = torch.cat(target, dim=0).numpy().tolist()
        output_logit_txt = []
        for row in output_logit:
            output_logit_txt.append([float(f"{x:.4f}") for x in row])
        json_input_labels = json.dumps(input_labels)
        json_client_logit = json.dumps(output_logit_txt)
        
        len(json_input_labels)
        self.db_data['lm_ld_client_labels'] = json_input_labels
        self.db_data['lm_ld_client_output_logit'] = json_client_logit
        return
       
        
    def sqlite_insert_data(self, db_name, _round, server_evaluation):
        data =  (_round, 
                 self.client_id.item(),
                 self.threshold,
                 self.client_epoch,
                 self.participant_frequency,
                 self.db_data['loss'],
                 self.db_data['label_ratio'],
                 self.db_data['msp'],
                 self.db_data['lm_ld_client_labels'],
                 self.db_data['lm_ld_client_output_logit'],
                 self.db_data['gm_ld_client_labels'],
                 self.db_data['gm_ld_output_logit'],
                 server_evaluation['Loss'],
                 server_evaluation['Accuracy'],
                 server_evaluation['labels'],
                 server_evaluation['output_logit'])
        conn = sqlite3.connect(db_name, timeout=5)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO fed_rl_data (round, 
                                     client_number, 
                                     threshold, 
                                     local_epoch,
                                     participant_frequency, 
                                     local_loss, 
                                     local_label_ratio, 
                                     local_msp, 
                                     lm_ld_client_labels,
                                     lm_ld_client_output_logit,
                                     gm_ld_client_labels,
                                     gm_ld_output_logit,
                                     server_loss,
                                     server_acc,
                                     server_labels,
                                     server_output_logit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', data)
        conn.commit()
        conn.close()

def save_model_state_dict(model_state_dict):
    return {k: v.cpu() for k, v in model_state_dict.items()}


def save_optimizer_state_dict(optimizer_state_dict):
    optimizer_state_dict_ = {}
    for k, v in optimizer_state_dict.items():
        if k == 'state':
            optimizer_state_dict_[k] = to_device(optimizer_state_dict[k], 'cpu')
        else:
            optimizer_state_dict_[k] = copy.deepcopy(optimizer_state_dict[k])
    return optimizer_state_dict_

class RLAgent:
    def __init__(self, num_clients, state_dim, window_size, device='cuda'):
        self.num_clients = num_clients
        self.device = device
        self.state_dim = state_dim
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 100
        self.window_size = window_size 
        # ServerAgentDQN 초기화
        self.agent = ServerAgentDQN(state_dim, num_clients, device)
        from agent.server import BATCH_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.current_round = 0
      

    def build_state(self, clients, current_round):
        """
        DB에서 클라이언트별 최근 상태를 가져와 state 벡터 생성
        """
        state = {
            'round': current_round/cfg['global']['num_epochs'],
            'participant_frequency': [],
            'client_loss': [],
            'client_msp': [],
            'data_size': [],
            'kld_softmax': []
        }

        db_name = cfg['client_db_path']
        conn = sqlite3.connect(db_name, timeout=5)
        cursor = conn.cursor()

        for cid in range(self.num_clients):
            cursor.execute('''
                SELECT participant_frequency, local_msp, lm_ld_client_labels, server_output_logit
                FROM fed_rl_data
                WHERE client_number = ?
                ORDER BY round DESC
                LIMIT 1
            ''', (cid,))
            result = cursor.fetchone()
            
            if result is not None:
                pfreq, msp, client_labels, server_output_logit = result
                state['participant_frequency'].append(pfreq/80.0)
                # state['client_loss'].append(loss)
                state['client_msp'].append(msp)
                if client_labels is not None:
                    try:
                        labels = ast.literal_eval(client_labels)
                        data_size = len(labels)
                    except Exception:
                        data_size = 0
                    state['data_size'].append(data_size/500.0)

                if server_output_logit is not None:
                    try:
                        logits = ast.literal_eval(server_output_logit)
                        kl_divergence = calc_kld_from_logits(logits)
                    except Exception:
                        kl_divergence = 0.0
                    state['kld_softmax'].append(kl_divergence)
            else:
                state['participant_frequency'].append(0.0)
                # state['client_loss'].append(0.0)
                state['client_msp'].append(0.8)
                state['data_size'].append(0.0)
                state['kld_softmax'].append(0.0)

            cursor.execute('''
                SELECT local_loss
                FROM fed_rl_data
                WHERE client_number = ?
                ORDER BY round DESC
                LIMIT ?
            ''', (cid, self.window_size))

            loss_results = cursor.fetchall()
            loss_history = [float(loss[0]) for loss in loss_results]
            
            # window_size만큼 채우기
            while len(loss_history) < self.window_size:
                loss_history.append(0.0)
                
            state['client_loss'].append(loss_history)
            
        conn.close()
        
        # 상태를 numpy 배열로 변환
        state_array = np.array([
            state['round'],
            *state['participant_frequency'],
            *[loss for loss_history in state['client_loss'] for loss in loss_history],  # flatten
            *state['client_msp'],
            *state['data_size'],
            *state['kld_softmax']
        ], dtype=np.float32)
        
        print("state_dim:", len(state_array))
        
        return state_array

    def select_clients(self, state, num_active_clients):
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                        np.exp(-1. * self.current_round / self.epsilon_decay)
        if random.random() < eps_threshold:
            selected = np.random.choice(self.num_clients, num_active_clients, replace=False).tolist()
            print(f"[RLAgent][Epsilon-Greedy] 랜덤 선택 (epsilon: {eps_threshold:.3f}) → {selected}")
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.agent.policy_net(state_tensor)
                # print(f"[RLAgent][Epsilon-Greedy] Q값 벡터: {q_values.cpu().numpy().flatten()}")
                selected = torch.topk(q_values, num_active_clients)[1].squeeze().cpu().numpy().tolist()
                # print(f"[RLAgent][Epsilon-Greedy] Q값 기반 선택 (epsilon: {eps_threshold:.3f}) → {selected}")
        return selected

    def update_dqn(self, current_state, client_ids, client_loss_now, server_loss_now, next_state, done=False):
        print(f"[RLAgent][ReplayBuffer] 현재 버퍼 크기: {len(self.agent.memory)}/{self.BATCH_SIZE}")
        rewards = []
        l_server_list = []
        delta_l_global_list = []
        kld_list = []
        
        from db import get_global_model_result
        prev_global_loss = 0.0  # 기본값 0으로 설정
        if self.current_round > 1:
            prev_result = get_global_model_result(cfg['server_db_path'], self.current_round - 1)
            if prev_result:
                prev_global_loss = prev_result['agg_ft_server_loss']
        
        # 현재 라운드 글로벌 모델 정보 가져오기
        current_result = get_global_model_result(cfg['server_db_path'], self.current_round)
        current_global_loss = 0.0
        if current_result:
            current_global_loss = current_result['agg_ft_server_loss']

        lambda1 = 0.5  # L_server 가중치
        lambda2 = 0.7  # delta(L_global) 가중치
        lambda3 = 0.3  # KLD 가중치
        
        if client_loss_now and server_loss_now:
            for i, cid in enumerate(client_ids):
                l_server = server_loss_now[i] if i < len(server_loss_now) else 0.0
                
                # delta L_global 계산 
                delta_l_global = current_global_loss - prev_global_loss
                
                # KLD 값 가져오기 (state에서 해당 클라이언트의 KLD 값)
                # 상태 구조: [round, *participant_frequency, *client_loss_histories, *client_msp, *data_size, *kld_softmax]
                # kld_softmax 인덱스: 1 + num_clients + window_size*num_clients + 2*num_clients + cid
                kld_index = 1 + self.num_clients + self.window_size * self.num_clients + 2 * self.num_clients + cid
                kld = current_state[kld_index] if kld_index < len(current_state) else 0.0
                
                l_server_list.append(l_server)
                delta_l_global_list.append(delta_l_global)
                kld_list.append(kld)
                
                # 리워드 계산: r = -(λ1*L_server + λ2*ΔL_global + λ3*KLD) + 1
                reward = -(lambda1 * l_server - lambda2 * delta_l_global + lambda3 * kld) + 1.0
                rewards.append(reward)
                print(f"[RLAgent][Reward] 클라이언트 {cid}: L_server={l_server:.4f}, ΔL_global={delta_l_global:.4f}, KLD={kld:.4f}, 보상={reward:.4f}")
        else:
            rewards = [0.0] * len(client_ids)

        for i, cid in enumerate(client_ids):
            reward = rewards[i] if i < len(rewards) else 0.0
            action_idx = cid
            self.agent.store_transition(current_state, action_idx, reward, next_state, done)
            print(f"[RLAgent][ReplayBuffer] 저장: state_shape={len(current_state)}, action={action_idx}, reward={reward:.4f}")
            loss = self.agent.learn()
            # if loss is not None:
            #     wandb.log({"epoch": self.current_round,
            #           "epsilon": self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            #           np.exp(-1. * self.current_round / self.epsilon_decay),
            #           "dqn/reward": reward, 
            #           "dqn/loss": loss})
                          
        # --- trajectory logging ---
        if l_server_list:
            wandb.log({
                "trajectory/l_server_mean": np.mean(l_server_list),
                "trajectory/l_server_std": np.std(l_server_list),
                "trajectory/delta_l_global_mean": np.mean(delta_l_global_list),
                "trajectory/delta_l_global_std": np.std(delta_l_global_list),
                "trajectory/kld_mean": np.mean(kld_list),
                "trajectory/kld_std": np.std(kld_list),
                "epoch": self.current_round
            }, step=self.current_round)
        
        if self.current_round % self.agent.TARGET_UPDATE == 0:
            self.agent.update_target_net()

    def update_round(self):
        self.current_round += 1

        
