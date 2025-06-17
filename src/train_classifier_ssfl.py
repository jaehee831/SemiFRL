import argparse
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from config import cfg, process_args
from data import fetch_dataset, split_dataset, make_data_loader, separate_dataset, separate_dataset_su, \
    make_batchnorm_dataset_su, make_batchnorm_stats
from metrics import Metric
from modules import Server, Client
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import make_logger
from db import create_train_database,create_test_database,sqlite_insert_test_data,get_global_model_result, \
                get_client_recent_info, get_client_server_output_softmax_sum
import sqlite3
import json

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        cfg['client_db_path'] = f"DB/exp5_{cfg['model_tag']}_client.db"
        cfg['server_db_path'] = f"DB/exp5_{cfg['model_tag']}_server.db"
        create_train_database(db_name=cfg['client_db_path'])
        create_test_database(db_name=cfg['server_db_path'])
        print('Experiment: {}'.format(cfg['model_tag']))
        
        cfg['participant_frequency'] = {i:0 for i in range(cfg['num_clients'])}
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    server_dataset = fetch_dataset(cfg['data_name'])
    client_dataset = fetch_dataset(cfg['data_name'])
    process_dataset(server_dataset)
    server_dataset['train'], client_dataset['train'], supervised_idx = separate_dataset_su(server_dataset['train'],
                                                                                           client_dataset['train'])
    data_loader = make_data_loader(server_dataset, 'global')
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    optimizer = make_optimizer(model.parameters(), 'local')
    scheduler = make_scheduler(optimizer, 'global')
    if cfg['sbn'] == 1:
        batchnorm_dataset = make_batchnorm_dataset_su(server_dataset['train'], client_dataset['train'])
    elif cfg['sbn'] == 0:
        batchnorm_dataset = server_dataset['train']
    else:
        raise ValueError('Not valid sbn')
    data_split = split_dataset(client_dataset, cfg['num_clients'], cfg['data_split_mode'])
    if cfg['loss_mode'] != 'sup':
        metric = Metric({'train': ['Loss', 'Accuracy', 'PAccuracy', 'MAccuracy', 'LabelRatio'],
                         'test': ['Loss', 'Accuracy']})
    else:
        metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'])
        last_epoch = result['epoch']
        if last_epoch > 1:
            data_split = result['data_split']
            supervised_idx = result['supervised_idx']
            server = result['server']
            client = result['client']
            optimizer.load_state_dict(result['optimizer_state_dict'])
            scheduler.load_state_dict(result['scheduler_state_dict'])
            logger = result['logger']
        else:
            server = make_server(model)
            client = make_client(model, data_split)
            logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    else:
        last_epoch = 1
        server = make_server(model)
        client = make_client(model, data_split)
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    
    for epoch in range(last_epoch, cfg['global']['num_epochs'] + 1):
        train_client(batchnorm_dataset, server_dataset['train'], client_dataset['train'], server, client, optimizer, metric, logger, epoch)
        if 'ft' in cfg and cfg['ft'] == 0:
            train_server(server_dataset['train'], server, optimizer, metric, logger, epoch)
            logger.reset()
            server.update_parallel(client)
        else:
            logger.reset()
            server.update(client)
            # 서버 모델 평가 추가
            model.load_state_dict(server.model_state_dict)
            data_loader_server_train = make_data_loader({'train': server_dataset['train']}, 'server')['train']
            model = model.to(cfg["device"])        
            server_eval = test(data_loader_server_train, model, metric, logger, epoch)
            
            train_server(server_dataset['train'], server, optimizer, metric, logger, epoch)
        
        logger.reset()
        model.load_state_dict(server.model_state_dict)
        test_model = make_batchnorm_stats(batchnorm_dataset, model, 'global')
        test_eval = test(data_loader['test'], test_model, metric, logger, epoch)
        
        # 서버 평가 후 DB 저장 추가
        logger.reset()
        data_loader_server_train = make_data_loader({'train': server_dataset['train']}, 'server')['train']
        server_eval_ft = test(data_loader_server_train, test_model, metric, logger, epoch)
        
        # 테스트 결과 DB에 저장
        sqlite_insert_test_data(cfg['server_db_path'], epoch, test_eval, server_eval, server_eval_ft)
        
        # optimizer.step() 이후에 scheduler.step() 호출
        scheduler.step()
        
        result = {'cfg': cfg, 'epoch': epoch + 1, 'server': server, 'client': client,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict(),
                  'supervised_idx': supervised_idx, 'data_split': data_split, 'logger': logger}
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
        
        # 마지막 라운드에서 메트릭 플로팅
        if epoch == cfg['global']['num_epochs']:
            # 결과 저장 디렉토리 생성
            result_dir = os.path.join('output', 'plots', cfg['model_tag'])
            os.makedirs(result_dir, exist_ok=True)
            
            # 테스트 정확도 플로팅
            plot_metrics(
                logger, 
                'test/Accuracy',
                os.path.join(result_dir, 'test_accuracy.png')
            )
            
            # 테스트 손실값 플로팅
            plot_metrics(
                logger, 
                'test/Loss',
                os.path.join(result_dir, 'test_loss.png')
            )
            
            # 학습 정확도 플로팅
            plot_metrics(
                logger, 
                'train/Accuracy',
                os.path.join(result_dir, 'train_accuracy.png')
            )
            
            # 학습 손실값 플로팅
            plot_metrics(
                logger, 
                'train/Loss',
                os.path.join(result_dir, 'train_loss.png')
            )
            
            print(f"\n메트릭 그래프가 저장되었습니다: {result_dir}")
    return


def make_server(model):
    server = Server(model)
    return server


def make_client(model, data_split):
    client_id = torch.arange(cfg['num_clients'])
    client = [None for _ in range(cfg['num_clients'])]
    for m in range(len(client)):
        client[m] = Client(client_id[m], model, {'train': data_split['train'][m], 'test': data_split['test'][m]})
    return client


def train_client(batchnorm_dataset, server_dataset_train, client_dataset, server, client, optimizer, metric, logger, epoch):
    logger.safe(True)
    num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
    client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
    for i in range(num_active_clients):
        client[client_id[i]].active = True
        # 참여 빈도 업데이트 추가
        cfg['participant_frequency'][client_id[i]] += 1
        client[client_id[i]].participant_frequency = cfg['participant_frequency'][client_id[i]]

    server.distribute(client, batchnorm_dataset)
    num_active_clients = len(client_id)
    start_time = time.time()
    lr = optimizer.param_groups[0]['lr']
    for i in range(num_active_clients):
        m = client_id[i]
        dataset_m = separate_dataset(client_dataset, client[m].data_split['train'])
        if 'batch' not in cfg['loss_mode'] and 'frgd' not in cfg['loss_mode'] and 'fmatch' not in cfg['loss_mode']:
            dataset_m = client[m].make_dataset(dataset_m, metric, logger)
        if dataset_m is not None:
            client[m].active = True
            client[m].train(dataset_m, lr, metric, logger)
            
            # 서버 데이터로 평가 및 DB 저장 추가
            data_loader_server_train = make_data_loader({'train': server_dataset_train}, 'server')['train']
            model = eval('models.{}()'.format(cfg['model_name']))
            model.load_state_dict(client[m].model_state_dict)
            model = model.to(cfg["device"])
            server_evaluation = test(data_loader_server_train, model, metric, logger, epoch)
            client[m].sqlite_insert_data(cfg['client_db_path'], epoch, server_evaluation)
        else:
            client[m].active = False
        if i % int((num_active_clients * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (i + 1)
            epoch_finished_time = datetime.timedelta(seconds=_time * (num_active_clients - i - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['global']['num_epochs'] - epoch) * _time * num_active_clients))
            exp_progress = 100. * i / num_active_clients
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch (C): {}({:.0f}%)'.format(epoch, exp_progress),
                             'Learning rate: {:.6f}'.format(lr),
                             'ID: {}({}/{})'.format(client_id[i], i + 1, num_active_clients),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    return


def train_server(dataset, server, optimizer, metric, logger, epoch):
    logger.safe(True)
    start_time = time.time()
    lr = optimizer.param_groups[0]['lr']
    server.train(dataset, lr, metric, logger)
    _time = (time.time() - start_time)
    epoch_finished_time = datetime.timedelta(seconds=round((cfg['global']['num_epochs'] - epoch) * _time))
    info = {'info': ['Model: {}'.format(cfg['model_tag']),
                     'Train Epoch (S): {}({:.0f}%)'.format(epoch, 100.),
                     'Learning rate: {:.6f}'.format(lr),
                     'Epoch Finished Time: {}'.format(epoch_finished_time)]}
    logger.append(info, 'train', mean=False)
    print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    return


def test(data_loader, model, metric, logger, epoch):
    logger.reset()
    labels = []
    output_logit = []
    logger.reset()
    logger.safe(True)
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            labels       += input['target'].cpu().numpy().tolist()
            output_logit += output['target'].cpu().numpy().tolist()
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
        final_metrics = {}
        for m_name in metric.metric_name['test']:
            key_in_logger = f'test/{m_name}'
            if key_in_logger in logger.mean:
                final_metrics[m_name] = logger.mean[key_in_logger]
    logger.safe(False)
    json_output_logit = json.dumps(np.round(output_logit,4).tolist())
    final_metrics['output_logit'] = json_output_logit 
    json_labels = json.dumps(labels)
    final_metrics['labels'] = json_labels 
    return final_metrics


def plot_metrics(logger, metric_name, save_path):
    """
    학습 중 기록된 메트릭을 플로팅하는 함수
    Args:
        logger: 학습 중 사용된 logger 객체
        metric_name: 플로팅할 메트릭 이름 (예: 'test/Accuracy')
        save_path: 그래프 저장 경로
    """
    import matplotlib.pyplot as plt
    
    # 메트릭 데이터 추출
    rounds = range(1, len(logger.mean[metric_name]) + 1)
    values = logger.mean[metric_name]
    
    # 그래프 생성
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, values, marker='o', linestyle='-', linewidth=2)
    plt.title(f'{metric_name} over Rounds')
    plt.xlabel('Round')
    plt.ylabel(metric_name.split('/')[-1])
    plt.grid(True)
    
    # 그래프 저장
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    main()
