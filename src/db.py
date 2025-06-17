import os
import sqlite3
from utils import softmax
import numpy as np
import json
def create_train_database(db_name):
    db_dir = os.path.dirname(db_name)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)

    conn = sqlite3.connect(db_name, timeout=5)
    cursor = conn.cursor()
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS fed_rl_data (
            round INTEGER,
            client_number INTEGER,
            threshold REAL,
            local_epoch INTEGER,
            participant_frequency INTEGER, 
            local_loss REAL,
            local_label_ratio REAL,
            local_msp REAL,
            lm_ld_client_labels TEXT,
            lm_ld_client_output_logit TEXT,
            gm_ld_client_labels TEXT,
            gm_ld_output_logit TEXT,
            server_loss REAL,
            server_acc REAL,
            server_labels TEXT,
            server_output_logit TEXT)
        ''')
    conn.commit()
    conn.close()



def create_test_database(db_name):
    conn = sqlite3.connect(db_name, timeout=5)
    cursor = conn.cursor()
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS fed_rl_data (
            round INTEGER,
            test_loss REAL,
            test_acc REAL,
            test_labels TEXT,
            test_output_logit TEXT,
            agg_server_loss REAL,
            agg_server_acc REAL,
            agg_server_labels TEXT,
            agg_server_output_logit TEXT,
            agg_ft_server_loss REAL,
            agg_ft_server_acc REAL,
            agg_ft_server_labels TEXT,
            agg_ft_server_output_logit TEXT)
        ''')
    conn.commit()
    conn.close()

def sqlite_insert_test_data(db_name,_round,test_evaluation,server_eval,server_eval2):
    data =  (_round, 
             test_evaluation['Loss'],
             test_evaluation['Accuracy'],
             test_evaluation['labels'],
             test_evaluation['output_logit'],
             server_eval['Loss'],
             server_eval['Accuracy'],
             server_eval['labels'],
             server_eval['output_logit'],
             server_eval2['Loss'],
             server_eval2['Accuracy'],
             server_eval2['labels'],
             server_eval2['output_logit'])
    
    conn = sqlite3.connect(db_name,timeout=5)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO fed_rl_data (round, 
                                 test_loss,
                                 test_acc,
                                 test_labels,
                                 test_output_logit,
                                 agg_server_loss,
                                 agg_server_acc,
                                 agg_server_labels,
                                 agg_server_output_logit,
                                 agg_ft_server_loss,
                                 agg_ft_server_acc,
                                 agg_ft_server_labels,
                                 agg_ft_server_output_logit)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', data)
    conn.commit()
    conn.close()

def get_global_model_result(db_name, _round):
    conn = sqlite3.connect(db_name, timeout=5)
    cursor = conn.cursor()
    cursor.execute(
        '''
        SELECT agg_ft_server_loss, agg_ft_server_acc FROM fed_rl_data WHERE round = ?
        ''', (_round,)
    )
    result = cursor.fetchone()
    conn.close()
    if result:
        return {'agg_ft_server_loss': result[0], 'agg_ft_server_acc': result[1]}
    return None
    
def get_client_recent_info(client_db_path,client_number):
    conn = sqlite3.connect(client_db_path,timeout=5)
    cursor = conn.cursor()
    cursor.execute(f'''
        SELECT round, local_loss, threshold, server_loss, local_epoch,local_msp
        FROM fed_rl_data
        WHERE client_number = {client_number}
        ORDER BY round DESC
        LIMIT 1
    ''')
    raw = cursor.fetchone()
    if raw is None:
        _round, local_loss,threshold,server_loss,local_epoch,local_msp = 0,0,0,0,0,0
    else:
        _round, local_loss,threshold,server_loss,local_epoch,local_msp = raw 
    conn.close()
    return {'_round':_round,
            'local_loss':local_loss,
            'threshold':threshold,
            'server_loss':server_loss,
            'local_epoch':local_epoch,
            'local_msp':local_msp}

def get_client_server_output_softmax_sum(client_db_path,client_number):
    conn = sqlite3.connect(client_db_path,timeout=5)
    cursor = conn.cursor()
    cursor.execute(f'''
        SELECT server_output_logit
        FROM fed_rl_data
        WHERE client_number = {client_number}
    ''')
    server_output_logit = cursor.fetchall()
    conn.close()
    client_output_softmax_mean = []
    for logit in server_output_logit:
        client_output_softmax_mean.append(softmax(np.array(json.loads(logit[0])).astype(float)).mean(axis=0))
    return np.mean(client_output_softmax_mean,axis=0)

def has_client_training_log(client_db_path,client_number):
    conn = sqlite3.connect(client_db_path,timeout=5)
    cursor = conn.cursor()
    cursor.execute(f'''
        SELECT Count(round)
        FROM fed_rl_data
        WHERE client_number = {client_number}
    ''')
    raw = cursor.fetchone()
    conn.close()
    return raw[0]>0