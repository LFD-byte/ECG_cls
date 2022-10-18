# coding=utf-8
import os
import random

Arrhythmia = ['164889003', '164890007', '713422000', '426177001', '426783006', '427084000', '426761007']
Arrhythmia_Dict = {'164889003': '0', '164890007': '1', '713422000': '2', '426177001': '3', '426783006': '4', '427084000': '5',
                   '426761007': '6'}


def get_label(path):
    '''
    :param path: path of ecg file(contain .hea, .mat)
    :return: the label
    '''
    with open(path, 'r') as f:
        JS_hea = f.readlines()
    label = JS_hea[-4][5:-1].split(',')
    labels = ''
    labels = [Arrhythmia_Dict[l] for l in label if l in Arrhythmia][0]
    if len(labels) == 0:
        labels = '0'
    return labels


def get_index(path):
    arrh_dir = os.listdir(path)
    index_ecg = []
    for arrh in arrh_dir:
        if arrh.endswith('hea'):
            index_ecg.append(' '.join([path + '/' + arrh.replace('.hea', '.mat'), get_label(path + '/' + arrh)]))
    return index_ecg


def save_index(data, path):
    with open(path, 'a+') as f:
        f.write('\n'.join(data) + '\n')


def file_clear(path):
    with open(path, 'w') as f:
        f.truncate(0)


def index_make(path):
    save_path_train = 'data/index_ecg_train.txt'
    save_path_val = 'data/index_ecg_val.txt'

    file_clear(save_path_train)
    file_clear(save_path_val)

    ecg_dirs = os.listdir(path)
    for ecg_dir in ecg_dirs:
        if os.path.isfile(path + '/' + ecg_dir):
            continue
        ecg_index = get_index(path + '/' + ecg_dir)
        random.shuffle(ecg_index)
        save_index(ecg_index[:int(len(ecg_index) * 0.7)], save_path_train)
        save_index(ecg_index[int(len(ecg_index) * 0.7) + 1:], save_path_val)


if __name__ == '__main__':
    path = 'data'
    index_make(path)