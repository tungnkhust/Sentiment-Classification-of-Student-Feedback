import os
import time
import torch
import json
import yaml
import pandas as pd
import argparse
from argparse import Namespace
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch_core.dictionary import build_vocab, get_label_dict
from torch_core.utils import get_n_params, load_w2v
from torch_core.utils import make_train_state, update_train_state
from torch_core.dataset import LabelDataset
from torch_core.modules.models import BasicModel
from torch_core.dictionary import Dictionary, TokenDictionary
from load_data import load_data


def train(
        model,
        train_dataset,
        val_dataset,
        configs,
        train_state
    ):
    """
    :param model: model pytorch
    :param train_dataset:
    :param val_dataset:
    :param args:
    :param train_state:
    :return:
    """

    model_dir = configs['MODEL']['MODEL_PATH']['model_dir'] + '/' +model.__class__.__name__
    checkpoint_path = model_dir + '/best_model.pth'

    if os.path.exists(model_dir) is False:
        os.mkdir(model_dir)

    train_cf = configs['TRAIN']
    batch_size = train_cf['batch_size']
    lr = train_cf['learning_rate']
    num_epochs = train_cf['num_epochs']
    clipping_gradient = train_cf['clipping_gradient']
    show_step = train_cf['show_step']
    if train_cf['checkpoint'] and os.path.exists(checkpoint_path):
        print("load model from " + checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device: ", device)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model.to(device)

    # set optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=num_epochs)

    # train model
    for epoch in range(num_epochs):
        start_time = time.time()
        print(32*'_' + ' '*9 + 31*'_')
        print(32*'_' + 'EPOCH {:3}'.format(epoch+1) + 31*'_')
        train_state['epoch_index'] = epoch
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        train_loss = 0.
        model.train()
        for i, (x_vector, sent_vector, topic_vector, x_mask) in enumerate(train_loader):
            sent_vector.to(device)
            topic_vector.to(device)
            optimizer.zero_grad()
            out = model(x_vector.to(device), x_mask.to(device))
            sent_out = out[0]
            topic_out = out[1]
            loss = model.compute_loss(sent_out, topic_out, sent_vector, topic_vector)
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_gradient)

            optimizer.step()
            train_loss += (loss.item() - train_loss) / (i + 1)
            if show_step:
                if (i + 1) % 10 == 0:
                    print("|{:4}| train_loss = {:.4f}".format(i + 1, loss.item()))

        # evaluate
        train_state['train_loss'].append(train_loss)
        optimizer.zero_grad()
        val_loss, sent_report, topic_report = model.evaluate(val_dataset, batch_size=batch_size)

        # scheduler.step(val_loss)
        print('Val loss: {:.4f}'.format(val_loss))
        scheduler.step()

        train_state['val_loss'].append(val_loss)
        ## update train state
        train_state = update_train_state(model, train_state)

        if train_state['stop_early']:
            print('Stop early.......!')
            break

    return model, train_state


def get_basic_model(configs, device='cpu', w2v=None):
    model = BasicModel(
        n_sent=configs['n_sent'],
        n_topic=configs['n_topic'],
        vocab_size=configs['vocab_size'],
        padding_idx=configs['padding_idx'],
        embed_size=configs['embed_size'],
        hidden_size=configs['hidden_size'],
        n_rnn_layers=configs['n_rnn_layers'],
        dropout=configs['dropout'],
        att_method=configs['att_method'],
        device=device,
        word2vec=w2v
    )

    return model

if __name__ == '__main__':
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-cp", default='configs/basic_configs.yaml', help="path to configs")
    args = parser.parse_args()

    with open(args.config_path, 'r') as pf:
        configs = yaml.load(pf, Loader=yaml.FullLoader)

    model_cf = configs['MODEL']
    train_cf = configs['TRAIN']
    normal_args = configs['DATA']['NORMALIZE']
    raw_path = configs['DATA']['RAW']['data_path']
    gen_cf = configs['DATA']['GENERAL']
    vocab_path = gen_cf['vocab_path']
    max_seq_len = gen_cf['max_seq_len']
    max_char_len = gen_cf['max_char_len']

    # sent2idx and topic2idx
    sent2idx = configs['DATA']['LABEL']['sentiment']
    topic2idx = configs['DATA']['LABEL']['topic']

    # path configure
    data_path = gen_cf['data_path']
    train_name = gen_cf['train_name']
    val_name = gen_cf['val_name']
    test_name = gen_cf['test_name']

    # load data
    train_df, val_df, test_df = load_data(data_path, train_name, test_name, val_name)

    # load vocab
    if os.path.exists(vocab_path) is True:
        seq_vocab = TokenDictionary.load(vocab_path)
    else:
        raise Exception("File {} don't exists".format(vocab_path))

    # create sentiment dictionary and topic dictionary
    sent_dict = Dictionary(item2idx=sent2idx)
    topic_dict = Dictionary(item2idx=topic2idx)
    print(sent_dict.item2idx)

    # Create train, val, test datasets.
    train_dataset = LabelDataset(train_df, seq_vocab, sent_dict, topic_dict, max_seq_len=max_seq_len)
    val_dataset = LabelDataset(val_df, seq_vocab, sent_dict, topic_dict, max_seq_len=max_seq_len)
    test_dataset = LabelDataset(test_df, seq_vocab, sent_dict, topic_dict, max_seq_len=max_seq_len)

    # Load word2vec embedding
    w2v_path = configs['MODEL']['MODEL_PATH']['w2v_path']
    embed_size = configs['MODEL']['embed_size']
    if w2v_path is not None:
        w2c_path = w2v_path + '/viglove_{}D.txt'.format(embed_size)
        print('Load word embedding from {}.'.format(w2c_path))
        w2v = load_w2v(w2c_path, seq_vocab.item2idx, embed_size=embed_size)
    else:
        w2v = None

    n_sent = len(sent_dict)
    n_topic = len(topic_dict)
    padding_idx = seq_vocab.padding_idx
    vocab_size = len(seq_vocab)

    model_cf['n_sent'] = n_sent
    model_cf['n_topic'] = n_topic
    model_cf['padding_idx'] = padding_idx
    model_cf['vocab_size'] = vocab_size

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = get_basic_model(model_cf, device=device, w2v=w2v)

    model_dir = configs['MODEL']['MODEL_PATH']['model_dir'] + '/' + model.__class__.__name__
    lr = configs['TRAIN']['learning_rate']

    if os.path.exists(model_dir) is False:
        os.mkdir(model_dir)
    with open(model_dir + '/configs.json', 'w') as pf:
        json.dump(configs, pf)

    print('_'*72)
    print()
    print(model)
    print('_'*72)
    print()
    print('Num training samples  :', len(train_df))
    print('Num validation samples:', len(val_df))
    print('Num test samples      :', len(test_df))
    print('Num sentiment         :', n_sent)
    print('Num topic             :', n_topic)
    print("Max sequence length   :", max_seq_len)
    print('Total parameter       :', get_n_params(model))
    print('_'*72)
    print()


    try:
        train_state = make_train_state(args, model_dir, lr)
        for i in range(train_cf['n_retrain']):
            model, train_state = train(model, train_dataset, val_dataset, configs, train_state)
            args.checkpoint = True

    except KeyboardInterrupt as e:
        print(e)
        print('\nSave last model at {}\n'.format(args.model_dir + '/final_model.pth'))
        torch.save(model.state_dict(), args.model_dir + '/final_model.pth')
    
    finally:
        model = get_basic_model(model_cf, device=device)
        print("Load best model to evaluate......!")
        model.from_pretrained(model_dir + '/best_model.pth')
        model.to(device)
        print(31*'_' + '          ' + 31*'_')
        print(31*'_' + 'EVALUATION' + 31*'_')
        print()
        val_loss, sent_report, topic_report = model.evaluate(test_dataset, batch_size=configs['TRAIN']['batch_size'])
