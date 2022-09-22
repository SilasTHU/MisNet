import os
import os.path as osp
import random
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import torch.nn as nn

from model import Model
from data_loader import dataset, collate_fn
from data_process import Verb_Processor
from utils import Logger
from configs.default import get_config
from train_val import train, val, load_plm, set_random_seeds


def parse_option():
    parser = argparse.ArgumentParser(description='Train on MOH-X dataset, do cross validation')
    parser.add_argument('--cfg', type=str, default='./configs/mohx.yaml', metavar="FILE",
                        help='path to config file')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids')
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--log', default='mohx', type=str)
    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return config


def get_kfold_data(k, i, raw_data):
    fold_size = len(raw_data) // k

    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        val_raw = raw_data[val_start:val_end]
        tr_raw = raw_data[0:val_start] + raw_data[val_end:]
    else:
        val_raw = raw_data[val_start:]
        tr_raw = raw_data[0:val_start]

    return tr_raw, val_raw


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sys.stdout = Logger(osp.join(args.TRAIN.output, f'{args.log}.txt'))
    print(args)
    set_random_seeds(args.seed)

    # prepare train-val datasets and dataloaders
    processor = Verb_Processor(args)
    data = processor.get_mohx()
    random.shuffle(data)

    accs = 0
    pres = 0
    recs = 0
    f1s = 0
    for i in range(10):
        print('*' * 20, f"training on fold #{i + 1}", '*' * 20)
        train_data, test_data = get_kfold_data(10, i, data)
        train_set = dataset(train_data)
        test_set = dataset(test_data)

        train_loader = DataLoader(train_set, batch_size=args.TRAIN.train_batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_set, batch_size=args.TRAIN.val_batch_size, shuffle=False,  collate_fn=collate_fn)

        # load model
        plm = load_plm(args)
        model = Model(args=args, plm=plm)
        model.cuda()

        # prepare optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.TRAIN.lr)

        # prepare loss function
        loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([1, args.TRAIN.class_weight]).cuda())

        best_acc = -1
        best_pre = -1
        best_rec = -1
        best_f1 = -1
        for epoch in range(args.TRAIN.train_epochs):
            print('===== Start training: epoch {} ====='.format(epoch + 1))
            train(epoch, model, loss_fn, optimizer, train_loader)
            t_a, t_p, t_r, t_f1 = val(model, test_loader)
            if t_f1 > best_f1:
                best_acc, best_pre, best_rec, best_f1 = t_a, t_p, t_r, t_f1
        accs += best_acc
        pres += best_pre
        recs += best_rec
        f1s += best_f1

    print('average result:')
    print(accs / 10)
    print(pres / 10)
    print(recs / 10)
    print(f1s / 10)


if __name__ == '__main__':
    args = parse_option()
    main(args)
