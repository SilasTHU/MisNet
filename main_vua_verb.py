import os
import os.path as osp
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
    parser = argparse.ArgumentParser(description='Train on VUA Verb dataset')
    parser.add_argument('--cfg', type=str, default='./configs/vua_verb.yaml', metavar="FILE",
                        help='path to config file')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids')
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--log', default='vua_verb', type=str)
    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return config


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sys.stdout = Logger(osp.join(args.TRAIN.output, f'{args.log}.txt'))
    print(args)
    set_random_seeds(args.seed)

    # load model
    plm = load_plm(args)
    model = Model(args=args, plm=plm)
    model.cuda()

    processor = Verb_Processor(args)

    if args.eval_mode:
        print("Evaluate only")
        processor = Verb_Processor(args)
        model.load_state_dict(torch.load('./checkpoints/best_vua_verb.pth'))

        test_data = processor.get_test_data()
        test_set = dataset(test_data)
        test_loader = DataLoader(test_set, batch_size=args.TRAIN.val_batch_size, shuffle=False, collate_fn=collate_fn)

        print('-------test-------')
        val(model, test_loader)
        return

    # training mode
    train_data = processor.get_train_data()
    val_data = processor.get_val_data()
    test_data = processor.get_test_data()

    train_set = dataset(train_data)
    val_set = dataset(val_data)
    test_set = dataset(test_data)

    train_loader = DataLoader(train_set, batch_size=args.TRAIN.train_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.TRAIN.val_batch_size, shuffle=False,  collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.TRAIN.val_batch_size, shuffle=False,  collate_fn=collate_fn)

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

    best_f1 = 0
    for epoch in range(args.TRAIN.train_epochs):
        print('===== Start training: epoch {} ====='.format(epoch + 1))
        train(epoch, model, loss_fn, optimizer, train_loader)
        a, p, r, f1 = val(model, val_loader)
        val(model, test_loader)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), './best_vua_verb.pth')


if __name__ == '__main__':
    args = parse_option()
    main(args)