from yacs.config import CfgNode as CN

_C = CN()

"""
    data settings
"""
_C.DATA = CN()
_C.DATA.data_dir = './data'
# tokens separated by these punctuations can mark a context
_C.DATA.sep_puncs = [',', ' ,', '?', ' ?', ';', ' ;', '.', ' .', '!', ' !', '</s>', '[SEP]']
# whether to use pos tag
_C.DATA.use_pos = True
# whether to use example sentence to trigger target word basic meaning
_C.DATA.use_eg_sent = True
# whether to use context feature
_C.DATA.use_context = True
# the pretrained language model to use. Please pre-download. default is RoBERTa.
_C.DATA.plm = './roberta-base'
# the max length of the left input.
_C.DATA.max_left_len = 150
# the max length of the right input.
_C.DATA.max_right_len = 70

"""
    model settings
"""
_C.MODEL = CN()
_C.MODEL.num_classes = 2
# embedding dim
_C.MODEL.embed_dim = 768
# use the average of the first and the last hidden layer of PLMs as word embeddings
_C.MODEL.first_last_avg = True
# drop out rate
_C.MODEL.dropout = 0.2
# number of attention heads
_C.MODEL.num_heads = 12
_C.MODEL.cat_method = 'cat_abs_dot'  # cat, abs, dot, abs_dot, cat_dot, cat_abs, cat_abs_dot
'''
    training settings
'''
_C.TRAIN = CN()
_C.TRAIN.train_batch_size = 32
_C.TRAIN.val_batch_size = 32
_C.TRAIN.lr = 3e-5
_C.TRAIN.train_epochs = 15
_C.TRAIN.class_weight = 5
_C.TRAIN.warmup_epochs = 2
# the directory to save the training logs
_C.TRAIN.output = './data/logs'

_C.gpu = '0'
_C.seed = 4
# do eval only
_C.eval_mode = False
_C.log = 'log_test'


def update_config(config, args):
    config.defrost()

    print('=> merge config from {}'.format(args.cfg))
    config.merge_from_file(args.cfg)

    if args.gpu:
        config.gpu = args.gpu

    if args.eval:
        config.eval_mode = True

    if args.log:
        config.log = args.log

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    config = _C.clone()
    update_config(config, args)

    return config
