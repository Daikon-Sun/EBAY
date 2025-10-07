import argparse
import torch
torch.set_printoptions(precision=7)

from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SFNN')

    # basic config
    parser.add_argument('--model_id', type=str, required=True, help='model id')
    parser.add_argument('--model', type=str, required=True)

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, help='root path of the data file')
    parser.add_argument('--data_path', type=str, help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=1, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, help='prediction sequence length')

    # model define
    parser.add_argument('--n_layers', type=int, help='num of heads')
    parser.add_argument('--dropout', type=float, help='dropout')
    parser.add_argument('--mixer', action='store_true')
    parser.add_argument('--layernorm', action='store_true')
    parser.add_argument('--need_norm', action='store_true')
    parser.add_argument('--norm_len', type=int)

    # optimization
    parser.add_argument('--train_epochs', type=int, help='train epochs')
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--batch_size', type=int, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, help='optimizer learning rate')
    parser.add_argument('--min_lr', type=float, default=5e-5)
    parser.add_argument('--loss_fn', type=str, help='options: [MSE, MAE]')

    parser.add_argument('--beg_ratio', type=float)
    parser.add_argument('--end_ratio', type=float)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--adapt_iters', type=int)
    parser.add_argument('--test_batch_size', type=int)
    parser.add_argument('--adapt_lr', type=float)


    args = parser.parse_args()

    assert args.loss_fn in ['MSE', 'MAE'], 'Loss function not recognized'

    if args.data_path in ['ETTh1.csv', 'ETTh2.csv', 'traffic.csv', 'electricity.csv']:
        args.period = 168
    elif args.data_path in ['solar.csv', 'weather.csv']:
        args.period = 144
    elif args.data_path in ['ETTm1.csv', 'ETTm2.csv']:
        args.period = 96
    elif args.data_path in ['ILI.csv']:
        args.period = 52
    elif args.data_path in ['exchange_rate.csv']:
        args.period = 5
    else:
        assert False, 'Data path not recognized'

    if args.norm_len is None:
        args.norm_len = args.period

    print('Args in experiment:')
    print(args)

    Exp = Exp_Long_Term_Forecast

    setting = f'{args.model_id}_{args.model}_sl{args.seq_len}_pl{args.pred_len}_nl{args.n_layers}' \
        f'_lr{args.learning_rate}_bs{args.batch_size}_do{args.dropout}_wd{args.weight_decay}_loss{args.loss_fn}_beg_ratio{args.beg_ratio}_end_ratio{args.end_ratio}'

    exp = Exp(args)
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)
