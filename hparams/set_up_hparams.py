import argparse
from .default_hparams import HparamsAbsorbing, HparamsAbsorbingConv


def add_common_args(parser):
    parser.add_argument("--model", type=str, default="conv_transformer")
    parser.add_argument("--tracks", type=str, default="melody")
    parser.add_argument("--amp", const=True, action="store_const", default=False)
    parser.add_argument("--ema", const=True, action="store_const", default=True)
    parser.add_argument("--load_dir", type=str, default=None)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--bars", type=int, default=64)
    parser.add_argument("--dataset_path", type=str, default='data/lakh_melody_64_1MIO.npy')


def add_train_args(parser):
    add_eval_args(parser, 1)

    parser.add_argument("--ema_beta", type=float, default=0.995)
    parser.add_argument("--load_optim", const=True, action="store_const", default=False)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps_per_update_ema", type=int, default=10)
    parser.add_argument("--steps_per_log", type=int, default=10)
    parser.add_argument("--steps_per_eval", type=int, default=10000)
    parser.add_argument("--steps_per_sample", type=int, default=5000)
    parser.add_argument("--steps_per_checkpoint", type=int, default=10000)
    parser.add_argument("--train_steps", type=int, default=100000000)
    parser.add_argument("--show_samples", type=int, default=32)


def add_eval_args(parser, num_evals=5):
    parser.add_argument("--mode", type=str, default='unconditional')
    parser.add_argument("--sampling_batch_size", type=int, default=256)
    parser.add_argument("--gap_start", type=int, default=-1)
    parser.add_argument("--gap_end", type=int, default=-1)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--evals_per_batch", type=int, default=10)
    parser.add_argument("--num_evals", type=int, default=num_evals)


def get_sampler_hparams(mode):
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    if mode == 'train':
        add_train_args(parser)
    elif mode == 'eval':
        add_eval_args(parser)
    elif mode == 'sample':
        pass


    parser_args = parser.parse_args()

    if parser_args.model == 'transformer':
        H = HparamsAbsorbing(parser_args)
    else:
        parser_args.model = 'conv_transformer'
        H = HparamsAbsorbingConv(parser_args)


    return H