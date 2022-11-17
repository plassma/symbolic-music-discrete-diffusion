def add_training_args(parser):
    parser.add_argument("--amp", const=True, action="store_const", default=False)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--custom_dataset_path", type=str)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ema_beta", type=float, default=0.995)
    parser.add_argument("--ema", const=True, action="store_const", default=False)
    parser.add_argument("--load_dir", type=str, default="test")
    parser.add_argument("--load_optim", const=True, action="store_const", default=False)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--steps_per_update_ema", type=int, default=10)
    parser.add_argument("--train_steps", type=int, default=100000000)

def apply_parser_values_to_H(H, args):
    # NOTE default args in H will be overwritten by any default parser args
    args = args.__dict__
    for arg in args:
        if args[arg] is not None:
            H[arg] = args[arg]

    return H