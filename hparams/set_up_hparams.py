class HparamsBase(dict):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value


class HparamsAbsorbing(HparamsBase):
    TIME_RESOLUTION = 16
    DURATION_SEC = 16

    def __init__(self, dataset):
        self.sampler = "absorbing"
        self.loss_type = "reweighted_elbo"
        self.sample_type = "diffusion"
        self.mask_schedule = "random"
        self.total_steps = 256
        self.sample_steps = 256
        self.attn_pdrop = 0.
        self.embd_pdrop = 0.
        self.resid_pdrop = 0.
        self.temp = 1.0
        self.steps_per_eval = 5000
        self.steps_per_checkpoint = 10000
        self.steps_per_log = 10
        self.steps_per_update_ema = 10
        self.steps_per_sample = 5000
        self.load_step = 0#20000
        self.log_dir = 'log'
        self.load_dir = self.log_dir
        self.visdom_port = 8097

        super().__init__(dataset)
        if self.dataset == "Lakh":
            self.n_samples = 24
            self.bert_n_emb = 512  # todo: try concat embedding instead of summing over them
            self.bert_n_head = 8
            self.bert_n_layers = 24
            self.block_size = 256
            self.lr = 1e-3
            self.warmup_iters = 10000
            self.codebook_size = (90, 90, 512)
            self.latent_shape = (256, 3)
            self.train_steps = 700000
            self.validation_set_size = 0.05
        else:
            raise KeyError(f"Defaults not defined for multinomial diffusion model on dataset: {self.dataset}")


def apply_parser_values_to_H(H, args):
    # NOTE default args in H will be overwritten by any default parser args
    args = args.__dict__
    for arg in args:
        if args[arg] is not None:
            H[arg] = args[arg]

    return H