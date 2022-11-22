class HparamsBase(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value


class HparamsAbsorbing(HparamsBase):

    def __init__(self, bars, tracks='melody'):
        super().__init__()

        self.NOTES = bars * 16
        self.sampler = "absorbing"
        self.loss_type = "reweighted_elbo"
        self.sample_type = "diffusion"
        self.mask_schedule = "random"
        self.total_steps = self.NOTES
        self.sample_steps = self.NOTES
        self.attn_pdrop = 0.2
        self.embd_pdrop = 0.2
        self.resid_pdrop = 0.2
        self.temp = 1.0
        self.steps_per_eval = 10000
        self.steps_per_checkpoint = 10000
        self.steps_per_log = 100
        self.steps_per_update_ema = 10
        self.steps_per_sample = 5000
        self.load_step = 0
        self.load_optim = self.load_step != 0
        self.log_dir = f'log_{tracks}_{self.NOTES}'
        self.load_dir = self.log_dir
        self.visdom_port = 8097

        self.sampling_batch_size = 24
        self.bert_n_emb = 512
        self.bert_n_head = 8
        self.bert_n_layers = 24
        self.block_size = self.NOTES
        self.lr = 5e-4
        self.warmup_iters = 10000
        self.codebook_size = (90, ) if tracks == 'melody' else (90, 90, 512)
        self.latent_shape = (self.NOTES, 1)
        self.train_steps = 700000
        self.validation_set_size = 0.05


class HparamsAbsorbingConv(HparamsBase):

    def __init__(self, bars, tracks='melody'):
        super().__init__()
        self.NOTES = bars * 16
        self.sampler = "absorbing"
        self.loss_type = "reweighted_elbo"
        self.sample_type = "diffusion"
        self.mask_schedule = "random"
        self.total_steps = self.NOTES
        self.sample_steps = self.NOTES
        self.attn_pdrop = 0.2
        self.embd_pdrop = 0.2
        self.resid_pdrop = 0.2
        self.temp = 1.0
        self.steps_per_eval = 10000
        self.steps_per_checkpoint = self.steps_per_eval
        self.steps_per_log = 100
        self.steps_per_update_ema = 10
        self.steps_per_sample = 5000
        self.load_step = 440000
        self.load_optim = self.load_step != 0
        self.log_dir = f'log_{tracks}_{self.NOTES}'
        self.load_dir = self.log_dir
        self.visdom_port = 8097

        self.sampling_batch_size = 24
        self.bert_n_emb = 1024
        self.bert_n_head = 8
        self.bert_n_layers = 24
        self.block_size = self.NOTES
        self.lr = 1e-4
        self.warmup_iters = 10000
        self.codebook_size = (90, ) if tracks == 'melody' else (90, 90, 512)
        self.latent_shape = (self.NOTES, 1)
        self.train_steps = 700000
        self.validation_set_size = 0.05
