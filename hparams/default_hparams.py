class HparamsBase(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value

    def apply_parser_values(self, parser):
        # NOTE default args will be overwritten by any default parser args
        parser = parser.__dict__
        for arg in parser:
            if parser[arg] is not None:
                self[arg] = parser[arg]


class HparamsAbsorbing(HparamsBase):

    def __init__(self, parser):
        super().__init__()

        self.sampler = "absorbing"
        self.loss_type = "reweighted_elbo"
        self.sample_type = "diffusion"
        self.mask_schedule = "random"
        self.sample_schedule = "random"#"barwise"
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
        self.sampling_batch_size = 24
        self.bert_n_emb = 512
        self.bert_n_head = 8
        self.bert_n_layers = 24
        self.lr = 5e-4
        self.warmup_iters = 10000
        self.validation_set_size = 0.05
        self.augment = True

        self.apply_parser_values(parser)

        self.NOTES = self.bars * 16
        self.total_steps = self.NOTES
        self.sample_steps = self.NOTES
        self.block_size = self.NOTES
        self.tracks = self.tracks
        self.log_dir = f'log_{self.model}_{self.tracks}_{self.NOTES}'
        self.load_dir = self.log_dir
        self.codebook_size = (90, ) if self.tracks == 'melody' else (90, 90, 512)
        self.latent_shape = (self.NOTES, len(self.codebook_size))
        self.load_optim = self.load_step != 0


class HparamsAbsorbingConv(HparamsAbsorbing):
    def __init__(self, parser):
        super().__init__(parser)
        self.bert_n_emb = 512
        self.conv_layers = 1
        self.conv_len = 4


class HparamsHierarchTransformer(HparamsAbsorbing):
    def __init__(self, parser):
        super().__init__(parser)
        self.sub_seq_len = 32
        self.bert_n_emb = 512
        self.upper_bert_n_emb = 512
        self.augment = False


class HparamsUTransformer(HparamsAbsorbing):
    def __init__(self, parser):
        super().__init__(parser)
        self.layers_per_level = 2
        self.bert_n_emb = 512
        self.conv_width = 4
        self.augment = False
