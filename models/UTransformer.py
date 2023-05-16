import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfOrCrossAttention(nn.Module):
    """
    copied from https://github.com/samb-t/unleashing-transformers
    added cross attn
    """

    def __init__(self, H):
        super().__init__()
        assert H.bert_n_emb % H.bert_n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(H.bert_n_emb, H.bert_n_emb)
        self.query = nn.Linear(H.bert_n_emb, H.bert_n_emb)
        self.value = nn.Linear(H.bert_n_emb, H.bert_n_emb)
        # regularization
        self.attn_drop = nn.Dropout(H.attn_pdrop)
        self.resid_drop = nn.Dropout(H.resid_pdrop)
        # output projection
        self.proj = nn.Linear(H.bert_n_emb, H.bert_n_emb)
        self.n_head = H.bert_n_head

    '''
    cross attn if y specified
    '''
    def forward(self, x, s=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim

        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = s if s is not None else x
        v = self.value(v).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_drop(self.proj(y))

        if s is not None:
            y = torch.sigmoid(y) * s + x

        return y


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, H):
        super().__init__()
        self.ln1 = nn.LayerNorm(H.bert_n_emb)
        self.ln2 = nn.LayerNorm(H.bert_n_emb)
        self.attn = CausalSelfOrCrossAttention(H)
        self.mlp = nn.Sequential(
            nn.Linear(H.bert_n_emb, 4 * H.bert_n_emb),
            nn.GELU(),  # nice
            nn.Linear(4 * H.bert_n_emb, H.bert_n_emb),
            nn.Dropout(H.resid_pdrop),
        )

    def forward(self, x, y=None):

        attn = self.attn(self.ln1(x), y)
        x = x + attn
        x = x + self.mlp(self.ln2(x))

        return x


class UpBlock(nn.Module):
    def __init__(self, H, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.blocks = nn.Sequential(*[Block(H) for _ in range(H.layers_per_level)])
        self.scale_seq = nn.ConvTranspose1d(H.bert_n_emb, H.bert_n_emb, H.conv_width, H.conv_width // 2, H.conv_width // 2 - 1)
        self.pos_emb = PositionalEncoding(H.bert_n_emb, max_len=seq_len)

    def forward(self, x, y):
        x = self.pos_emb(x)
        y = self.pos_emb(y)
        x = self.scale_seq(x.transpose(1, 2)).transpose(1, 2)
        x = x + y
        x = torch.relu(x)
        #x = self.blocks[0](x, y)#cross attn
        x = self.blocks(x)
        #x = self.blocks[1:](x)
        #n, C_in, L_in
        return x


class DownBlock(nn.Module):

    def __init__(self, H, seq_len, ):
        super().__init__()
        self.seq_len = seq_len
        self.blocks = nn.Sequential(*[Block(H) for _ in range(H.layers_per_level)])
        p = int(H.conv_width / 2 - 0.5)
        self.scale_seq = nn.Conv1d(H.bert_n_emb, H.bert_n_emb, H.conv_width, H.conv_width // 2, p)
        self.pos_emb = PositionalEncoding(H.bert_n_emb, max_len=seq_len)

    def forward(self, x, y=None):
        x = self.pos_emb(x)
        x = self.blocks(x)
        #n, C_in, L_in
        r = self.scale_seq(x.transpose(1, 2)).transpose(1, 2)
        r = torch.relu(r)
        return r, x




class UTransformer(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, H):
        super().__init__()

        self.vocab_size = [h + 1 for h in H.codebook_size]
        self.n_embd = H.bert_n_emb
        self.block_size = H.block_size
        self.n_layers = H.bert_n_layers
        self.codebook_size = H.codebook_size
        self.causal = H.sampler == 'autoregressive'
        if self.causal:
            self.vocab_size = H.codebook_size

        n_emb = [self.n_embd] if H.tracks == 'melody' else [self.n_embd // i for i in [4, 4, 2]]

        self.tok_emb = nn.ModuleList([nn.Embedding(vs, n_emb[i]) for i, vs in enumerate(self.vocab_size)])
        self.drop = nn.Dropout(H.embd_pdrop)

        # transformer
        self.down_blocks = nn.ModuleList(
            [DownBlock(H, self.block_size // 2 ** i) for i in range(self.n_layers // (H.layers_per_level * 2))])
        self.up_blocks = nn.ModuleList([UpBlock(H, self.block_size // 2 ** i) for i in
                                        reversed(range(self.n_layers // (H.layers_per_level * 2)))])
        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.head = nn.ModuleList([nn.Linear(self.n_embd, cs, bias=False) for cs in self.codebook_size])

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        else:
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, idx, t=None):
        # each index maps to a (learnable) vector
        token_embeddings = [t(idx[:, :, i]) for i, t in enumerate(self.tok_emb)]
        token_embeddings = torch.cat(token_embeddings, -1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # each position maps to a (learnable) vector

        x = token_embeddings
        x = self.drop(x)

        intermediates = []
        for block in self.down_blocks:
            x, intermediate = block(x)
            intermediates.append(intermediate)

        for block in self.up_blocks:
            x = block(x, intermediates.pop())

        x = self.ln_f(x)
        logits = [h(x) for h in self.head]

        return logits