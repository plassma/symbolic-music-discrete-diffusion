import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
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
        self.causal = True if H.sampler == 'autoregressive' else False
        if self.causal:
            block_size = np.prod(H.latent_shape)
            mask = torch.tril(torch.ones(block_size, block_size))
            self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        present = torch.stack((k, v))
        if self.causal and layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.causal and layer_past is None:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, H):
        super().__init__()
        self.ln1 = nn.LayerNorm(H.bert_n_emb)
        self.ln2 = nn.LayerNorm(H.bert_n_emb)
        self.attn = CausalSelfAttention(H)
        self.mlp = nn.Sequential(
            nn.Linear(H.bert_n_emb, 4 * H.bert_n_emb),
            nn.GELU(),  # nice
            nn.Linear(4 * H.bert_n_emb, H.bert_n_emb),
            nn.Dropout(H.resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False):

        attn, present = self.attn(self.ln1(x), layer_past)
        x = x + attn
        x = x + self.mlp(self.ln2(x))

        if layer_past is not None or return_present:
            return x, present
        return x


def get_conv(layers, n_embd, conv_len, vocab_size):
    assert n_embd % (conv_len * layers) == 0 and conv_len % layers == 0, ""
    assert 1 <= layers <= 2
    result = []
    for _ in vocab_size:
        r = []
        for l in range(layers):
            r.append(nn.Conv1d(n_embd * (l + 1) // conv_len, n_embd // (layers - l), conv_len, conv_len // layers, layers // 2))  # todo: padding fishy?
            r.append(nn.ReLU())
        result.append(nn.Sequential(*r))

    return nn.ModuleList(result)

def get_transpose_conv(layers, n_embd, conv_len, vocab_size):
    result = []
    for _ in vocab_size:
        r = []
        for l in range(layers):
            r.append(nn.ConvTranspose1d(n_embd // (l + 1), n_embd * layers // ((l + 1) * conv_len), conv_len, conv_len // layers))
            r.append(nn.ReLU())
        result.append(nn.Sequential(*r))

    return nn.ModuleList(result)

class ConVormer(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, H):
        super().__init__()

        self.vocab_size = [h + 1 for h in H.codebook_size]
        self.n_embd = H.bert_n_emb
        self.conv_len = H.conv_len
        self.block_size = H.block_size // self.conv_len
        self.n_layers = H.bert_n_layers
        self.codebook_size = H.codebook_size
        self.causal = H.sampler == 'autoregressive'

        self.conv = get_conv(H.conv_layers, H.bert_n_emb, H.conv_len, H.codebook_size)
        self.transpose_conv = get_transpose_conv(H.conv_layers, H.bert_n_emb, H.conv_len, H.codebook_size)

        if self.causal:
            self.vocab_size = H.codebook_size

        self.tok_emb = nn.ModuleList([nn.Embedding(vs, self.n_embd // self.conv_len) for vs in self.vocab_size])
        self.emb_red = nn.Linear(self.n_embd * len(self.vocab_size), self.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size, self.n_embd))
        self.bar_pos_emb = nn.Parameter(torch.zeros(1, self.conv_len, self.n_embd // self.conv_len))
        self.start_tok = nn.Parameter(torch.zeros(1, 1, self.n_embd))
        self.drop = nn.Dropout(H.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(H) for _ in range(self.n_layers)])
        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.head = nn.ModuleList([nn.Linear(self.n_embd // self.conv_len, cs, bias=False) for cs in self.codebook_size])

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
        bpe = self.bar_pos_emb[0, torch.arange(self.conv_len).repeat(idx.shape[1] // self.conv_len)]
        token_embeddings = [t(idx[:, :, i]) + bpe for i, t in enumerate(self.tok_emb)]
        token_embeddings = [c(token_embeddings[i].transpose(1, 2)).transpose(1, 2) for i, c in enumerate(self.conv)]
        #act_fn is part of shared conv
        token_embeddings = torch.cat(token_embeddings,-1)
        token_embeddings = self.emb_red(token_embeddings)
        token_embeddings = F.relu(token_embeddings)

        if self.causal:
            token_embeddings = torch.cat(
                (self.start_tok.repeat(token_embeddings.size(0), 1, 1), token_embeddings),
                dim=1
            )

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # each position maps to a (learnable) vector

        position_embeddings = self.pos_emb[:, :t, :]

        x = token_embeddings + position_embeddings
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = [d(x.transpose(1, 2)).transpose(1, 2) for i, d in enumerate(self.transpose_conv)]

        if x[0].shape[1] > idx.shape[1]:
            pad = x[0].shape[1] - idx.shape[1]
            assert pad % 2 == 0
            pad //= 2
            x = [_x[:, pad:-pad] for _x in x]

        logits = [h(x[i]) for i, h in enumerate(self.head)]

        return logits

# class ConVormer(nn.Module):
#     """  the full GPT language model, with a context size of block_size """
#
#     def __init__(self, H):
#         super().__init__()
#
#         self.vocab_size = [h + 1 for h in H.codebook_size]
#         self.n_embd = H.bert_n_emb
#         self.conv_len = 4
#         self.block_size = H.block_size // self.conv_len
#         self.n_layers = H.bert_n_layers
#         self.codebook_size = H.codebook_size
#         self.causal = H.sampler == 'autoregressive'
#
#         self.conv = nn.Conv1d(self.n_embd // self.conv_len, self.n_embd, self.conv_len, self.conv_len)
#         self.deconv = nn.ConvTranspose1d(self.n_embd, self.n_embd // self.conv_len, self.conv_len, self.conv_len)
#
#         if self.causal:
#             self.vocab_size = H.codebook_size
#
#         self.tok_emb = nn.ModuleList([nn.Embedding(vs, self.n_embd // self.conv_len) for vs in self.vocab_size])
#         #self.emb_red = nn.Linear(1536, 512)
#         self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size, self.n_embd))
#         self.bar_pos_emb = nn.Parameter(torch.zeros(1, self.conv_len, self.n_embd // self.conv_len))
#         self.start_tok = nn.Parameter(torch.zeros(1, 1, self.n_embd))
#         self.drop = nn.Dropout(H.embd_pdrop)
#
#         # transformer
#         self.blocks = nn.Sequential(*[Block(H) for _ in range(self.n_layers)])
#         # decoder head
#         self.ln_f = nn.LayerNorm(self.n_embd)
#         self.head = nn.ModuleList([nn.Linear(self.n_embd // self.conv_len, cs, bias=False) for cs in self.codebook_size])
#
#     def get_block_size(self):
#         return self.block_size
#
#     def _init_weights(self, module):
#         if isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#         else:
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             if isinstance(module, nn.Linear) and module.bias is not None:
#                 module.bias.data.zero_()
#
#     def forward(self, idx, t=None):
#         # each index maps to a (learnable) vector
#         token_embeddings = [t(idx[:, :, i]) for i, t in enumerate(self.tok_emb)]
#         token_embeddings = torch.cat(token_embeddings,-1)# todo: try other combinations of embeddings (concat?)
#         #token_embeddings = self.emb_red(token_embeddings)
#         token_embeddings += self.bar_pos_emb[0, torch.arange(self.conv_len).repeat(idx.shape[1] // self.conv_len)]
#         token_embeddings = self.conv(token_embeddings.transpose(1, 2)).transpose(1, 2)
#         token_embeddings = F.relu(token_embeddings)
#         if self.causal:
#             token_embeddings = torch.cat(
#                 (self.start_tok.repeat(token_embeddings.size(0), 1, 1), token_embeddings),
#                 dim=1
#             )
#
#         t = token_embeddings.shape[1]
#         assert t <= self.block_size, "Cannot forward, model block size is exhausted."
#         # each position maps to a (learnable) vector
#
#         position_embeddings = self.pos_emb[:, :t, :]
#
#         x = token_embeddings + position_embeddings
#         x = self.drop(x)
#         for block in self.blocks:
#             x = block(x)
#         x = self.ln_f(x)
#         x = self.deconv(x.transpose(1, 2)).transpose(1, 2)
#         x = F.relu(x)
#         logits = [h(x) for h in self.head]
#
#         return logits