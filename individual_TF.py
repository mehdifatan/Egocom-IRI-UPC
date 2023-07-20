import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.decoder import Decoder
from transformer.multihead_attention import MultiHeadAttention
from transformer.positional_encoding import PositionalEncoding
from transformer.pointerwise_feedforward import PointerwiseFeedforward
from transformer.encoder_decoder import EncoderDecoder
from transformer.encoder import Encoder
from transformer.encoder_layer import EncoderLayer
from transformer.decoder_layer import DecoderLayer
from transformer.batch import subsequent_mask
import numpy as np
import scipy.io
import os

import copy
import math

class IndividualTF(nn.Module):
    def __init__(self, src, trg, dec_out_size, N=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1):
        super(IndividualTF, self).__init__()
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model)
        ff = PointerwiseFeedforward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn),
                                 c(ff), dropout), N),
            nn.Sequential(LinearEmbedding(src,d_model), c(position)),
            nn.Sequential(LinearEmbedding(trg,d_model), c(position)),
            Generator(d_model, trg))

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, *input):
        return self.model.generator(self.model(*input))

class LinearEmbedding(nn.Module):
    def __init__(self, inp_size,d_model):
        super(LinearEmbedding, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(inp_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, out_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, out_size)

    def forward(self, x):
        x=self.proj(x)   
        return x


