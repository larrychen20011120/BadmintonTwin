import argparse
import importlib.util
import json
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CTCLoss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if project_path not in sys.path:
    sys.path.append(project_path)

from Agent.base_agent import BaseAgent
from Utils.helper import (
    action_col,
    edit_distance,
    normalized_dtw_ndim_distance,
    state_col,
    terminate,
    transition,
)
from Utils.runtime_report import print_runtime_box
from Preprocessing.rally_data import filter_rally_rows, load_rally_dataset, rally_key


PAD = 0
BOS_SHOT = 12
SHOT_OUTPUT_NUM = 12
SHOT_INPUT_NUM = 13
DEFAULT_ENCODE_LENGTH = 4
DEFAULT_MAX_BALL_ROUND = 72


DEFAULT_CONFIG = {
    'encode_length': DEFAULT_ENCODE_LENGTH,
    'max_ball_round': DEFAULT_MAX_BALL_ROUND,
    'shot_dim': 32,
    'area_dim': 32,
    'player_dim': 32,
    'encode_dim': 32,
    'area_num': 5,
    'shot_input_num': SHOT_INPUT_NUM,
    'shot_output_num': SHOT_OUTPUT_NUM,
    'dropout': 0.1,
}


def resolve_device(device):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device(device)


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_pad_mask(seq):
    return (seq != PAD).unsqueeze(-2)


def get_subsequent_mask(seq):
    batch_size, seq_len = seq.size()
    mask = 1 - torch.triu(torch.ones((1, seq_len, seq_len), device=seq.device), diagonal=1)
    return mask.bool().expand(batch_size, -1, -1)


def alternatemerge(seq_a, seq_b, merge_len, player):
    batch_size, seq_len, dim = seq_a.shape
    merged_seq = torch.zeros(batch_size, merge_len, dim, device=seq_a.device, dtype=seq_a.dtype)

    if seq_len * 2 == (merge_len - 1):
        merged_seq[:, 1::2, :] = seq_a[:, :seq_len, :]
        merged_seq[:, 2::2, :] = seq_b[:, :seq_len, :]
    elif (seq_len * 2 - 1) == merge_len:
        merged_seq[:, ::2, :] = seq_a[:, :seq_len, :]
        merged_seq[:, 1::2, :] = seq_b[:, :merge_len - seq_len, :]
    elif seq_len * 2 == merge_len:
        if player == 'A':
            merged_seq[:, ::2, :] = seq_a[:, :seq_len, :]
            merged_seq[:, 1::2, :] = seq_b[:, :seq_len, :]
        elif player == 'B':
            merged_seq[:, 1::2, :] = seq_a[:, :seq_len, :]
            merged_seq[:, 2::2, :] = seq_b[:, :seq_len - 1, :]
        else:
            raise ValueError("player must be 'A' or 'B'")
    elif merge_len == seq_len:
        merged_seq[:, :, :] = seq_a[:, :merge_len, :]
    else:
        raise ValueError(f'Cannot alternate merge seq_len={seq_len} merge_len={merge_len}')

    return merged_seq


class PositionalEncoding(nn.Module):
    def __init__(self, dim, encode_length, n_position=200):
        super().__init__()
        self.encode_length = encode_length
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, dim))

    @staticmethod
    def _get_sinusoid_encoding_table(n_position, dim):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / dim) for hid_j in range(dim)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x, mode='encode'):
        if mode == 'encode':
            return x + self.pos_table[:, :x.size(1)].clone().detach()
        if mode == 'decode':
            start = self.encode_length
            end = self.encode_length + x.size(1)
            return x + self.pos_table[:, start:end].clone().detach()
        raise ValueError("mode must be 'encode' or 'decode'")


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q

        q = self.w_qs(q).view(batch_size, len_q, n_head, d_k)
        k = self.w_ks(k).view(batch_size, len_k, n_head, d_k)
        v = self.w_vs(v).view(batch_size, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q, attn


class TypeAreaScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q_a, k_a, v_a, q_s, k_s, v_s, mask=None, return_attns=False):
        a2a = torch.matmul(q_a, k_a.transpose(2, 3))
        a2s = torch.matmul(q_a, k_s.transpose(2, 3))
        s2a = torch.matmul(q_s, k_a.transpose(2, 3))
        s2s = torch.matmul(q_s, k_s.transpose(2, 3))
        attention_score = (a2a + a2s + s2a + s2s) / self.temperature

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)

        disentangled = {}
        if return_attns:
            parts = {'a2a': a2a, 'a2s': a2s, 's2a': s2a, 's2s': s2s}
            for key, score in parts.items():
                score = score / self.temperature
                if mask is not None:
                    score = score.masked_fill(mask == 0, -1e9)
                disentangled[key] = self.dropout(F.softmax(score, dim=-1))

        attention_score = self.dropout(F.softmax(attention_score, dim=-1))
        output = torch.matmul(attention_score, v_a + v_s)
        return output, attention_score, disentangled


class TypeAreaMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.w_qa = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ka = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_va = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = TypeAreaScaledDotProductAttention(temperature=(4 * d_k) ** 0.5, attn_dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q_a, k_a, v_a, q_s, k_s, v_s, mask=None, return_attns=False):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch_size, len_q, len_k, len_v = q_a.size(0), q_a.size(1), k_a.size(1), v_a.size(1)
        residual_a = q_a
        residual_s = q_s

        q_a = self.w_qa(q_a).view(batch_size, len_q, n_head, d_k)
        k_a = self.w_ka(k_a).view(batch_size, len_k, n_head, d_k)
        v_a = self.w_va(v_a).view(batch_size, len_v, n_head, d_v)
        q_s = self.w_qs(q_s).view(batch_size, len_q, n_head, d_k)
        k_s = self.w_ks(k_s).view(batch_size, len_k, n_head, d_k)
        v_s = self.w_vs(v_s).view(batch_size, len_v, n_head, d_v)

        q_a, k_a, v_a = q_a.transpose(1, 2), k_a.transpose(1, 2), v_a.transpose(1, 2)
        q_s, k_s, v_s = q_s.transpose(1, 2), k_s.transpose(1, 2), v_s.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        output, attn, disentangled = self.attention(
            q_a, k_a, v_a, q_s, k_s, v_s, mask=mask, return_attns=return_attns
        )
        output = output.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual_a + residual_s
        output = self.layer_norm(output)
        return output, attn, disentangled


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.gelu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return self.layer_norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.disentangled_attention = TypeAreaMultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, encode_area, encode_shot, slf_attn_mask=None):
        encode_output, enc_slf_attn, _ = self.disentangled_attention(
            encode_area, encode_area, encode_area,
            encode_shot, encode_shot, encode_shot,
            mask=slf_attn_mask,
        )
        encode_output = self.pos_ffn(encode_output)
        return encode_output, enc_slf_attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.decoder_attention = TypeAreaMultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.decoder_encoder_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, decode_area, decode_shot, encode_output, slf_attn_mask=None, dec_enc_attn_mask=None, return_attns=False):
        decode_output, dec_slf_attn, disentangled_weight = self.decoder_attention(
            decode_area, decode_area, decode_area,
            decode_shot, decode_shot, decode_shot,
            mask=slf_attn_mask,
            return_attns=return_attns,
        )
        decode_output, dec_enc_slf_attn = self.decoder_encoder_attention(
            decode_output, encode_output, encode_output, mask=dec_enc_attn_mask
        )
        decode_output = self.pos_ffn(decode_output)
        return decode_output, dec_slf_attn, dec_enc_slf_attn, disentangled_weight


class GatedFusionLayer(nn.Module):
    def __init__(self, d, d_inner, encode_length, n_position=200):
        super().__init__()
        n = 3
        self.hidden1 = nn.Linear(d, d_inner, bias=False)
        self.hidden2 = nn.Linear(d, d_inner, bias=False)
        self.hidden3 = nn.Linear(d, d_inner, bias=False)
        self.gated1 = nn.Linear(d_inner * n, d, bias=False)
        self.gated2 = nn.Linear(d_inner * n, d, bias=False)
        self.gated3 = nn.Linear(d_inner * n, d, bias=False)
        self.decode_length = n_position - encode_length
        self.w_A = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_B = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.w_L = nn.Parameter(torch.zeros([self.decode_length, d]), requires_grad=True)
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x_A, x_B, x_L):
        batch_size, seq_len, _ = x_A.shape
        w_A = self.w_A.unsqueeze(0).repeat_interleave(batch_size, dim=0)
        w_B = self.w_B.unsqueeze(0).repeat_interleave(batch_size, dim=0)
        w_L = self.w_L.unsqueeze(0).repeat_interleave(batch_size, dim=0)
        h_A = self.tanh_f(self.hidden1(x_A))
        h_B = self.tanh_f(self.hidden2(x_B))
        h_L = self.tanh_f(self.hidden3(x_L))
        x = torch.cat((x_A, x_B, x_L), dim=-1)
        z1 = self.sigmoid_f(self.gated1(x)) * h_A
        z2 = self.sigmoid_f(self.gated2(x)) * h_B
        z3 = self.sigmoid_f(self.gated3(x)) * h_L
        z1 = w_A[:, :seq_len, :] * z1
        z2 = w_B[:, :seq_len, :] * z2
        z3 = w_L[:, :seq_len, :] * z3
        return self.sigmoid_f(z1 + z2 + z3)


class ShotGenEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.area_embedding = nn.Linear(2, config['area_dim'])
        self.shot_embedding = nn.Embedding(config['shot_input_num'], config['shot_dim'], padding_idx=PAD)
        self.player_embedding = nn.Embedding(config['player_num'], config['player_dim'], padding_idx=PAD)

        n_heads = 2
        d_model = config['encode_dim']
        d_inner = config['encode_dim'] * 2
        dropout = config.get('dropout', 0.1)
        n_position = config.get('max_ball_round', DEFAULT_MAX_BALL_ROUND) + config.get('encode_length', DEFAULT_ENCODE_LENGTH) + 4

        self.position_embedding = PositionalEncoding(config['shot_dim'], config['encode_length'], n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.global_layer = EncoderLayer(d_model, d_inner, n_heads, d_model, d_model, dropout=dropout)
        self.local_layer = EncoderLayer(d_model, d_inner, n_heads, d_model, d_model, dropout=dropout)

    def _encode_branch(self, area, shot, mask, valid_mask):
        if area.shape[1] == 0:
            batch_size = area.shape[0]
            dim = area.shape[-1]
            zeros = torch.zeros(batch_size, 1, dim, device=area.device, dtype=area.dtype)
            return zeros
        output, _ = self.global_layer(area, shot, slf_attn_mask=mask)
        return output * valid_mask

    def forward(self, input_shot, input_x, input_y, input_player):
        area = torch.cat((input_x.unsqueeze(-1), input_y.unsqueeze(-1)), dim=-1).float()
        embedded_area = F.relu(self.area_embedding(area))
        embedded_shot = self.shot_embedding(input_shot.long())
        embedded_player = self.player_embedding(input_player.long())
        h_a = embedded_area + embedded_player
        h_s = embedded_shot + embedded_player

        h_a_A, h_a_B = h_a[:, ::2], h_a[:, 1::2]
        h_s_A, h_s_B = h_s[:, ::2], h_s[:, 1::2]
        shot_A, shot_B = input_shot[:, ::2], input_shot[:, 1::2]

        valid_local = (input_shot != PAD).unsqueeze(-1).float()
        valid_A = (shot_A != PAD).unsqueeze(-1).float() if shot_A.shape[1] else None
        valid_B = (shot_B != PAD).unsqueeze(-1).float() if shot_B.shape[1] else None

        local_area = self.dropout(self.position_embedding(h_a, mode='encode')) * valid_local
        local_shot = self.dropout(self.position_embedding(h_s, mode='encode')) * valid_local
        area_A = self.dropout(self.position_embedding(h_a_A, mode='encode')) * valid_A if h_a_A.shape[1] else h_a_A
        area_B = self.dropout(self.position_embedding(h_a_B, mode='encode')) * valid_B if h_a_B.shape[1] else h_a_B
        shot_A_emb = self.dropout(self.position_embedding(h_s_A, mode='encode')) * valid_A if h_s_A.shape[1] else h_s_A
        shot_B_emb = self.dropout(self.position_embedding(h_s_B, mode='encode')) * valid_B if h_s_B.shape[1] else h_s_B

        local_mask = get_pad_mask(input_shot)
        mask_A = get_pad_mask(shot_A) if shot_A.shape[1] else None
        mask_B = get_pad_mask(shot_B) if shot_B.shape[1] else None

        encode_global_A = self._encode_branch(area_A, shot_A_emb, mask_A, valid_A)
        encode_global_B = self._encode_branch(area_B, shot_B_emb, mask_B, valid_B)
        encode_local_output, _ = self.local_layer(local_area, local_shot, slf_attn_mask=local_mask)
        encode_local_output = encode_local_output * valid_local
        return encode_local_output, encode_global_A, encode_global_B


class ShotGenDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.area_embedding = nn.Linear(2, config['area_dim'])
        self.shot_embedding = nn.Embedding(config['shot_input_num'], config['shot_dim'], padding_idx=PAD)
        self.player_embedding = nn.Embedding(config['player_num'], config['player_dim'], padding_idx=PAD)

        n_heads = 2
        d_model = config['encode_dim']
        d_inner = config['encode_dim'] * 2
        dropout = config.get('dropout', 0.1)
        n_position = config.get('max_ball_round', DEFAULT_MAX_BALL_ROUND) + config.get('encode_length', DEFAULT_ENCODE_LENGTH) + 4

        self.position_embedding = PositionalEncoding(config['shot_dim'], config['encode_length'], n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.global_layer = DecoderLayer(d_model, d_inner, n_heads, d_model, d_model, dropout=dropout)
        self.local_layer = DecoderLayer(d_model, d_inner, n_heads, d_model, d_model, dropout=dropout)
        self.gated_fusion = GatedFusionLayer(d_model, d_model, config['encode_length'], n_position=n_position)

    def forward(self, input_shot, input_x, input_y, input_player, encode_local_output, encode_global_A, encode_global_B, source_shot=None, return_attns=False):
        area = torch.cat((input_x.unsqueeze(-1), input_y.unsqueeze(-1)), dim=-1).float()
        mask_A = input_shot[:, ::2]
        mask_B = input_shot[:, 1::2]

        trg_local_mask = get_pad_mask(input_shot) & get_subsequent_mask(input_shot)
        trg_global_A_mask = get_pad_mask(mask_A) & get_subsequent_mask(mask_A) if mask_A.shape[1] else None
        trg_global_B_mask = get_pad_mask(mask_B) & get_subsequent_mask(mask_B) if mask_B.shape[1] else None
        src_local_mask = get_pad_mask(source_shot) if source_shot is not None else None
        src_global_A_mask = get_pad_mask(source_shot[:, ::2]) if source_shot is not None else None
        src_global_B_mask = get_pad_mask(source_shot[:, 1::2]) if source_shot is not None and source_shot[:, 1::2].shape[1] else None

        embedded_area = F.relu(self.area_embedding(area))
        embedded_shot = self.shot_embedding(input_shot.long())
        embedded_player = self.player_embedding(input_player.long())
        h_a = embedded_area + embedded_player
        h_s = embedded_shot + embedded_player

        h_a_A, h_a_B = h_a[:, ::2], h_a[:, 1::2]
        h_s_A, h_s_B = h_s[:, ::2], h_s[:, 1::2]

        decode_area = self.dropout(self.position_embedding(h_a, mode='decode'))
        decode_shot = self.dropout(self.position_embedding(h_s, mode='decode'))
        decode_area_A = self.dropout(self.position_embedding(h_a_A, mode='decode')) if h_a_A.shape[1] else h_a_A
        decode_area_B = self.dropout(self.position_embedding(h_a_B, mode='decode')) if h_a_B.shape[1] else h_a_B
        decode_shot_A = self.dropout(self.position_embedding(h_s_A, mode='decode')) if h_s_A.shape[1] else h_s_A
        decode_shot_B = self.dropout(self.position_embedding(h_s_B, mode='decode')) if h_s_B.shape[1] else h_s_B

        decode_global_A, _, _, _ = self.global_layer(
            decode_area_A, decode_shot_A, encode_global_A,
            slf_attn_mask=trg_global_A_mask,
            dec_enc_attn_mask=src_global_A_mask,
            return_attns=return_attns,
        )
        if decode_area_B.shape[1] != 0:
            decode_global_B, _, _, _ = self.global_layer(
                decode_area_B, decode_shot_B, encode_global_B,
                slf_attn_mask=trg_global_B_mask,
                dec_enc_attn_mask=src_global_B_mask,
                return_attns=return_attns,
            )
        else:
            decode_global_B = torch.zeros(decode_area.shape, device=decode_area.device, dtype=decode_area.dtype)

        decode_local_output, _, _, _ = self.local_layer(
            decode_area, decode_shot, encode_local_output,
            slf_attn_mask=trg_local_mask,
            dec_enc_attn_mask=src_local_mask,
            return_attns=return_attns,
        )

        if decode_area_B.shape[1] != 0:
            decode_output_A = alternatemerge(decode_global_A, decode_global_A, decode_local_output.shape[1], 'A')
            decode_output_B = alternatemerge(decode_global_B, decode_global_B, decode_local_output.shape[1], 'B')
        else:
            decode_output_A = decode_global_A.clone()
            decode_output_B = torch.zeros(decode_local_output.shape, device=decode_local_output.device)

        return self.gated_fusion(decode_output_A, decode_output_B, decode_local_output)


class ShotGenPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shotgen_decoder = ShotGenDecoder(config)
        self.area_decoder = nn.Linear(config['encode_dim'], config['area_num'], bias=False)
        self.move_decoder = nn.Linear(config['encode_dim'], config['area_num'], bias=False)
        self.shot_decoder = nn.Linear(config['encode_dim'], config['shot_output_num'], bias=False)
        self.player_embedding = nn.Embedding(config['player_num'], config['player_dim'], padding_idx=PAD)

    def forward(self, input_shot, input_x, input_y, input_player, encode_local_output, encode_global_A, encode_global_B, target_player, source_shot=None):
        embedded_target_player = self.player_embedding(target_player.long())
        decode_output = self.shotgen_decoder(
            input_shot, input_x, input_y, input_player,
            encode_local_output, encode_global_A, encode_global_B,
            source_shot=source_shot,
        )
        decode_output = decode_output + embedded_target_player
        area_logits = self.area_decoder(decode_output)
        move_logits = self.move_decoder(decode_output)
        shot_logits = self.shot_decoder(decode_output)
        return area_logits, shot_logits, move_logits


def gaussian2d_nll(pred_params, target, reduction='mean'):
    mu_x = pred_params[:, 0]
    mu_y = pred_params[:, 1]
    log_sx = torch.clamp(pred_params[:, 2], min=-7.0, max=7.0)
    log_sy = torch.clamp(pred_params[:, 3], min=-7.0, max=7.0)
    corr = torch.tanh(pred_params[:, 4])
    sx = torch.exp(log_sx)
    sy = torch.exp(log_sy)
    norm_x = target[:, 0] - mu_x
    norm_y = target[:, 1] - mu_y
    one_minus_corr2 = torch.clamp(1 - corr ** 2, min=1e-5)
    z = (norm_x / sx) ** 2 + (norm_y / sy) ** 2 - 2 * corr * norm_x * norm_y / (sx * sy)
    nll = torch.log(2 * torch.pi * sx * sy * torch.sqrt(one_minus_corr2)) + z / (2 * one_minus_corr2)
    if reduction == 'none':
        return nll
    if reduction == 'sum':
        return nll.sum()
    return nll.mean()


def sample_gaussian2d(pred_params):
    mu = pred_params[:, 0:2]
    sx = torch.exp(torch.clamp(pred_params[:, 2], min=-7.0, max=7.0))
    sy = torch.exp(torch.clamp(pred_params[:, 3], min=-7.0, max=7.0))
    corr = torch.tanh(pred_params[:, 4])
    eps_x = torch.randn_like(sx)
    eps_y = torch.randn_like(sy)
    sample_x = mu[:, 0] + sx * eps_x
    sample_y = mu[:, 1] + sy * (corr * eps_x + torch.sqrt(torch.clamp(1 - corr ** 2, min=1e-5)) * eps_y)
    return torch.clamp(torch.stack([sample_x, sample_y], dim=-1), -1.0, 1.0)


def deterministic_gaussian_mean(pred_params):
    return torch.clamp(pred_params[:, 0:2], -1.0, 1.0)


def build_shot_mask(launch, device):
    if launch:
        values = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    else:
        values = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
    return torch.tensor(values, dtype=torch.bool, device=device)


def apply_shot_mask(shot_logits, launch):
    mask = build_shot_mask(launch, shot_logits.device)
    return shot_logits.masked_fill(~mask, -1e9)


def external_to_internal_player(player_id):
    return int(player_id) + 1


def internal_to_external_player(player_id):
    return int(player_id) - 1


def build_config(player2index, args_or_dict=None):
    config = DEFAULT_CONFIG.copy()
    if args_or_dict is not None:
        source = vars(args_or_dict) if not isinstance(args_or_dict, dict) else args_or_dict
        for key in ('encode_length', 'max_ball_round', 'shot_dim', 'area_dim', 'player_dim', 'encode_dim', 'area_num', 'dropout'):
            if source.get(key) is not None:
                config[key] = source[key]
    config['player_num'] = len(player2index) + 1
    config['shot_input_num'] = SHOT_INPUT_NUM
    config['shot_output_num'] = SHOT_OUTPUT_NUM
    return config


def row_to_history_item(row, player2index):
    player_name = row['player']
    return {
        'shot': int(row['player_type']),
        'x': float(row['landing_x']),
        'y': float(row['landing_y']),
        'move_x': float(row['moving_x']),
        'move_y': float(row['moving_y']),
        'player': external_to_internal_player(player2index[player_name]),
    }


def build_sample_from_history(history, target_row, player2index, encode_length):
    target_player = external_to_internal_player(player2index[target_row['player']])
    enc_shot = np.zeros(encode_length, dtype=np.int64)
    enc_x = np.zeros(encode_length, dtype=np.float32)
    enc_y = np.zeros(encode_length, dtype=np.float32)
    enc_player = np.zeros(encode_length, dtype=np.int64)

    history_tail = history[-encode_length:]
    for idx, item in enumerate(history_tail):
        enc_shot[idx] = item['shot']
        enc_x[idx] = item['x']
        enc_y[idx] = item['y']
        enc_player[idx] = item['player']

    if history_tail:
        prev = history_tail[-1]
        dec_shot = prev['shot']
        dec_x = prev['x']
        dec_y = prev['y']
        dec_player = prev['player']
    else:
        dec_shot = BOS_SHOT
        dec_x = 0.0
        dec_y = 0.0
        dec_player = target_player

    return {
        'enc_shot': enc_shot,
        'enc_x': enc_x,
        'enc_y': enc_y,
        'enc_player': enc_player,
        'dec_shot': np.array([dec_shot], dtype=np.int64),
        'dec_x': np.array([dec_x], dtype=np.float32),
        'dec_y': np.array([dec_y], dtype=np.float32),
        'dec_player': np.array([dec_player], dtype=np.int64),
        'target_player': np.array([target_player], dtype=np.int64),
        'target_shot': np.array(int(target_row['player_type']), dtype=np.int64),
        'target_land': np.array([float(target_row['landing_x']), float(target_row['landing_y'])], dtype=np.float32),
        'target_move': np.array([float(target_row['moving_x']), float(target_row['moving_y'])], dtype=np.float32),
        'launch': np.array(int(target_row['ball_round']) == 1, dtype=np.bool_),
    }


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def prepare_shuttlenet_data(merged_csv_path, split_path, player_map_path, output_dir, encode_length=DEFAULT_ENCODE_LENGTH):
    import pandas as pd
    from Preprocessing.rally_data import filter_rally_rows, load_rally_dataset, rally_key

    player2index = load_json(player_map_path)
    split_content = load_rally_dataset(split_path)
    df = pd.read_csv(merged_csv_path)
    required_cols = ['match_id', 'set', 'rally', 'ball_round', 'player', 'player_type', 'landing_x', 'landing_y', 'moving_x', 'moving_y']
    df = df.dropna(subset=required_cols).copy()
    df = df[df['player'].isin(player2index)].reset_index(drop=True)
    df['match_id'] = df['match_id'].astype(int)
    df = filter_rally_rows(df, split_content['train'])
    if df.empty:
        raise ValueError('No ShuttleNet rows matched the rally-level training split')

    expected_keys = {rally_key(record) for record in split_content['train']}
    matched_keys = set(
        df[['match_id', 'set', 'rally']]
        .astype(int)
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    skipped_keys = sorted(expected_keys - matched_keys)

    samples = []
    group_cols = ['match_id', 'set', 'rally']
    for _, rally_df in tqdm(df.groupby(group_cols), desc='Preparing ShuttleNet samples'):
        rally_df = rally_df.sort_values('ball_round').reset_index(drop=True)
        history = []
        for _, row in rally_df.iterrows():
            samples.append(build_sample_from_history(history, row, player2index, encode_length))
            history.append(row_to_history_item(row, player2index))

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'train_samples.pkl')
    payload = {
        'samples': samples,
        'metadata': {
            'encode_length': encode_length,
            'num_samples': len(samples),
            'merged_csv_path': merged_csv_path,
            'split_path': split_path,
            'player_map_path': player_map_path,
            'physical_train_rallies_in_split': len(expected_keys),
            'physical_train_rallies_matched': len(matched_keys),
            'physical_train_rallies_skipped': len(skipped_keys),
            'skipped_rally_keys': [
                {'match_id': key[0], 'set': key[1], 'rally': key[2]}
                for key in skipped_keys
            ],
        },
    }
    with open(output_path, 'wb') as f:
        pickle.dump(payload, f)
    print(f'ShuttleNet training data saved to {output_path} ({len(samples)} samples)')
    return output_path


class ShuttleNetDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'enc_shot': torch.tensor(sample['enc_shot'], dtype=torch.long),
            'enc_x': torch.tensor(sample['enc_x'], dtype=torch.float32),
            'enc_y': torch.tensor(sample['enc_y'], dtype=torch.float32),
            'enc_player': torch.tensor(sample['enc_player'], dtype=torch.long),
            'dec_shot': torch.tensor(sample['dec_shot'], dtype=torch.long),
            'dec_x': torch.tensor(sample['dec_x'], dtype=torch.float32),
            'dec_y': torch.tensor(sample['dec_y'], dtype=torch.float32),
            'dec_player': torch.tensor(sample['dec_player'], dtype=torch.long),
            'target_player': torch.tensor(sample['target_player'], dtype=torch.long),
            'target_shot': torch.tensor(sample['target_shot'], dtype=torch.long),
            'target_land': torch.tensor(sample['target_land'], dtype=torch.float32),
            'target_move': torch.tensor(sample['target_move'], dtype=torch.float32),
        }


def forward_batch(encoder, predictor, batch):
    encode_local_output, encode_global_A, encode_global_B = encoder(
        batch['enc_shot'], batch['enc_x'], batch['enc_y'], batch['enc_player']
    )
    return predictor(
        batch['dec_shot'], batch['dec_x'], batch['dec_y'], batch['dec_player'],
        encode_local_output, encode_global_A, encode_global_B, batch['target_player'],
        source_shot=batch['enc_shot'],
    )


def move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def train_model(args):
    device = resolve_device(args.device)
    player2index = load_json(args.player_map)
    data_path = os.path.join(args.data_dir, 'train_samples.pkl')
    if not os.path.exists(data_path) or args.reprepare:
        prepare_shuttlenet_data(args.merged_csv, args.split, args.player_map, args.data_dir, args.encode_length)

    with open(data_path, 'rb') as f:
        payload = pickle.load(f)
    samples = payload['samples']
    if not samples:
        raise ValueError(f'No ShuttleNet training samples found in {data_path}')

    config = build_config(player2index, args)
    dataset = ShuttleNetDataset(samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    encoder = ShotGenEncoder(config).to(device)
    predictor = ShotGenPredictor(config).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=args.lr)
    ce_loss = nn.CrossEntropyLoss(ignore_index=PAD)

    print_runtime_box(
        'Runtime Parameters: Agent/shuttlenet.py train',
        [
            ('device', device),
            ('data_path', data_path),
            ('output', args.output),
            ('epochs', args.epochs),
            ('batch_size', args.batch_size),
            ('learning_rate', args.lr),
            ('encode_length', config['encode_length']),
            ('max_ball_round', config['max_ball_round']),
            ('samples', len(samples)),
        ],
    )

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        predictor.train()
        total_loss = 0.0
        total_land = 0.0
        total_move = 0.0
        total_shot = 0.0
        for batch in tqdm(loader, desc=f'Epoch {epoch}/{args.epochs}', leave=False):
            batch = move_batch_to_device(batch, device)
            land_params, shot_logits, move_params = forward_batch(encoder, predictor, batch)
            land_params = land_params[:, -1, :]
            move_params = move_params[:, -1, :]
            shot_logits = shot_logits[:, -1, :]

            loss_land = gaussian2d_nll(land_params, batch['target_land'])
            loss_move = gaussian2d_nll(move_params, batch['target_move'])
            loss_shot = ce_loss(shot_logits, batch['target_shot'])
            loss = loss_land + loss_move + loss_shot

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(predictor.parameters()), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_land += loss_land.item()
            total_move += loss_move.item()
            total_shot += loss_shot.item()

        denom = max(len(loader), 1)
        print(
            f'Epoch {epoch}: total={total_loss / denom:.4f} '
            f'land_nll={total_land / denom:.4f} move_nll={total_move / denom:.4f} shot_ce={total_shot / denom:.4f}'
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(
        {
            'encoder_state_dict': encoder.state_dict(),
            'predictor_state_dict': predictor.state_dict(),
            'config': config,
            'player2index': player2index,
        },
        args.output,
    )
    print(f'ShuttleNet model saved to {args.output}')


class ShuttleNetPolicy:
    def __init__(self, model_path, device='cuda', sample=True):
        self.device = resolve_device(device)
        self.sample = sample
        checkpoint = safe_torch_load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        self.player2index = checkpoint['player2index']
        self.encoder = ShotGenEncoder(self.config).to(self.device)
        self.predictor = ShotGenPredictor(self.config).to(self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.predictor.load_state_dict(checkpoint['predictor_state_dict'])
        self.encoder.eval()
        self.predictor.eval()

    def tensors_from_history(self, history, target_player):
        sample = build_sample_from_history_for_inference(history, target_player, self.config['encode_length'])
        return {key: value.to(self.device) for key, value in sample.items()}

    @torch.inference_mode()
    def predict_next(self, history, target_player, launch=False):
        batch = self.tensors_from_history(history, target_player)
        land_params, shot_logits, move_params = forward_batch(self.encoder, self.predictor, batch)
        land_params = land_params[:, -1, :]
        move_params = move_params[:, -1, :]
        shot_logits = shot_logits[:, -1, :]

        masked_logits = apply_shot_mask(shot_logits, launch)
        shot_probs = F.softmax(masked_logits, dim=-1)
        if self.sample:
            shot_type = torch.distributions.Categorical(probs=shot_probs).sample()
            land = sample_gaussian2d(land_params)
            move = sample_gaussian2d(move_params)
        else:
            shot_type = torch.argmax(shot_probs, dim=-1)
            land = deterministic_gaussian_mean(land_params)
            move = deterministic_gaussian_mean(move_params)

        return {
            'shot_type': int(shot_type.item()),
            'landing': land.squeeze(0).detach().cpu().tolist(),
            'moving': move.squeeze(0).detach().cpu().tolist(),
            'shot_logits': masked_logits.squeeze(0).detach().cpu(),
            'shot_probs': shot_probs.squeeze(0).detach().cpu().tolist(),
        }


def build_sample_from_history_for_inference(history, target_player, encode_length):
    enc_shot = torch.zeros(1, encode_length, dtype=torch.long)
    enc_x = torch.zeros(1, encode_length, dtype=torch.float32)
    enc_y = torch.zeros(1, encode_length, dtype=torch.float32)
    enc_player = torch.zeros(1, encode_length, dtype=torch.long)

    history_tail = history[-encode_length:]
    for idx, item in enumerate(history_tail):
        enc_shot[0, idx] = int(item['shot'])
        enc_x[0, idx] = float(item['x'])
        enc_y[0, idx] = float(item['y'])
        enc_player[0, idx] = int(item['player'])

    if history_tail:
        prev = history_tail[-1]
        dec_shot = int(prev['shot'])
        dec_x = float(prev['x'])
        dec_y = float(prev['y'])
        dec_player = int(prev['player'])
    else:
        dec_shot = BOS_SHOT
        dec_x = 0.0
        dec_y = 0.0
        dec_player = int(target_player)

    return {
        'enc_shot': enc_shot,
        'enc_x': enc_x,
        'enc_y': enc_y,
        'enc_player': enc_player,
        'dec_shot': torch.tensor([[dec_shot]], dtype=torch.long),
        'dec_x': torch.tensor([[dec_x]], dtype=torch.float32),
        'dec_y': torch.tensor([[dec_y]], dtype=torch.float32),
        'dec_player': torch.tensor([[dec_player]], dtype=torch.long),
        'target_player': torch.tensor([[target_player]], dtype=torch.long),
    }


class ShuttleNetRollout:
    def __init__(self, policy):
        self.policy = policy
        self.history = []

    def reset(self):
        self.history = []

    def step(self, current_state, launch=False):
        target_external_player = int(current_state[17].item())
        target_player = external_to_internal_player(target_external_player)
        prediction = self.policy.predict_next(self.history, target_player, launch=launch)
        shot_type = prediction['shot_type']
        landing = prediction['landing']
        moving = prediction['moving']
        self.history.append({
            'shot': shot_type,
            'x': landing[0],
            'y': landing[1],
            'move_x': moving[0],
            'move_y': moving[1],
            'player': target_player,
        })
        action_tensor = torch.tensor([landing[0], landing[1], float(shot_type), moving[0], moving[1]], dtype=torch.float32)
        return action_tensor, prediction


class ShuttleNetAgent(BaseAgent):
    def __init__(self, player_name, model_path, player_id=None, device='cuda', sample=True):
        super().__init__(player_name)
        self.policy = ShuttleNetPolicy(model_path, device=device, sample=sample)
        if player_id is None:
            player_id = self.policy.player2index[player_name]
        self.player_id = int(player_id)
        self.history = []

    def reset_history(self):
        self.history = []

    def predict(self, state, info, launch):
        target_player = external_to_internal_player(self.player_id)
        prediction = self.policy.predict_next(self.history, target_player, launch=launch)
        shot_type = prediction['shot_type']
        landing = prediction['landing']
        moving = prediction['moving']
        self.history.append({
            'shot': shot_type,
            'x': landing[0],
            'y': landing[1],
            'move_x': moving[0],
            'move_y': moving[1],
            'player': target_player,
        })
        action = (shot_type, state[-1], tuple(landing), tuple(moving), tuple(prediction['shot_probs'][1:11]))
        return action, ''


def build_state_tensor_from_row(row, player_id):
    state_values = [row[col] for col in state_col]
    state_values[-1] = player_id
    return torch.tensor(state_values, dtype=torch.float32)


def build_action_tensor_from_row(row):
    return torch.tensor([row[col] for col in action_col], dtype=torch.float32)


def get_opponent_name(match_df, target_player_name):
    for player_name in match_df['player'].unique():
        if player_name != target_player_name:
            return player_name
    return None


def evaluate_shuttlenet_on_test(test_split_path, merged_csv_path, player_index_map_path, model_path,
                                device='cuda', skip_first_n=0, sample=True, max_steps_multiplier=2):
    import pandas as pd

    player2index = load_json(player_index_map_path)
    test_split = load_json(test_split_path)
    df = pd.read_csv(merged_csv_path)
    target_cols = state_col + action_col
    df = df.dropna(subset=target_cols).reset_index(drop=True)
    policy = ShuttleNetPolicy(model_path, device=device, sample=sample)
    ctc_loss_fn = CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    results = {}

    for target_player_name, test_match_ids in tqdm(test_split.get('test', {}).items(), desc='Evaluating players'):
        if target_player_name not in player2index:
            continue
        target_player_id = player2index[target_player_name]
        rally_land_dtw_list = []
        rally_move_dtw_list = []
        rally_shot_ctc_list = []
        rally_edit_dist_list = []
        total_rallies = 0
        total_shots = 0

        for match_id in tqdm(test_match_ids, desc=f'Matches for {target_player_name}', leave=False):
            match_df = df[df['match_id'] == match_id].copy()
            if match_df.empty:
                continue
            opponent_name = get_opponent_name(match_df, target_player_name)
            if opponent_name is None or opponent_name not in player2index:
                continue
            opponent_player_id = player2index[opponent_name]

            for _, rally_df in match_df.groupby(['set', 'rally']):
                rally_df = rally_df.sort_values('ball_round').reset_index(drop=True)
                if rally_df.empty:
                    continue

                gt_target_df = rally_df[rally_df['player'] == target_player_name].reset_index(drop=True)
                if len(gt_target_df) <= skip_first_n:
                    continue

                rollout = ShuttleNetRollout(policy)
                first_row = rally_df.iloc[0]
                first_player_id = target_player_id if first_row['player'] == target_player_name else opponent_player_id
                current_state = build_state_tensor_from_row(first_row, first_player_id)
                max_steps = max(len(rally_df) * max_steps_multiplier + 2, len(rally_df), 1)
                target_seen = 0
                target_recorded = 0
                land_pred_seq, move_pred_seq = [], []
                shot_logits_seq, shot_pred_seq = [], []

                for _ in range(max_steps):
                    current_player_id = int(current_state[17].item())
                    is_target_turn = current_player_id == target_player_id
                    launch = current_state[1].item() == 1
                    action_tensor, prediction = rollout.step(current_state, launch=launch)

                    if is_target_turn:
                        target_seen += 1
                        if target_seen > skip_first_n:
                            land_pred_seq.append(np.array(prediction['landing'], dtype=np.float32))
                            move_pred_seq.append(np.array(prediction['moving'], dtype=np.float32))
                            shot_logits_seq.append(prediction['shot_logits'])
                            shot_pred_seq.append(int(prediction['shot_type']))
                            target_recorded += 1
                        if target_recorded >= len(gt_target_df) - skip_first_n:
                            break

                    other_player_id = opponent_player_id if is_target_turn else target_player_id
                    next_state = transition(action_tensor.cpu(), current_state.cpu(), other_player_id)
                    if terminate(next_state.cpu()):
                        break
                    current_state = next_state

                if not land_pred_seq:
                    continue

                gt_eval_df = gt_target_df.iloc[skip_first_n:]
                land_gt_seq = gt_eval_df[['landing_x', 'landing_y']].to_numpy(dtype=np.float32)
                move_gt_seq = gt_eval_df[['moving_x', 'moving_y']].to_numpy(dtype=np.float32)
                shot_gt_seq = gt_eval_df['player_type'].astype(int).tolist()
                if len(land_gt_seq) == 0:
                    continue

                rally_land_dtw_list.append(normalized_dtw_ndim_distance(np.array(land_pred_seq), land_gt_seq))
                rally_move_dtw_list.append(normalized_dtw_ndim_distance(np.array(move_pred_seq), move_gt_seq))
                if shot_logits_seq:
                    logits_tensor = torch.stack(shot_logits_seq)
                    target_tensor = torch.tensor(shot_gt_seq, dtype=torch.long)
                    log_probs = F.log_softmax(logits_tensor.unsqueeze(1), dim=-1)
                    input_length = torch.tensor([log_probs.shape[0]], dtype=torch.long)
                    target_length = torch.tensor([len(target_tensor)], dtype=torch.long)
                    rally_shot_ctc_list.append(ctc_loss_fn(log_probs, target_tensor, input_length, target_length).item())
                rally_edit_dist_list.append(edit_distance(shot_pred_seq, shot_gt_seq))
                total_rallies += 1
                total_shots += len(shot_gt_seq)

        if total_rallies > 0:
            results[target_player_name] = {
                'land_dtw': float(np.mean(rally_land_dtw_list)),
                'move_dtw': float(np.mean(rally_move_dtw_list)),
                'shot_ctc': float(np.mean(rally_shot_ctc_list)) if rally_shot_ctc_list else float('inf'),
                'edit_dist': float(np.mean(rally_edit_dist_list)),
                'num_rallies': total_rallies,
                'num_shots': total_shots,
            }
        else:
            results[target_player_name] = {
                'land_dtw': float('inf'),
                'move_dtw': float('inf'),
                'shot_ctc': float('inf'),
                'edit_dist': float('inf'),
                'num_rallies': 0,
                'num_shots': 0,
            }

    return results


def print_results(results):
    print('\n' + '=' * 105)
    print('ShuttleNet Evaluation Results on Test Set')
    print('=' * 105)
    print(f"{'Player Name':<30} {'Land DTW':<12} {'Move DTW':<12} {'Shot CTC':<12} {'Edit Dist':<12} {'Rallies':<8} {'Shots':<8}")
    print('-' * 105)
    land_scores, move_scores, ctc_scores, edit_scores, rally_counts, shot_counts = [], [], [], [], [], []
    for player_name, scores in sorted(results.items()):
        if scores['num_rallies'] > 0:
            print(f"{player_name:<30} {scores['land_dtw']:<12.4f} {scores['move_dtw']:<12.4f} "
                  f"{scores['shot_ctc']:<12.4f} {scores['edit_dist']:<12.4f} {scores['num_rallies']:<8} {scores['num_shots']:<8}")
            land_scores.append(scores['land_dtw'])
            move_scores.append(scores['move_dtw'])
            ctc_scores.append(scores['shot_ctc'])
            edit_scores.append(scores['edit_dist'])
            rally_counts.append(scores['num_rallies'])
            shot_counts.append(scores['num_shots'])
    print('-' * 105)
    if rally_counts:
        rally_sum = sum(rally_counts)
        print(f"{'Weighted Average (by rallies)':<30} "
              f"{sum(s * n for s, n in zip(land_scores, rally_counts)) / rally_sum:<12.4f} "
              f"{sum(s * n for s, n in zip(move_scores, rally_counts)) / rally_sum:<12.4f} "
              f"{sum(s * n for s, n in zip(ctc_scores, rally_counts)) / rally_sum:<12.4f} "
              f"{sum(s * n for s, n in zip(edit_scores, rally_counts)) / rally_sum:<12.4f} "
              f"{rally_sum:<8} {sum(shot_counts):<8}")
    print('=' * 105)


def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    base_evaluator_module = load_module_from_path(
        'shuttlenet_base_evaluator',
        os.path.join(project_path, 'Scripts', 'single_player_eval', 'evaluators', 'base_evaluator.py'),
    )
    trajectory_store_module = load_module_from_path(
        'shuttlenet_trajectory_store',
        os.path.join(project_path, 'Scripts', 'single_player_eval', 'trajectory_store.py'),
    )
    BaseEvaluator = base_evaluator_module.BaseEvaluator
    build_gt_shot_from_row = trajectory_store_module.build_gt_shot_from_row
    build_pred_shot = trajectory_store_module.build_pred_shot
    build_rollout_metadata = trajectory_store_module.build_rollout_metadata
except Exception:  # pragma: no cover - only used when single-player evaluator package is unavailable.
    BaseEvaluator = object
    build_gt_shot_from_row = None
    build_pred_shot = None
    build_rollout_metadata = None


class ShuttleNetEvaluator(BaseEvaluator):
    def __init__(self, player_name=None, experiment='rally', device='cuda', skip_first_n=0,
                 model_path=None, save_trajectories=False, sample=True, all_rallies=False):
        super().__init__(player_name, experiment, device, skip_first_n, save_trajectories=save_trajectories)
        self.all_rallies = all_rallies
        if model_path is None:
            model_path = self.config['paths']['shuttlenet']['model']
        self.model_path = model_path
        self.sample = sample

        with open(self.player_map_path, 'r') as f:
            self.player2index = json.load(f)
        if not self.all_rallies and self.player_name not in self.player2index:
            raise ValueError(f"Player '{self.player_name}' not found in player map")
        self.player_id = self.player2index.get(self.player_name, 0)
        self.policy = ShuttleNetPolicy(self.model_path, device=self.device, sample=self.sample)

        import pandas as pd

        self.df = pd.read_csv(self.merged_csv_path)
        target_cols = state_col + action_col
        self.df = self.df.dropna(subset=target_cols).reset_index(drop=True)
        self.print_runtime_parameters(
            'shuttlenet',
            sample=self.sample,
            extra_rows=[('shuttlenet.model_path', self.model_path)],
        )

    def get_algorithm_name(self):
        return 'shuttlenet'

    def simulate(self):
        dataset = load_rally_dataset(self.test_split_path, self.rally_profiles_path)
        records = dataset['test']
        if not self.all_rallies:
            records = [record for record in records if self.player_name in record['players']]
        selected_rows = filter_rally_rows(self.df, records)
        rally_frames = {
            tuple(int(value) for value in key): rally_df.sort_values('ball_round').reset_index(drop=True)
            for key, rally_df in selected_rows.groupby(['match_id', 'set', 'rally'], sort=False)
        }
        missing_keys = sorted({rally_key(record) for record in records} - set(rally_frames))
        if missing_keys:
            raise ValueError(f'No model-ready rows found for test rallies: {missing_keys[:5]}')
        rally_trajectories = []
        seen = set()
        description = 'Simulating physical rallies' if self.all_rallies else f'Simulating {self.player_name}'
        for record in tqdm(records, desc=description):
            key = rally_key(record)
            if key in seen:
                raise ValueError(f'Duplicate physical test rally: {key}')
            seen.add(key)
            match_id, set_num, rally_num = key
            rally_df = rally_frames[key]

            players = list(dict.fromkeys(rally_df['player'].astype(str)))
            if len(players) != 2:
                raise ValueError(f'Expected two players for rally {key}, found {players}')
            missing_players = [player for player in players if player not in self.player2index]
            if missing_players:
                raise ValueError(f'Players missing from player map for rally {key}: {missing_players}')
            player_ids = {player: self.player2index[player] for player in players}
            id_to_player = {player_id: player for player, player_id in player_ids.items()}

            rally_gt_full_seq = [build_gt_shot_from_row(row) for _, row in rally_df.iterrows()]
            max_rollout_steps = len(rally_gt_full_seq) + 2
            rally_pred_full_seq = []
            rollout = ShuttleNetRollout(self.policy)
            first_row = rally_df.iloc[0]
            current_player_name = str(first_row['player'])
            current_state = build_state_tensor_from_row(first_row, player_ids[current_player_name])
            termination_reason = 'max_rollout_reached'

            for _ in range(max_rollout_steps):
                current_player_id = int(current_state[17].item())
                if current_player_id not in id_to_player:
                    raise ValueError(f'Unknown player id {current_player_id} during rally {key}')
                current_player_name = id_to_player[current_player_id]
                launch = current_state[1].item() == 1
                ball_round = int(current_state[1].item())
                action_tensor, prediction = rollout.step(current_state, launch=launch)
                land_2d = prediction['landing']
                move_2d = prediction['moving']
                shot_type = int(prediction['shot_type'])
                rally_pred_full_seq.append(build_pred_shot(
                    current_player_name, shot_type, land_2d, move_2d, ball_round
                ))

                other_player_id = next(
                    player_id for player_id in id_to_player if player_id != current_player_id
                )
                next_state = transition(action_tensor.cpu(), current_state.cpu(), other_player_id)
                if terminate(next_state.cpu()):
                    termination_reason = 'natural_terminate'
                    break
                current_state = next_state

            target_player = None if self.all_rallies else self.player_name
            opponent_player = None
            if target_player in players:
                opponent_player = next(player for player in players if player != target_player)
            rally_trajectories.append({
                'match_id': int(match_id),
                'set': int(set_num),
                'rally': int(rally_num),
                'players': players,
                'player_profiles': record.get('player_profiles', {}),
                'instruction_mode': 'no_instruct',
                'target_player': target_player,
                'opponent_player': opponent_player,
                'gt_sequence': rally_gt_full_seq,
                'pred_sequence': rally_pred_full_seq,
                'target_player_won': None,
                **build_rollout_metadata(
                    rally_gt_full_seq,
                    rally_pred_full_seq,
                    termination_reason,
                ),
            })

        self._save_trajectories_if_needed(rally_trajectories)
        return rally_trajectories


def prepare_model_paths(args):
    default_split = 'Data/intermediate_data/rally_split.json'
    default_data_dir = 'Experiments/shuttlenet/data/rally'
    default_output = 'Experiments/shuttlenet/weight/rally_model.pth'
    default_eval_output = 'Experiments/shuttlenet/results/rally_eval.json'

    args.split = args.split or default_split
    args.data_dir = args.data_dir or default_data_dir
    args.output = args.output or default_output
    args.model_path = args.model_path or args.output
    args.eval_output = args.eval_output or default_eval_output
    return args


def run_prepare(args):
    prepare_shuttlenet_data(args.merged_csv, args.split, args.player_map, args.data_dir, args.encode_length)


def run_eval(args):
    raise RuntimeError(
        'Use Scripts/single_player_eval/eval_all_rallies.py --algorithm '
        'shuttlenet --no-instruct for physical-rally evaluation.'
    )


def parse_args():
    parser = argparse.ArgumentParser(description='ShuttleNet Agent - prepare, train, or evaluate')
    parser.add_argument('--mode', choices=['prepare', 'train', 'eval'], default='train')
    parser.add_argument('--merged_csv', default='Data/intermediate_data/merged.csv')
    parser.add_argument('--player_map', default='Data/intermediate_data/player_index_map.json')
    parser.add_argument('--split', default=None)
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--eval_output', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--encode_length', type=int, default=DEFAULT_ENCODE_LENGTH)
    parser.add_argument('--max_ball_round', type=int, default=DEFAULT_MAX_BALL_ROUND)
    parser.add_argument('--shot_dim', type=int, default=32)
    parser.add_argument('--area_dim', type=int, default=32)
    parser.add_argument('--player_dim', type=int, default=32)
    parser.add_argument('--encode_dim', type=int, default=32)
    parser.add_argument('--area_num', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--skip_first_n', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true', help='Use argmax/mean instead of sampling during eval')
    parser.add_argument('--reprepare', action='store_true', help='Rebuild train_samples.pkl before training')
    return prepare_model_paths(parser.parse_args())


if __name__ == '__main__':
    cli_args = parse_args()
    if cli_args.mode == 'prepare':
        run_prepare(cli_args)
    elif cli_args.mode == 'train':
        train_model(cli_args)
    elif cli_args.mode == 'eval':
        run_eval(cli_args)
