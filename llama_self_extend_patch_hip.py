# transfromers version 4.32.0
import os
from matplotlib import pyplot as plt
import skimage
import torch
from transformers.models.llama.modeling_llama import *
import numpy as np
import torch.nn as nn
import math
from typing import Optional, Tuple
import torch.nn.functional as F
from timber import timber_attention

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin) 
    k_embed = (k * cos) + (rotate_half(k) * sin) 
    return q_embed, k_embed

def apply_grouped_rotary_pos_emb(q, k, cos, sin, position_ids, g_size_1=4, g_size_2=2048):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    position_ids_q = position_ids//g_size_1 + g_size_2 - g_size_2//g_size_1
    # position_ids_q = position_ids
    position_ids_k = position_ids//g_size_1

    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos_q = cos[position_ids_q].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin_q = sin[position_ids_q].unsqueeze(1)  # [bs, 1, seq_len, dim]
    cos_k = cos[position_ids_k].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin_k = sin[position_ids_k].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos_q) + (rotate_half(q) * sin_q) 
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k) 
    return q_embed, k_embed

def self_extend_forward(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    group_size_1: Optional[float] = 4,
    group_size_2: Optional[float] = 2048,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    
    is_self_extend = True
    
    if self.layer_index >= 0:
        # neighbor_query_states, neighbor_key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids) # normal attention 
    
        # # ********************************************************************************************************************* #

        # _re_group_size_2 = 0 if position_ids.max() < group_size_2 else group_size_2 # in case that, the smallest q position, g2-g2//g1 exceed the max position
        # group_query_states, group_key_states = apply_grouped_rotary_pos_emb(query_states, key_states, cos, sin, position_ids, g_size_1=group_size_1, g_size_2=_re_group_size_2) # grouped attention


        # if past_key_value is not None:
        #     # reuse k, v, self_attention
        #     neighbor_key_states = torch.cat([past_key_value[0], neighbor_key_states], dim=2)
        #     group_key_states = torch.cat([past_key_value[1], group_key_states], dim=2)     # cache group_key_states

        # group_key_states = repeat_kv(group_key_states, self.num_key_value_groups)
        # neighbor_key_states = repeat_kv(neighbor_key_states, self.num_key_value_groups)
        
        # neighbor_attn_weights = torch.matmul(neighbor_query_states, neighbor_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        # group_attn_weights = torch.matmul(group_query_states, group_key_states.transpose(2, 3)) / math.sqrt(self.head_dim) 

        # if group_attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        #     raise ValueError(
        #         f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
        #         f" {group_attn_weights.size()}"
        #     )
        
        # if attention_mask is not None:
        #     if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
        #         raise ValueError(
        #             f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
        #         )
        #     group_attn_weights = group_attn_weights + attention_mask
        #     neighbor_attn_weights = neighbor_attn_weights + attention_mask # causal mask. 
        
        # if q_len == 1:
        #     # take effect with KV cache. 
        #     neighbor_attention_mask = torch.zeros((q_len, kv_seq_len), device=neighbor_attn_weights.device)
        #     neighbor_attention_mask[:, -group_size_2:] = 1
        # elif q_len == kv_seq_len:
        #     # no cache OR prefill
        #     neighbor_attention_mask = torch.ones((q_len, kv_seq_len), device=neighbor_attn_weights.device)
        #     neighbor_attention_mask = torch.tril(neighbor_attention_mask)
        #     if q_len-group_size_2 > 0:
        #         # seq length is larger than group_size_2, should do replacement. 
        #         group_attention_mask =  torch.tril(torch.ones((q_len-group_size_2, kv_seq_len-group_size_2), device=group_attn_weights.device))
        #         neighbor_attention_mask[group_size_2:, :-group_size_2] -= group_attention_mask

        # else:
        #     raise ValueError("q_len should be 1 or seq_len.")

        # merged_attn_weights = torch.where(neighbor_attention_mask.bool(), neighbor_attn_weights, group_attn_weights) # replace the group attention with neighbor attention within the neighbor window. 
        # merged_attn_weights = nn.functional.softmax(merged_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) 

        # # ********************************************************************************************************************* #

        # x = merged_attn_weights.detach().clone()
        # x.scatter_(
        #     dim=-1, index=torch.topk(x, k=512, dim=-1).indices, value=1.0
        # )
        # x = skimage.measure.block_reduce(x.cpu().numpy()[0,0], (1, 1), np.max) ** 0.2
        # # x = np.repeat(x, BLOCK_SIZE_Q, 0)
        # # x = np.repeat(x, 1, 1)
        # if x.shape[0] == 1:
        #     x = x.repeat(32, 0)
        # plt.clf()
        # plt.title(f'sum:{x.sum()}')
        # plt.imshow(x)
        # plt.colorbar()
        # os.makedirs('saves/models/self_extend', exist_ok=True)
        # path = f'saves/models/self_extend/debug.png'
        # # path = f'saves/models/timber_attention/block.png'
        # print('saved', path)
        # plt.savefig(path, dpi=96, bbox_inches='tight')
        
        if not is_self_extend:
            query_states, key_states = apply_rotary_pos_emb(
                query_states, 
                key_states, 
                cos, 
                sin, 
                position_ids
            )
        
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        if use_cache:
            past_key_value = (key_states, value_states) 
        else:
            past_key_value = None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # print(query_states.shape, key_states.shape, value_states.shape, position_ids.shape, cos.shape, sin.shape)
        # [1, 16, 1, 128] [1, 16, 3770, 128] [1, 16, 3770, 128] [1, 1] [1, 1, 3770, 128] [1, 1, 3770, 128]
        
        _cos = cos.squeeze(0).squeeze(0)
        _sin = sin.squeeze(0).squeeze(0)
        _pids = position_ids.repeat_interleave(self.num_heads, 0)
        
        # print(_cos.data_ptr(), _cos.stride(), _sin.data_ptr(), _sin.stride(), _pids.data_ptr(), _pids.stride(), _pids.shape)
        # print(_pids)
        
        attn_output = timber_attention(
            query_states.view(bsz * self.num_heads, q_len, self.head_dim) / math.sqrt(self.head_dim), 
            key_states.view(bsz * self.num_heads, kv_seq_len, self.head_dim),
            value_states.view(bsz * self.num_heads, kv_seq_len, self.head_dim),
            
            mask_k=1024,
            block_size_q=16,
            block_size_k=2,
            dense_queries_exp=0,
            
            rope_method='self_extend' if is_self_extend else 'none',
            rope_cos=_cos,
            rope_sin=_sin,
            position_ids=_pids,
        )[0].view(bsz, self.num_heads, q_len, self.head_dim)
    else:
        neighbor_query_states, neighbor_key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids) # normal attention 
    
        # ********************************************************************************************************************* #

        _re_group_size_2 = 0 if position_ids.max() < group_size_2 else group_size_2 # in case that, the smallest q position, g2-g2//g1 exceed the max position
        group_query_states, group_key_states = apply_grouped_rotary_pos_emb(query_states, key_states, cos, sin, position_ids, g_size_1=group_size_1, g_size_2=_re_group_size_2) # grouped attention


        if past_key_value is not None:
            # reuse k, v, self_attention
            neighbor_key_states = torch.cat([past_key_value[0], neighbor_key_states], dim=2)
            group_key_states = torch.cat([past_key_value[1], group_key_states], dim=2)     # cache group_key_states
            value_states = torch.cat([past_key_value[2], value_states], dim=2)

        if use_cache:
            past_key_value = (neighbor_key_states, group_key_states, value_states) 
        else:
            past_key_value = None

        group_key_states = repeat_kv(group_key_states, self.num_key_value_groups)
        neighbor_key_states = repeat_kv(neighbor_key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        neighbor_attn_weights = torch.matmul(neighbor_query_states, neighbor_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        group_attn_weights = torch.matmul(group_query_states, group_key_states.transpose(2, 3)) / math.sqrt(self.head_dim) 

        if group_attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {group_attn_weights.size()}"
            )
        
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            group_attn_weights = group_attn_weights + attention_mask
            neighbor_attn_weights = neighbor_attn_weights + attention_mask # causal mask. 
        

        if q_len == 1:
            # take effect with KV cache. 
            neighbor_attention_mask = torch.zeros((q_len, kv_seq_len), device=neighbor_attn_weights.device)
            neighbor_attention_mask[:, -group_size_2:] = 1
        elif q_len == kv_seq_len:
            # no cache OR prefill
            neighbor_attention_mask = torch.ones((q_len, kv_seq_len), device=neighbor_attn_weights.device)
            neighbor_attention_mask = torch.tril(neighbor_attention_mask)
            if q_len-group_size_2 > 0:
                # seq length is larger than group_size_2, should do replacement. 
                group_attention_mask =  torch.tril(torch.ones((q_len-group_size_2, kv_seq_len-group_size_2), device=group_attn_weights.device))
                neighbor_attention_mask[group_size_2:, :-group_size_2] -= group_attention_mask

        else:
            raise ValueError("q_len should be 1 or seq_len.")

        merged_attn_weights = torch.where(neighbor_attention_mask.bool(), neighbor_attn_weights, group_attn_weights) # replace the group attention with neighbor attention within the neighbor window. 
        merged_attn_weights = nn.functional.softmax(merged_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) 

        # ********************************************************************************************************************* #

        # x = merged_attn_weights.detach().clone()
        # x.scatter_(
        #     dim=-1, index=torch.topk(x, k=512, dim=-1).indices, value=1.0
        # )
        # x = skimage.measure.block_reduce(x.cpu().numpy()[0,0], (1, 1), np.max) ** 0.2
        # # x = np.repeat(x, BLOCK_SIZE_Q, 0)
        # # x = np.repeat(x, 1, 1)
        # if x.shape[0] == 1:
        #     x = x.repeat(32, 0)
        # plt.clf()
        # plt.title(f'sum:{x.sum()}')
        # plt.imshow(x)
        # plt.colorbar()
        # os.makedirs('saves/models/self_extend', exist_ok=True)
        # path = f'saves/models/self_extend/debug.png'
        # # path = f'saves/models/timber_attention/block.png'
        # print('saved', path)
        # plt.savefig(path, dpi=96, bbox_inches='tight')
        
        attn_output = torch.matmul(merged_attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
