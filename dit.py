# Code heavily based on https://github.com/Alpha-VLLM/LLaMA2-Accessory
# this is modeling code for DiT-LLaMA model

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            dtype=next(self.parameters()).dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb

class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = int(dropout_prob > 0)
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0]) < self.dropout_prob
            drop_ids = drop_ids.cuda()
            drop_ids = drop_ids.to(labels.device)
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier=None):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _forward_sin_gating(self, x1, x3):
        return torch.sin(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_sin_gating(self.w1(x), self.w3(x)))
    
class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.LayerNorm(self.n_heads * self.head_dim)
        self.k_norm = nn.LayerNorm(self.n_heads * self.head_dim)

    @staticmethod
    def reshape_for_broadcast(freqs_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        _freqs_cis = freqs_cis[: x.shape[1]]
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return _freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(xq, xk, freqs_cis):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis_xq = Attention.reshape_for_broadcast(freqs_cis, xq_)
        freqs_cis_xk = Attention.reshape_for_broadcast(freqs_cis, xk_)

        xq_out = torch.view_as_real(xq_ * freqs_cis_xq).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis_xk).flatten(3)
        return xq_out, xk_out

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape
        dtype = x.dtype

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq, xk = xq.to(dtype), xk.to(dtype)

        xq = xq.permute(0, 2, 1, 3)
        xk = xk.permute(0, 2, 1, 3)
        xv = xv.permute(0, 2, 1, 3)

        scale_factor = 1 / math.sqrt(xq.size(-1))
        attn_weight = xq @ xk.transpose(-2, -1) * scale_factor
        attn_weight = torch.softmax(attn_weight, dim=-1)
        output = attn_weight @ xv
        output = output.permute(0, 2, 1, 3).flatten(-2)

        return self.wo(output)

class LongTermMemory(nn.Module):
    def __init__(self, dim, n_heads, memory_layers=2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False) # Output dim should be dim, not n_heads * head_dim in this case
        self.q_norm = nn.LayerNorm(self.n_heads * self.head_dim)
        self.k_norm = nn.LayerNorm(self.n_heads * self.head_dim)

        # Memory Module (MLP)
        memory_layers_list = []
        for _ in range(memory_layers - 1):
            memory_layers_list.append(nn.Linear(self.head_dim, self.head_dim))
            memory_layers_list.append(nn.ReLU()) # Or other activation
        memory_layers_list.append(nn.Linear(self.head_dim, self.head_dim))
        self.memory_mlp = nn.Sequential(*memory_layers_list)

        # Learnable parameters for memory control (alpha, eta, theta) - using simple Linear for now
        self.alpha_proj = nn.Linear(dim, self.n_heads * 1) # Output 1 per head, then average or similar
        self.eta_proj = nn.Linear(dim, self.n_heads * 1)
        self.theta_proj = nn.Linear(dim, self.n_heads * 1)
        self.sigmoid = nn.Sigmoid() # To keep alpha, eta, theta in [0, 1] or similar range
        self.relu = nn.ReLU() # For theta, ensure non-negative

        # Initialize memory and surprise - head-specific
        self.register_buffer('memory', torch.zeros(1, n_heads, self.head_dim)) # (1, n_heads, head_dim) - initialized per head
        self.register_buffer('surprise', torch.zeros(1, n_heads, self.head_dim)) # (1, n_heads, head_dim) - initialized per head


    @staticmethod
    def reshape_for_broadcast(freqs_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        _freqs_cis = freqs_cis[: x.shape[1]]
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return _freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(xq, xk, freqs_cis):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis_xq = LongTermMemory.reshape_for_broadcast(freqs_cis, xq_)
        freqs_cis_xk = LongTermMemory.reshape_for_broadcast(freqs_cis, xk_)

        xq_out = torch.view_as_real(xq_ * freqs_cis_xq).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis_xk).flatten(3)
        return xq_out, xk_out

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape
        dtype = x.dtype

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq, xk = xq.to(dtype), xk.to(dtype)

        output_tokens = []
        # Expand memory and surprise to batch size at the beginning of the sequence
        current_memory = self.memory.expand(bsz, -1, -1) # (bsz, n_heads, head_dim)
        current_surprise = self.surprise.expand(bsz, -1, -1) # (bsz, n_heads, head_dim)


        for t in range(seqlen):
            xt = x[ :, t, :] # (bsz, dim)
            qt = xq[:, t, :, :] # (bsz, n_heads, head_dim)
            kt = xk[:, t, :, :] # (bsz, n_heads, head_dim)
            vt = xv[:, t, :, :] # (bsz, n_heads, head_dim)

            # Memory control parameters - data-dependent
            alpha_t_raw = self.alpha_proj(xt).view(bsz, self.n_heads, 1) # (bsz, n_heads, 1)
            eta_t_raw = self.eta_proj(xt).view(bsz, self.n_heads, 1)
            theta_t_raw = self.theta_proj(xt).view(bsz, self.n_heads, 1)

            alpha_t = self.sigmoid(alpha_t_raw) # Ensure between 0 and 1
            eta_t = self.sigmoid(eta_t_raw) # Ensure between 0 and 1
            theta_t = self.relu(theta_t_raw) # Ensure non-negative

            # Loss and Surprise - using direct difference as surprise signal
            memory_readout = self.memory_mlp(kt) # M_{t-1}(k_t) - (bsz, n_heads, head_dim)
            surprise_signal = memory_readout - vt # (bsz, n_heads, head_dim) -  M_{t-1}(k_t) - v_t

            # Memory Update - Equations (13) and (14)
            current_surprise = eta_t * current_surprise - theta_t * surprise_signal # S_t = eta_t * S_{t-1} - theta_t * (M_{t-1}(k_t) - v_t)
            current_memory = (1 - alpha_t) * current_memory + current_surprise # M_t = (1 - alpha_t) * M_{t-1} + S_t

            # Memory Retrieval - Equation (15) - M*(q_t)
            yt = self.memory_mlp(qt) # M*(q_t) - (bsz, n_heads, head_dim)
            output_tokens.append(yt)


        output = torch.stack(output_tokens, dim=1) # (bsz, seqlen, n_heads, head_dim)
        output = output.flatten(2, 3) # Flatten n_heads and head_dim to get (bsz, seqlen, dim)

        # Update persistent memory (for next forward pass in sequence)
        # Take the last batch's memory state and store it as the persistent memory
        self.memory.copy_(current_memory[:1].detach()) # Detach and take only the first batch item to update persistent memory
        self.surprise.copy_(current_surprise[:1].detach()) # Detach and take only the first batch item for surprise


        return self.wo(output)

class MoEAttention(nn.Module):
    def __init__(self, dim, n_heads, num_experts=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.num_experts = num_experts

        self.wq = nn.ModuleList([nn.Linear(dim, n_heads * self.head_dim, bias=False) for _ in range(num_experts)])
        self.wk = nn.ModuleList([nn.Linear(dim, n_heads * self.head_dim, bias=False) for _ in range(num_experts)])
        self.wv = nn.ModuleList([nn.Linear(dim, n_heads * self.head_dim, bias=False) for _ in range(num_experts)])
        self.wo = nn.ModuleList([nn.Linear(n_heads * self.head_dim, dim, bias=False) for _ in range(num_experts)])

        self.q_norm = nn.ModuleList([nn.LayerNorm(self.n_heads * self.head_dim) for _ in range(num_experts)])
        self.k_norm = nn.ModuleList([nn.LayerNorm(self.n_heads * self.head_dim) for _ in range(num_experts)])

        self.gate = nn.Linear(dim, num_experts)

    @staticmethod
    def reshape_for_broadcast(freqs_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        _freqs_cis = freqs_cis[: x.shape[1]]
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return _freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(xq, xk, freqs_cis):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis_xq = MoEAttention.reshape_for_broadcast(freqs_cis, xq_)
        freqs_cis_xk = MoEAttention.reshape_for_broadcast(freqs_cis, xk_)

        xq_out = torch.view_as_real(xq_ * freqs_cis_xq).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis_xk).flatten(3)
        return xq_out, xk_out

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape
        dtype = x.dtype

        # Gating
        gate_logits = self.gate(x)
        gate_weights = torch.sigmoid(gate_logits)

        expert_outputs = []
        for i in range(self.num_experts):
            xq_e, xk_e, xv_e = self.wq[i](x), self.wk[i](x), self.wv[i](x)

            xq_e = self.q_norm[i](xq_e)
            xk_e = self.k_norm[i](xk_e)

            xq_e = xq_e.view(bsz, seqlen, self.n_heads, self.head_dim)
            xk_e = xk_e.view(bsz, seqlen, self.n_heads, self.head_dim)
            xv_e = xv_e.view(bsz, seqlen, self.n_heads, self.head_dim)

            xq_e, xk_e = self.apply_rotary_emb(xq_e, xk_e, freqs_cis=freqs_cis)
            xq_e, xk_e = xq_e.to(dtype), xk_e.to(dtype)

            xq_e = xq_e.permute(0, 2, 1, 3)
            xk_e = xk_e.permute(0, 2, 1, 3)
            xv_e = xv_e.permute(0, 2, 1, 3)

            scale_factor = 1 / math.sqrt(xq_e.size(-1))
            attn_weight_e = xq_e @ xk_e.transpose(-2, -1) * scale_factor
            attn_weight_e = torch.softmax(attn_weight_e, dim=-1)
            output_e = attn_weight_e @ xv_e
            output_e = output_e.permute(0, 2, 1, 3).flatten(-2)
            expert_outputs.append(self.wo[i](output_e))

        # Combine expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=0)  # (num_experts, bsz, seqlen, dim)
        output = torch.einsum("ebsd,bse->bsd", expert_outputs, gate_weights)
        return output

class MoEFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, num_experts=4, ffn_dim_multiplier=None):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            FeedForward(dim, hidden_dim, multiple_of, ffn_dim_multiplier) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x):
        # Gating
        gate_logits = self.gate(x)
        gate_weights = torch.sigmoid(gate_logits)

        # Expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=0)  # (num_experts, bsz, seqlen, dim)

        # Combine expert outputs
        output = torch.einsum("ebsd,bse->bsd", expert_outputs, gate_weights)
        return output

class UTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id,
        dim,
        n_heads,
        multiple_of,
        ffn_dim_multiplier,
        norm_eps,
        num_moe_experts=4,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.layer_id = layer_id
        self.attention_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.num_moe_experts = num_moe_experts

        if num_moe_experts > 1:
            self.attention = MoEAttention(dim, n_heads, num_experts=num_moe_experts)
            self.feed_forward = MoEFeedForward(
                dim=dim,
                hidden_dim=4 * dim,
                multiple_of=multiple_of,
                num_experts=num_moe_experts,
                ffn_dim_multiplier=ffn_dim_multiplier,
            )
        else:
            self.attention = Attention(dim, n_heads)
            self.feed_forward = FeedForward(
                dim=dim,
                hidden_dim=4 * dim,
                multiple_of=multiple_of,
                ffn_dim_multiplier=ffn_dim_multiplier,
            )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(dim, 1024), 6 * dim, bias=True),
        )

    def forward(self, x, freqs_cis, adaln_input=None):
        if adaln_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(adaln_input).chunk(6, dim=1)
            )

            x = x + gate_msa.unsqueeze(1) * self.attention(
                modulate(self.attention_norm(x), shift_msa, scale_msa), freqs_cis
            )
            x = x + gate_mlp.unsqueeze(1) * self.feed_forward(
                modulate(self.ffn_norm(x), shift_mlp, scale_mlp)
            )
        else:
            x = x + self.attention(self.attention_norm(x), freqs_cis)
            x = x + self.feed_forward(self.ffn_norm(x))

        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, 1024), 2 * hidden_size, bias=True),
        )
        # # init zero
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiUT_Llama(nn.Module):
    def __init__(
        self,
        in_channels=3,
        input_size=32,
        patch_size=2,
        dim=512,
        n_steps=5,
        n_layers=1,
        n_heads=16,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        class_dropout_prob=0.1,
        num_classes=10,
        num_moe_experts=4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_moe_experts = num_moe_experts

        self.init_conv_seq = nn.Sequential(
            nn.Conv2d(in_channels, dim // 2, kernel_size=5, padding=2, stride=1),
            nn.SiLU(),
            nn.GroupNorm(32, dim // 2),
            nn.Conv2d(dim // 2, dim // 2, kernel_size=5, padding=2, stride=1),
            nn.SiLU(),
            nn.GroupNorm(32, dim // 2),
        )

        self.x_embedder = nn.Linear(patch_size * patch_size * dim // 2, dim, bias=True)
        nn.init.constant_(self.x_embedder.bias, 0)

        self.t_embedder = TimestepEmbedder(min(dim, 1024))
        self.y_embedder = LabelEmbedder(num_classes, min(dim, 1024), class_dropout_prob)

        self.layers = nn.ModuleList(
            [
                UTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    num_moe_experts=num_moe_experts,
                )
                for layer_id in range(n_layers)
            ]
        )
        self.n_steps = n_steps

        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)

        self.freqs_cis = DiUT_Llama.precompute_freqs_cis(dim // n_heads, 4096)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def patchify(self, x):
        B, C, H, W = x.size()
        x = x.view(
            B,
            C,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
        return x

    def forward(self, x, t, y):
        self.freqs_cis = self.freqs_cis.to(x.device)

        x = self.init_conv_seq(x)

        x = self.patchify(x)
        x = self.x_embedder(x)

        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        adaln_input = t.to(x.dtype) + y.to(x.dtype)

        for step in range(self.n_steps):
            for layer in self.layers:
                x = layer(x, self.freqs_cis[: x.size(1)], adaln_input=adaln_input)

        x = self.final_layer(x, adaln_input)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    @staticmethod
    def precompute_freqs_cis(dim, end, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

def DiUT_Llama_600M_patch2(**kwargs):
    return DiUT_Llama(patch_size=2, dim=256, n_steps=16, n_layers=2, n_heads=32, **kwargs)

def DiUT_Llama_3B_patch2(**kwargs):
    return DiUT_Llama(patch_size=2, dim=3072, n_steps=32, n_heads=32, **kwargs)

if __name__ == "__main__":
    # Test with MoE
    model_moe = DiUT_Llama_600M_patch2(num_moe_experts=4)
    model_moe.eval()
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 100, (2,))
    y = torch.randint(0, 10, (2,))

    with torch.no_grad():
        out_moe = model_moe(x, t, y)
        print(f"MoE Output shape: {out_moe.shape}")
        out_moe_cfg = model_moe.forward_with_cfg(x, t, y, 0.5)
        print(f"MoE CFG Output shape: {out_moe_cfg.shape}")

    # Test without MoE
    model_no_moe = DiUT_Llama_600M_patch2(num_moe_experts=1)
    model_no_moe.eval()

    with torch.no_grad():
        out_no_moe = model_no_moe(x, t, y)
        print(f"No MoE Output shape: {out_no_moe.shape}")
        out_no_moe_cfg = model_no_moe.forward_with_cfg(x, t, y, 0.5)
        print(f"No MoE CFG Output shape: {out_no_moe_cfg.shape}")