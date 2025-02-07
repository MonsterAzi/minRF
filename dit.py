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
    def __init__(self, dim):
        super().__init__()
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)

    def forward(self, x):
        bsz, seqlen, _ = x.shape
        dtype = x.dtype

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.permute(0, 2, 1)
        xk = xk.permute(0, 2, 1)
        xv = xv.permute(0, 2, 1)

        attn_weight = xq @ xk.transpose(-2, -1)
        attn_weight = torch.softmax(attn_weight, dim=-1)
        output = attn_weight @ xv
        output = output.permute(0, 2, 1).flatten(-1)

        return self.wo(output)

class MoEFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, num_experts, ffn_dim_multiplier=None):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            FeedForward(dim, hidden_dim, multiple_of, ffn_dim_multiplier)
            for _ in range(num_experts)
        ])
        self.router = nn.Linear(dim, num_experts)

    def forward(self, x):
        B, SeqLen, Dim = x.shape
        num_experts = self.num_experts
        top_k = min(2, num_experts)

        # 1. Routing probabilities and top-k expert selection
        router_logits = F.normalize(self.router(x))
        router_logits_mean = router_logits.mean(dim=-1, keepdim=True)
        router_logits_std = router_logits.std(dim=-1, keepdim=True)
        router_logits_normalized = (router_logits - router_logits_mean) / (router_logits_std + 1e-6) # Add small epsilon for numerical stability
        router_probs = F.softmax(router_logits, dim=-1)
        expert_weights, expert_indices = torch.topk(router_probs, k=top_k, dim=-1, sorted=False)

        # Flatten input for efficient indexing
        x_flat = x.reshape(-1, Dim)  # (B * SeqLen, Dim)
        expert_indices_flat = expert_indices.reshape(-1, top_k)  # (B * SeqLen, top_k)
        expert_weights_flat = expert_weights.reshape(-1, top_k)  # (B * SeqLen, top_k)

        # Initialize output tensor
        moe_output_flat = torch.zeros_like(x_flat)

        # Vectorized Grouping and Expert Processing (Iterate through unique experts)
        batch_seq_indices = torch.arange(B * SeqLen, device=x.device)
        expanded_batch_seq_indices = batch_seq_indices.unsqueeze(-1).repeat(1, top_k)

        global_token_indices = expanded_batch_seq_indices.flatten()
        expert_indices_for_tokens = expert_indices_flat.flatten()
        expert_weights_for_tokens = expert_weights_flat.flatten()

        # Find unique expert indices that are actually used
        unique_expert_indices = torch.unique(expert_indices_for_tokens)

        # Process each expert using vectorized operations and scatter directly
        for expert_idx in unique_expert_indices: # Iterate through unique experts
            expert_idx = expert_idx.item() # Convert to int for comparison
            expert_mask = (expert_indices_for_tokens == expert_idx)
            if expert_mask.any():
                current_expert_token_indices = global_token_indices[expert_mask]
                current_expert_weights = expert_weights_for_tokens[expert_mask]
                current_expert_input = x_flat[current_expert_token_indices]

                # Expert forward pass
                current_expert_output = self.experts[expert_idx](current_expert_input)

                # Weighted output (no normalization now)
                weighted_expert_output = current_expert_output * current_expert_weights.unsqueeze(-1)

                # Scatter the weighted output (no normalization)
                moe_output_flat.index_add_(0, current_expert_token_indices, weighted_expert_output)


        # Reshape back to original shape
        moe_output = moe_output_flat.reshape(B, SeqLen, Dim)

        # 5. Auxiliary Loss (remains the same)
        average_router_probs_per_expert = router_probs.mean(dim=0) # Average prob for each expert over tokens
        target_prob = 1.0 / num_experts # Ideal proportion (assuming simplification to 1/n)
        aux_loss = torch.sum((target_prob - average_router_probs_per_expert)**2)

        return moe_output, aux_loss

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
            self.attention = Attention(dim)
            self.feed_forward = MoEFeedForward(
                dim=dim,
                hidden_dim=4 * dim,
                multiple_of=multiple_of,
                num_experts=num_moe_experts,
                ffn_dim_multiplier=ffn_dim_multiplier,
            )
        else:
            self.attention = Attention(dim)
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
        aux_loss = torch.tensor([0.0], device=x.device) # Initialize aux_loss
        if adaln_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(adaln_input).chunk(6, dim=1)
            )

            x = x + gate_msa.unsqueeze(1) * self.attention(
                modulate(self.attention_norm(x), shift_msa, scale_msa)
            )
            if self.num_moe_experts > 1:
                x, block_aux_loss = self.feed_forward(
                    modulate(self.ffn_norm(x), shift_mlp, scale_mlp)
                )
                aux_loss = block_aux_loss # Accumulate aux_loss from MoE block
            else:
                x = x + gate_mlp.unsqueeze(1) * self.feed_forward(
                    modulate(self.ffn_norm(x), shift_mlp, scale_mlp)
                )

        else:
            x = x + self.attention(self.attention_norm(x))
            if self.num_moe_experts > 1:
                 x, block_aux_loss = self.feed_forward(self.ffn_norm(x))
                 aux_loss = block_aux_loss # Accumulate aux_loss from MoE block
            else:
                x = x + self.feed_forward(self.ffn_norm(x))

        return x, aux_loss # Return aux_loss from block

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

        total_aux_loss = torch.tensor([0.0], device=x.device) # Initialize total aux loss
        for step in range(self.n_steps):
            for layer in self.layers:
                x, layer_aux_loss = layer(x, self.freqs_cis[: x.size(1)], adaln_input=adaln_input)
                total_aux_loss += layer_aux_loss # Accumulate aux loss from each layer

        x = self.final_layer(x, adaln_input)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        return x, total_aux_loss # Return total aux loss

    def forward_with_cfg(self, x, t, y, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[0][:, : self.in_channels], model_out[0][:, self.in_channels :]
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
        out_moe, aux_loss_moe = model_moe(x, t, y)
        print(f"MoE Output shape: {out_moe.shape}, Aux Loss: {aux_loss_moe.item():.4f}")
        out_moe_cfg = model_moe.forward_with_cfg(x, t, y, 0.5)
        print(f"MoE CFG Output shape: {out_moe_cfg.shape}")

    # Test without MoE
    model_no_moe = DiUT_Llama_600M_patch2(num_moe_experts=1)
    model_no_moe.eval()

    with torch.no_grad():
        out_no_moe, aux_loss_no_moe = model_no_moe(x, t, y)
        print(f"No MoE Output shape: {out_no_moe.shape}, Aux Loss: {aux_loss_no_moe.item():.4f}")
        out_no_moe_cfg = model_no_moe.forward_with_cfg(x, t, y, 0.5)
        print(f"No MoE CFG Output shape: {out_no_moe_cfg.shape}")
        
    import time
        
    # Benchmark MoE model
    start_time = time.time()
    torch.cuda.synchronize() # Wait for CUDA operations to complete
    for _ in range(25): # Run forward pass multiple times for better averaging
        out_moe, aux_loss_moe = model_moe(x, t, y)
    torch.cuda.synchronize()
    moe_forward_time = (time.time() - start_time) / 25
    print(f"MoE Forward Pass Time (averaged over 25 runs): {moe_forward_time:.4f} seconds")

    # Benchmark No-MoE model
    start_time = time.time()
    torch.cuda.synchronize()
    for _ in range(25): # Run forward pass multiple times for better averaging
        out_no_moe, aux_loss_no_moe = model_no_moe(x, t, y)
    torch.cuda.synchronize()
    no_moe_forward_time = (time.time() - start_time) / 25
    print(f"No MoE Forward Pass Time (averaged over 25 runs): {no_moe_forward_time:.4f} seconds")