import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tritlin import linear_recurrence
import torch.utils.checkpoint as checkpoint
import torch.optim as optim

MODEL_CHUNK_SIZE = 128

def soft_clamp(x: torch.Tensor, min_val: float = None, max_val: float = None) -> torch.Tensor:
    eps = 1e-1
    if min_val is not None:
        diff = min_val - x
        x = (x + min_val + diff*diff/(diff.abs() + eps) + eps)/2
    if max_val is not None:
        diff = max_val - x
        x = (x + max_val - diff*diff/(diff.abs() + eps) - eps)/2
    return x


def parallel_scan_complex(a: torch.Tensor, b: torch.Tensor, h_init: torch.Tensor) -> torch.Tensor:
    """
    Parallel associative scan for complex linear recurrence:
        h_t = a_t * h_{t-1} + b_t
    
    Runs in O(log T) parallel steps on GPU.

    Args:
        a: [B, T, D] Complex transition tensor
        b: [B, T, D] Complex driving input tensor
    Returns:
        b: [B, T, D] Complex state trajectory h_t for all t
    """
    B, T, D = a.shape
    
    b[:, 0:1, :] = b[:, 0:1, :] + a[:, 0:1, :] * h_init
    d = 1
    
    ones = torch.ones_like(a)
    zeros = torch.zeros_like(b)
    while d < T:
        a_shifted = torch.cat([ones[..., T-d:,:], a[..., :T-d,:]], dim=1)
        b_shifted = torch.cat([zeros[..., T-d:,:], b[..., :T-d,:]], dim=1)

        new_b = a * b_shifted + b
        new_a = a * a_shifted
        
        a, b = new_a, new_b
        d *= 2
    return b

class GatedState(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, is_complex: bool = False, use_inversion: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.is_complex = is_complex
        self.use_inversion = use_inversion
        if is_complex:
            hidden_size = hidden_size * 2
        self.gate_proj = nn.Linear(input_size, hidden_size)
        self.state_proj = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if self.is_complex:
            state = torch.cat([state.real, state.imag], dim=-1)
        gate = self.gate_proj(x)
        if self.use_inversion:
            gate = F.softsign(gate)
        else:
            gate = torch.sigmoid(gate)
        state = state * gate
        res = self.state_proj(state)
        return res

def generate_halton_centers(num_centers: int, l_max: int) -> torch.Tensor:
    """Generates quasi-random, maximally spaced center positions using Halton Base-2."""
    tau = torch.zeros(num_centers)
    for i in range(num_centers):
        f = 1.0
        r = 0.0
        index = i + 1
        while index > 0:
            f /= 2.0
            r += f * (index % 2)
            index //= 2
        tau[i] = r * l_max
    return tau
 
class S4DEncoder_state(nn.Module):
    """
    S4D Encoder returning the direct complex state grid h_t.
    
    Returns State h shape: [Batch, Seq_Len, state_dim] (Complex)
    
    Args:
        state_dim (int): Number of HiPPO polynomial modes N.
        l_max (int): Maximum context horizon.
        sigma_ratio (float): Width of Gaussian centers.
    """
    def __init__(
        self, 
        state_dim: int
    ):
        super().__init__()
        self.state_dim = state_dim        # N (HiPPO modes)

        # 2. Pure HiPPO Operators (A and B)
        n = torch.arange(state_dim, dtype=torch.float32)
        
        omega = math.pi * n                                 # Frequencies [N]
        self.log_delta = nn.Parameter(torch.log(torch.ones(state_dim) * 1e-3))
        self.log_alpha = nn.Parameter(torch.log(torch.ones(state_dim) * 0.5))
        self.omega = nn.Parameter(omega)

    def reshape_for_linear_recurrence(self, x: torch.Tensor, B_size: int, inT: int) -> torch.Tensor:
        return x.reshape(B_size, inT, self.state_dim//2, 2).permute(0, 2, 1, 3).reshape(B_size * self.state_dim//2, inT, 2)

    def reshape_for_output(self, x: torch.Tensor, B_size: int, inT: int) -> torch.Tensor:
        return x.reshape(B_size,self.state_dim//2, inT, 2).permute(0, 2, 1, 3).reshape(B_size, inT, self.state_dim)

    def forward(self, x: torch.Tensor, contribution_factor: torch.Tensor, h_prev: torch.Tensor = None) -> torch.Tensor:
        """
        Input x:   [Batch, Seq_Len, d_model] (Real)
        Returns h: [Batch, Seq_Len, K_centers, N_modes, d_model] (Complex)
        """
        B_size, L, D = x.shape
        device = x.device
        delta = torch.exp(soft_clamp(self.log_delta, min_val=-25, max_val=-2))
        x_impulse = x * delta

        if h_prev is None:
            h_prev = torch.zeros(B_size* self.state_dim//2, 2, dtype=torch.complex64, device=device)
        
        alpha = torch.exp(-soft_clamp(self.log_alpha, min_val=-25, max_val=25))
        complement = 1 - contribution_factor
        # A_sys = self.A_vector.view(1, 1, self.state_dim)  # [1, 1, N]
        A_sys = torch.exp(-(1j * self.omega + alpha) * delta).view(1, 1, self.state_dim)
        A_sys = complement + A_sys * contribution_factor
        x_impulse = x_impulse * contribution_factor
        x_impulse = self.reshape_for_linear_recurrence(x_impulse, B_size, L)
        A_sys = self.reshape_for_linear_recurrence(A_sys, B_size, L)
        h = linear_recurrence(x_impulse, A_sys, h_prev)
        h_prev = h[:, -1, :]
        h = self.reshape_for_output(h, B_size, L)
        return h, h_prev

class MultiCenterS4DStateEncoder(nn.Module):
    """
    S4D Encoder returning the direct complex state grid h_t.
    
    Returns State h shape: [Batch, Seq_Len, state_dim] (Complex)
    
    Args:
        state_dim (int): Number of HiPPO polynomial modes N.
        l_max (int): Maximum context horizon.
        sigma_ratio (float): Width of Gaussian centers.
    """
    def __init__(
        self, 
        state_dim: int, 
        l_max: int = 100_000,
        # sigma_ratio: float = 1
        sigma: float = 200
    ):
        super().__init__()
        self.state_dim = state_dim        # N (HiPPO modes)
        self.l_max = l_max

        # 1. Distribute K centers (tau_k) evenly across [0, L_max]
        # factor of 2 because contribution gating averages<=0.5
        center_spacing = l_max / (2*(state_dim + 1))
        tau = generate_halton_centers(state_dim, l_max)
        self.log_alpha = nn.Parameter(torch.log(torch.ones(state_dim) * 0.5))
        # sigma = center_spacing * sigma_ratio

        self.register_buffer("tau", tau)      # [K, 1]
        # self.register_buffer("sigma", torch.tensor(sigma))
        self.sigma = nn.Parameter(torch.ones(state_dim) * sigma)
        self.scale = nn.Parameter(torch.ones(state_dim) * 1e-3)

        # 2. Pure HiPPO Operators (A and B)
        n = torch.arange(state_dim, dtype=torch.float32)
        
        omega = math.pi * n                                 # Frequencies [N]
        self.delta = nn.Parameter(torch.log(torch.ones(state_dim) * 1e-3))                                      # Discretization step
        
        # A_vector: Pure phase rotation e^(i * omega * delta)
        # A_vector = torch.complex(torch.cos(omega * self.delta), torch.sin(omega * self.delta)) # [N] (Complex)

        # self.register_buffer("A_vector", A_vector)        # [N]
        # self.A_vector = nn.Parameter(A_vector)
        self.omega = nn.Parameter(omega)

    def reshape_for_linear_recurrence(self, x: torch.Tensor, B_size: int, inT: int) -> torch.Tensor:
        return x.reshape(B_size, inT, self.state_dim//2, 2).permute(0, 2, 1, 3).reshape(B_size * self.state_dim//2, inT, 2)

    def reshape_for_output(self, x: torch.Tensor, B_size: int, inT: int) -> torch.Tensor:
        return x.reshape(B_size,self.state_dim//2, inT, 2).permute(0, 2, 1, 3).reshape(B_size, inT, self.state_dim)

    def forward(self, x: torch.Tensor, contribution_gate: torch.Tensor, h_prev: torch.Tensor = None, prev_cumulative_contribution: torch.Tensor = None) -> torch.Tensor:
        """
        Input x:   [Batch, Seq_Len, d_model] (Real)
        Returns h: [Batch, Seq_Len, K_centers, N_modes, d_model] (Complex)
        """
        B_size, L, D = x.shape
        device = x.device
        if prev_cumulative_contribution is None or isinstance(prev_cumulative_contribution, int):
            prev_cumulative_contribution = 0
        else:
            prev_cumulative_contribution = prev_cumulative_contribution.unsqueeze(1)
        contribution_gate = contribution_gate

        cumulative_contribution = torch.cumsum(contribution_gate, dim=1) + prev_cumulative_contribution
        prev_cumulative_contribution = cumulative_contribution[:, -1, :]
        delta = torch.exp(soft_clamp(self.delta, min_val=-4.5, max_val=-1))
        alpha = torch.exp(- (cumulative_contribution - self.tau) ** 2 / (2 * self.sigma ** 2))
        complement = 1 - contribution_gate*alpha
        x_impulse = x * delta

        if h_prev is None:
            h_prev = torch.zeros(B_size, self.state_dim, dtype=torch.complex64, device=device)

        # A_sys = self.A_vector.view(1, 1, self.state_dim)  # [1, 1, N]
        a_alpha = torch.exp(soft_clamp(self.log_alpha, min_val=-25, max_val=25))
        A_sys = torch.exp(-(a_alpha + 1j * self.omega ) * delta).view(1, 1, self.state_dim)
        A_contr = A_sys * alpha * contribution_gate
        A_sys = 1*complement + A_contr
        x_impulse = x_impulse * alpha * contribution_gate
        x_impulse = self.reshape_for_linear_recurrence(x_impulse, B_size, L)
        A_sys = self.reshape_for_linear_recurrence(A_sys, B_size, L)
        h = linear_recurrence(x_impulse, A_sys, h_prev)
        h_prev = h[:, -1, :]
        h = self.reshape_for_output(h, B_size, L) * self.scale
        return h, h_prev, prev_cumulative_contribution


class HippoMultiCenterS4DStateEncoderBlock(nn.Module):
    def __init__(self, d_model: int, state_dim: int, l_max: int, sigma_ratio: float = 2.0):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.l_max = l_max
        self.sigma_ratio = sigma_ratio
        self.proj_state = nn.Linear(d_model, 2*state_dim)
        self.multi_center_s4d_state_encoder = MultiCenterS4DStateEncoder(state_dim, l_max, sigma_ratio)
        self.contribution_gate = nn.Linear(d_model, state_dim)
        self.state_norm = nn.LayerNorm(2*state_dim)
        self.gated_proj = GatedState(d_model, 2*state_dim)
        
    def forward(self, x: torch.Tensor, state=None) -> torch.Tensor:
        B, T, D = x.shape
        if state is None:
            prev_state_trajectory = torch.zeros(B * self.state_dim//2, 2, dtype=torch.complex64, device=x.device)
            prev_cumulative_contribution = 0
        else:
            prev_state_trajectory = state[0]
            prev_cumulative_contribution = state[1]
        s_real, s_imag = torch.chunk(self.proj_state(x), 2, dim=-1)
        s = torch.complex(s_real, s_imag)
        contribution_gate = torch.sigmoid(self.contribution_gate(x))
        h, prev_state_trajectory, cumulative_contribution = self.multi_center_s4d_state_encoder(s, contribution_gate, prev_state_trajectory, prev_cumulative_contribution)
        h = torch.cat([h.real, h.imag], dim=-1)
        h = self.state_norm(h)
        h = self.gated_proj(x, h)
        return h, [prev_state_trajectory, cumulative_contribution]

class FullMultiBlock(nn.Module):
    def __init__(self, d_model: int, state_dim: int, l_max: int, sigma_ratio: float = 2.0):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.multi_center_s4d_state_encoder = HippoMultiCenterS4DStateEncoderBlock(d_model, state_dim, l_max, sigma_ratio)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, state=None) -> torch.Tensor:
        x_norm = self.norm(x)
        global_out, state = self.multi_center_s4d_state_encoder(x_norm, state=state)
        x = x + global_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, state

class FullS4DBlock(nn.Module):
    def __init__(self, d_model: int, state_dim: int):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.proj_state = nn.Linear(d_model, 2 * state_dim)
        self.s4d = S4DEncoder_state(state_dim)
        self.proj_out = GatedState(d_model, 2 * state_dim)
        self.proj_contribution = nn.Linear(d_model, state_dim)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, state=None) -> torch.Tensor:
        x_norm = self.norm(x)
        contribution_factor = torch.sigmoid(self.proj_contribution(x_norm))
        s = torch.chunk(self.proj_state(x_norm), 2, dim=-1)
        s = torch.complex(s[0], s[1])
        s, state = self.s4d(s, contribution_factor, state)
        x = x + self.proj_out(x, torch.cat([s.real, s.imag], dim=-1))
        x = x + self.ffn(self.ffn_norm(x))
        return x, state

class S4DEncoderBlock(nn.Module):
    def __init__(self, d_model: int, state_dim: int, l_max: int, sigma_ratio: float = 2.0):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.sigma_ratio = sigma_ratio
        self.l_max = l_max
        self.full_s4d_block = FullS4DBlock(d_model, state_dim)
        self.full_multi_block = FullMultiBlock(d_model, state_dim, l_max, sigma_ratio)

    def forward(self, x: torch.Tensor, state=None) -> torch.Tensor:
        B, T, D = x.shape
        if state is None:
            s4d_state = None
            multi_center_state = None
        else:
            s4d_state = state[0]
            multi_center_state = state[1]
        x, s4d_state = self.full_s4d_block(x, s4d_state)
        x, multi_center_state = self.full_multi_block(x, multi_center_state)
        return x, [s4d_state, multi_center_state]

class LociS4DLm(nn.Module):
    def __init__(self, vocab_size: int, 
    embedding_dim: int,
     model_dim: int, 
     state_dim: int, 
     seq_len: int, 
     sigma_ratio: float = 2.0, 
     num_blocks: int = 4,
     loss_fn: nn.Module = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.model_dim = model_dim
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.num_blocks = num_blocks
        self.max_chunk_size = MODEL_CHUNK_SIZE
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.blocks = nn.ModuleList([
            S4DEncoderBlock(model_dim, state_dim, seq_len, sigma_ratio)
            for _ in range(num_blocks)
        ])
        self.embedding_proj = nn.Linear(embedding_dim, model_dim)
        self.pre_head = nn.Linear(model_dim, embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)
        self.head.weight = self.embedding.weight
        self.head.bias = nn.Parameter(torch.zeros(vocab_size))
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = lambda x, y: F.cross_entropy(x.reshape(-1, x.shape[-1]), y.reshape(-1))

    def _forward(self, x: torch.Tensor, states=None, y = None) -> torch.Tensor:
        x = self.embedding_proj(x)
        new_states = []
        for i, block in enumerate(self.blocks):
            state = states[i] if states is not None else None
            x, new_state = block(x, state)
            new_states.append(new_state)
        x = self.pre_head(x)
        x = self.head(x)
        if y is not None:
            loss = self.loss_fn(x, y)
            return [loss, x.detach()], new_states
        return x, new_states

    def _forward_chunked(self, x: torch.Tensor, states, y) -> torch.Tensor:
        B, T = x.shape
        if not y is None:
            loss = 0
            num_chunks = 0
        chunks = []
        for i in range(0, T, self.max_chunk_size):
            x_chunk = x[:, i:i+self.max_chunk_size]
            x_chunk = self.embedding(x_chunk)
            y_chunk = y[:, i:i+self.max_chunk_size] if y is not None else None
            chunk, states = checkpoint.checkpoint(self._forward, x_chunk, states, y_chunk, use_reentrant=False)
            if y is not None:
                num_chunks += 1
                loss += chunk[0]
                chunk = chunk[1].detach()
            chunks.append(chunk)
        if y is not None:
            num_chunks = max(num_chunks, 1)
            results = [loss / num_chunks, torch.cat(chunks, dim=1)]
            return results, states
        return torch.cat(chunks, dim=1), states

    def forward(self, x: torch.Tensor, states=None, y = None) -> torch.Tensor:
        if x.shape[1] > self.max_chunk_size:
            return self._forward_chunked(x, states, y)
        return self._forward(self.embedding(x), states, y)
        

    def generate(self, prompt: torch.Tensor, max_new_tokens: int, temperature: float = 0.75) -> torch.Tensor:
        prompt = prompt.unsqueeze(0)
        out = prompt
        states = None
        for _ in range(max_new_tokens):
            logits, states = self(prompt, states)
            next_token = torch.multinomial(F.softmax(logits/temperature, dim=-1).reshape(-1, logits.shape[-1]) , num_samples=1)[-1]
            prompt = next_token.unsqueeze(0)
            out = torch.cat([out, prompt], dim=-1)
        return out.squeeze(0)

def causal_leak_test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = LociS4DLm(vocab_size=64, embedding_dim=16, model_dim=16, state_dim=16, seq_len=512,
     sigma_ratio=2.0, num_blocks=1).to(device)
    prompt = torch.randint(0, 62, (1, 512), dtype=torch.long).to(device)
    logits1, _ = model(prompt)
    prompt[:, -1] = 63
    logits2, _ = model(prompt)
    print(torch.allclose(logits1[:, :-1, :], logits2[:, :-1, :]))
def gradient_causal_leak_test( seq_len=512, vocab_size=64):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LociS4DLm(vocab_size=64, embedding_dim=16, model_dim=16, state_dim=16, seq_len=512,
     sigma_ratio=2.0, num_blocks=1).to(device)
    model.eval()
    
    # Use continuous embeddings to compute gradients directly
    dummy_embeds = torch.randn(1, seq_len, model.embedding_dim, requires_grad=True)
    
    # Forward pass using embeddings (if model supports forward via inputs_embeds)
    logits, _ = model(inputs_embeds=dummy_embeds)
    
    # Pick a logit at an intermediate position, e.g., position 10
    target_pos = seq_len // 2
    target_logit = logits[0, target_pos, :].sum()
    target_logit.backward()
    
    # Check gradients w.r.t embeddings at future positions (j > target_pos)
    future_grads = dummy_embeds.grad[0, target_pos + 1:, :]
    
    leak_magnitude = torch.max(torch.abs(future_grads)).item()
    assert leak_magnitude == 0.0, f"Causal leak! Non-zero gradient found from future positions: {leak_magnitude}"
    
    print("Gradient causality test passed.")

def strict_causal_leak_test(seq_len=512, vocab_size=64):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LociS4DLm(vocab_size=64, embedding_dim=16, model_dim=16, state_dim=16, seq_len=512,
     sigma_ratio=2.0, num_blocks=1).to(device)
    model.to(device).eval()
    
    with torch.no_grad():
        base_prompt = torch.randint(0, vocab_size - 1, (1, seq_len), device=device)
        base_logits, _ = model(base_prompt)
        
        # Check several intermediate positions
        for pos in [1, seq_len // 2, seq_len - 1]:
            modified_prompt = base_prompt.clone()
            # Change the token at index `pos`
            modified_prompt[0, pos] = (modified_prompt[0, pos] + 1) % vocab_size
            
            modified_logits, _ = model(modified_prompt)
            
            # Logits strictly before `pos` MUST remain identical
            past_base = base_logits[:, :pos, :]
            past_mod = modified_logits[:, :pos, :]
            
            max_diff = torch.max(torch.abs(past_base - past_mod)).item()
            assert max_diff == 0.0 or torch.allclose(past_base, past_mod, atol=1e-6), \
                f"Causal leak detected! Modifying position {pos} altered logits at earlier positions. Max diff: {max_diff}"

    print("Causality test passed.")


if __name__ == "__main__":
    # causal_leak_test()
    torch.manual_seed(42)
    strict_causal_leak_test()
    # Vocabulary Simulation (e.g., character-level tokenization)
    seq_len = 64       # Sequence block chunk size
    batch_size = 24
    max_seq_len = 1024
    
    # Simulated Shakespearean data chunk (highly repetitive structures)
    # 0: ' ', 1: 't', 2: 'o', 3: 'b', 4: 'e', etc.
    path = "C:/Users/Sean/Desktop/DEV/PhysicsTransmitterReceiver/tinyshakespeare.txt"
    # ── Data ─────────────────────────────────────────────────────────────────────
    with open(path) as f:
        text = f.read()
    
    chars  = sorted(set(text))
    stoi   = {c: i for i, c in enumerate(chars)}
    itos   = {i: c for i, c in enumerate(chars)}
    # VOCAB  = len(cross_chars)
    VOCAB  = len(chars)
    encode = lambda s: [stoi[c] for c in s]
        # return [bigram[s[2*i:(2*i)+2]] for i in range(len(s)//2)]
    decode = lambda ids: ''.join(itos[i] for i in ids)

    data  = torch.tensor(encode(text), dtype=torch.long)
    n     = int(0.9 * len(data))
    train = data[:n]
    val   = data[n:]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    # model = ShakespeareSubtractiveLM(vocab_size, token_dim, state_dim, seq_len, hidden_dim)
    # model = ParallelShakespeareSubtractiveLM(VOCAB, token_dim, state_dim, max_seq_len, hidden_dim).to(device)
    EMBED_DIM = 32
    LATENT_DIM = 64
    CHARACTERISTIC_MAX_SEQ_LEN = 1024
    BATCH_SIZE = 4
    NUM_BLOCKS = 2
    # (self, input_dim: int, state_dim: int, num_blocks: int, embed_dim: int, vocab_size: int)
    # Initialize Model & Optimizer
    model = LociS4DLm(
        vocab_size=VOCAB,
        embedding_dim=EMBED_DIM,
        model_dim=LATENT_DIM,
        state_dim=LATENT_DIM,
        seq_len=CHARACTERISTIC_MAX_SEQ_LEN,
        sigma_ratio=2.0,
        num_blocks=NUM_BLOCKS,
    ).to(device)

    # Optimized Adam configuration for custom rational coordination
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.99, 0.999))
    
    print("Initializing Coordinated Dual-Objective Optimization Loop...\n")
    def get_batch(split):
        d  = train if split == 'train' else val
        ix = torch.randint(len(d) - current_seq_len - 1, (batch_size,)) + 1
        x  = torch.stack([d[i:i+current_seq_len]     for i in ix]).to(device)
        y  = torch.stack([d[i+1:i+current_seq_len+1] for i in ix]).to(device)
        x_prev = torch.stack([d[i-1:i+current_seq_len-1] for i in ix]).to(device)
        return x, y, x_prev

    target_recon_acc = 95.0
    min_predict_acc = 35.0  # A realistic baseline for high-entropy next-char/token prediction
    consecutive_stable_steps = 0
    step_increase = 8
    current_seq_len = seq_len

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    for epoch in range(200000):
        batch_size = int(2*max_seq_len/current_seq_len)
        model.train()
        optimizer.zero_grad()
        x, y, x_prev = get_batch('train')
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) %100 == 0:
            with torch.no_grad():
                x, y, x_prev = get_batch('val')    
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                prompt =torch.tensor(encode("ROMEO: ")).to(device)
                sample = model.generate(prompt, 1000)
                print(decode(sample.tolist()))
            print(f"Epoch {epoch+1:4d} | Loss: {loss.item():.4f}")
