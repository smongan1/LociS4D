import torch
import torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import numpy as np
import math
import time
import json
import os
from pathlib import Path

from BruteForceHippo import LociS4DLm
from loadLLMData import getLiveData, decode
# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SHAKESPEARE_PATH = os.path.join(CURRENT_DIR, "tinyshakespeare.txt")
CHECKPOINT_PATH_BEST  = "best_model.pth"
CHECKPOINT_PATH_LAST  = "last_model.pth"
HISTORY_PATH     = "history.json"
USE_PROFILER = False
BLOCK_SIZE       = 2048
BATCH_SIZE       = 1
BATCH_GRADIENT_ACCUMULATION_STEPS = 1
# LR               = 0.0007017723774753744
LR               = 1e-3
# WEIGHT_DECAY     = 0.001976712
WEIGHT_DECAY     = 1e-4
WARMUP_STEPS     = 1000
VALUE_CLIP       = 100
NORM_CLIP        = 1
EVAL_EVERY_SECS  = 120    # evaluate and (maybe) checkpoint every N seconds
EVAL_ITERS       = 50
HISTORY_DECAY    = 0.9
ADAMW_EPS        = 1e-6
ADAMW_BETAS      = (0.99, 0.999)

TRAIN_MINUTES    = 600     # total wall-clock budget
USE_CHARACTER_LEVEL = False
FOCAL_LOSS = False
if USE_PROFILER:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns


class CharacterFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, ignore_index=-100, reduction='mean'):
        super(CharacterFocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        if self.reduction == 'mean':
            # Create a mask to avoid averaging over ignored/padding indices
            if self.ignore_index != -100:
                valid_mask = (targets != self.ignore_index).float()
                return focal_loss.sum() / (valid_mask.sum() + 1e-8)
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def make_plot(profiler, output_filename="gradient_norms.png"):
    """Plots the gradient norms for each model parameter over time and saves it as a PNG."""
    # 1. Set a clean, modern aesthetic
    sns.set_theme(style="whitegrid")

    # 2. Convert profiler dict to DataFrame (columns = layers, rows = steps)
    df = pd.DataFrame(profiler)
    df.to_csv(f"{output_filename.rstrip('.png')}.csv")
    df.index.name = "Training Step"

    # 3. Create the figure
    plt.figure(figsize=(12, 6))

    # Seaborn automatically plots each column as a separate line when given wide-form data
    ax = sns.lineplot(data=df, linewidth=1.5, dashes=False)

    # 4. Style the labels and title
    plt.title("Gradient Norm Tracking per Layer", fontsize=14, fontweight="bold")
    plt.xlabel("Training Step / Iteration", fontsize=12)
    plt.ylabel("Gradient Norm", fontsize=12)

    # 5. Handle the legend gracefully
    # Deep learning layer names can get long, so we push the legend outside the plot area
    plt.legend(
        title="Model Parameters",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0,
    )

    # 6. Save the plot
    # bbox_inches='tight' is crucial here so the outside legend doesn't get cut off
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()  # Free up memory

    print(f"[Profiler] Plot successfully saved to '{output_filename}'")

# ─────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────

def load_data(path: str, device: str):
    print("Loading tiny Shakespeare...")
    with open(path, "r") as f:
        text = f.read()

    chars      = sorted(set(text))
    vocab_size = len(chars)
    stoi       = {c: i for i, c in enumerate(chars)}
    itos       = {i: c for c, i in stoi.items()}
    enc        = lambda s: torch.tensor([stoi[c] for c in s], dtype=torch.long).to(device)
    dec        = lambda t: "".join(itos[i] for i in (t.tolist() if isinstance(t, torch.Tensor) else t))

    data       = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n          = int(0.9 * len(data))
    train_data = data[:n]
    val_data   = data[n:]
    print(f"Vocab size: {vocab_size} | Train: {len(train_data):,} | Val: {len(val_data):,}")
    return train_data, val_data, vocab_size, enc, dec


def get_batch_shakespeare(src, block_size: int, batch_size: int, device: str):
    ix   = torch.randint(len(src) - block_size - 1, (batch_size,))
    seqs = torch.stack([src[i : i + block_size] for i in ix]).to(device)
    x    = seqs[:, :-1]
    y    = seqs[:, 1:].clone()
    return x, y

def _ascii_encode_padded(text: str, length: int) -> list[int]:
    """Encode text as ASCII bytes (dropping non-ASCII chars), padded/truncated to `length`."""
    b = ''.join(text).encode("ascii", errors="ignore")
    if len(b) < length:
        b = b + bytes(length - len(b))  # zero-pad
    else:
        b = b[:length]
    return list(b)

def _tokens_to_ascii_tensor(input_ids, tokenizer, block_size: int, device: str):
    """Decode fineweb token ids back to text (fineweb decode), then re-encode as ASCII bytes."""
    length = block_size + 1
    rows = []
    for ids in input_ids:
        text = decode(tokenizer, ids)
        rows.append(_ascii_encode_padded(text, length))
    data = torch.tensor(rows, dtype=torch.long).to(device)
    x = data[:, :block_size]
    y = data[:, 1:block_size + 1]
    return x, y

def get_batch_fineweb(dataloader_iterable, block_size: int, batch_size: int, device: str, tokenizer=None, use_character_level: bool = USE_CHARACTER_LEVEL):
    batch = next(dataloader_iterable)
    if use_character_level:
        x, y = _tokens_to_ascii_tensor(batch["input_ids"], tokenizer, block_size, device)
    else:
        x = batch["input_ids"][:, :-1].to(device)
        y = batch["input_ids"][:, 1:].to(device)
    return x, y

def get_batch(train_data, block_size: int, batch_size: int, device: str, tokenizer=None):
    if isinstance(train_data, torch.Tensor):
        return get_batch_shakespeare(train_data, block_size, batch_size, device)
    return get_batch_fineweb(train_data, block_size, batch_size, device, tokenizer=tokenizer)
# ─────────────────────────────────────────────
# Gradient clipping
# ─────────────────────────────────────────────

def clip_gradients(model, value_clip: float, norm_clip: float):
    # torch.nn.utils.clip_grad_value_(model.parameters(), value_clip)
    return torch.nn.utils.clip_grad_norm_(model.parameters(), norm_clip)


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

@torch.no_grad()
def estimate_loss_shakespeare(model, train_data, val_data, block_size, batch_size, device, eval_iters, loss_function):
    results = {}
    for split, src in (("train", train_data), ("val", val_data)):
        losses, accs = [], []
        for _ in range(eval_iters):
            xb, yb = get_batch_shakespeare(src, block_size, batch_size, device)
            logits = model(xb)
            if isinstance(logits, tuple):
                logits = logits[0]
            losses.append(loss_function(logits, yb).item())
            accs.append((logits.argmax(-1) == yb).float().mean().item())
        results[split] = (float(np.mean(losses)), float(np.mean(accs)))
    return results

@torch.no_grad()
def estimate_loss_fineweb(model, train_data, val_data, block_size, batch_size, device, eval_iters, loss_function, tokenizer=None, use_character_level: bool = USE_CHARACTER_LEVEL):
    results = {}
    for split, src in (("train", train_data), ("val", val_data)):
        losses, accs = [], []
        if use_character_level:
            xb, yb = _tokens_to_ascii_tensor(src["input_ids"], tokenizer, block_size, device)
        else:
            xb, yb = src["input_ids"][:, :-1].to(device), src["input_ids"][:, 1:].to(device)
        logits = model(xb)
        if isinstance(logits, tuple):
            logits = logits[0]
        losses.append(loss_function(logits, yb).item())
        accs.append((logits.argmax(-1) == yb).float().mean().item())
        results[split] = (float(np.mean(losses)), float(np.mean(accs)))
    return results
@torch.no_grad()
def run_eval(model, train_data, val_data, device, step, norm,
             history, best_val_loss, checkpoint_path_best, checkpoint_path_last, enc, dec, sample_text, loss_function, tokenizer=None, use_character_level: bool = USE_CHARACTER_LEVEL):
    model.eval()
    if isinstance(train_data, torch.Tensor):
        stats = estimate_loss_shakespeare(model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, device, EVAL_ITERS, loss_function)
    else:
        stats = estimate_loss_fineweb(model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, device, EVAL_ITERS, loss_function, tokenizer=tokenizer, use_character_level=use_character_level)
    tl, ta = stats["train"]
    vl, va = stats["val"]
    try:
        ppl    = math.exp(vl)
    except:
        ppl = float("inf")

    print(f"{step:>7}  {tl:>10.4f}  {vl:>10.4f}  {ta:>9.3f}  {va:>9.3f}  {ppl:>8.2f}  {float(norm):>8.4f}")

    history["step"].append(step)
    history["train_loss"].append(tl)
    history["val_loss"].append(vl)
    history["train_acc"].append(ta)
    history["val_acc"].append(va)
    if use_character_level:
        sample_length = 1000
    else:
        sample_length = 100
    sample_out = model.generate(enc(sample_text), max_new_tokens=sample_length, temperature=0.75)
    print("Sample:", dec(sample_out), "\n")

    if vl < best_val_loss[0]:
        best_val_loss[0] = vl
        torch.save(model.state_dict(), checkpoint_path_best)
        print(f"  ✓ New best val loss {vl:.4f} — saved to {checkpoint_path_best}")
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f"  ✓ New last val loss {vl:.4f} — saved to {checkpoint_path_last}")

    return best_val_loss[0]

def train_loop(model, optimizer, scheduler, train_data, val_data, device, eval_every_secs, 
train_minutes, checkpoint_path_best, checkpoint_path_last, enc, dec, eval_train_data, eval_val_data, loss_function, sample_text, pad_id=None, tokenizer=None):
    # ── Loop ───────────────────────────────────────────────────────────────
        # ── State ──────────────────────────────────────────────────────────────
    best_val_loss           = [float("inf")]
    history                 = {"step": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    deadline                = time.time() + train_minutes * 60
    last_eval_time          = time.time()
    step                    = 0
    t0                      = time.time()
    if USE_PROFILER:
        profiler = {name:[] for name, _ in model.named_parameters()}
    norm   = 0.0
    optimizer.zero_grad()
    while time.time() < deadline:
        model.train()
        step += 1
        xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, device, tokenizer=tokenizer)
        if step < WARMUP_STEPS:
            warmup_lr = LR * step / WARMUP_STEPS
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        results, _ = model(xb, y = yb)
        logits = results[1]
        loss = results[0]/BATCH_GRADIENT_ACCUMULATION_STEPS
        # loss = loss_function(logits, yb)/BATCH_GRADIENT_ACCUMULATION_STEPS

        loss.backward()
        if USE_PROFILER:
            for name, param in model.named_parameters():
                try:
                    profiler[name].append(param.grad.norm().item())
                except:
                    profiler[name].append(0)
        if step % BATCH_GRADIENT_ACCUMULATION_STEPS == 0:
            norm = clip_gradients(model, VALUE_CLIP, NORM_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    

        # ── Periodic evaluation ────────────────────────────────────────────
        now = time.time()
        if now - last_eval_time >= eval_every_secs:
            run_eval(model, eval_train_data, eval_val_data, device, step, norm,
                     history, best_val_loss, checkpoint_path_best, checkpoint_path_last, enc, dec, sample_text, loss_function, tokenizer=tokenizer)
            last_eval_time = now
            if USE_PROFILER:
                make_plot(profiler)

    
    elapsed = time.time() - t0
    print(f"\nTraining finished: {elapsed/60:.1f} min  |  {step:,} steps  ({step/elapsed:.1f} steps/sec)")

    # ── Save history ────────────────────────────────────────────────────────
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {HISTORY_PATH}")
    print(f"Running final evaluation...")
    run_eval(model, eval_train_data, eval_val_data, device, step, norm,
                     history, best_val_loss, checkpoint_path_best, checkpoint_path_last, enc, dec, sample_text, loss_function, tokenizer=tokenizer)
    return model, optimizer, scheduler, history, best_val_loss

def train_shakespeare(model_constructor, device, train_minutes, checkpoint_path_best, checkpoint_path_last, eval_every_secs):
    train_data, val_data, vocab_size, enc, dec = load_data(SHAKESPEARE_PATH, device)
    model = model_constructor(vocab_size, device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}\n")
    
    print(f"{'Step':>7}  {'Train Loss':>10}  {'Val Loss':>10}  {'Train Acc':>9}  {'Val Acc':>9}  {'PPL':>8}  {'Norm':>8}")
    print("─" * 80)
    # ── Optimiser / scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, eps=ADAMW_EPS, betas=ADAMW_BETAS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2000, eta_min=LR / 10)
    if FOCAL_LOSS:
        focal_loss = CharacterFocalLoss(gamma=2.0, ignore_index=-100)
    
    def loss_fn(logits, yb):
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits.reshape(-1, logits.shape[-1])
        yb = yb.reshape(-1)
        if FOCAL_LOSS:
            return focal_loss(logits, yb)
        return F.cross_entropy(logits, yb)
    model.loss_fn = loss_fn
    sample_text = "ROMEO:"
    model, optimizer, scheduler, history, best_val_loss = train_loop(model, optimizer, scheduler, train_data, 
    val_data, device, eval_every_secs, train_minutes, checkpoint_path_best, checkpoint_path_last, enc, dec, train_data, val_data, loss_fn, sample_text, pad_id=None)
    return model, best_val_loss

def train_fineweb(model_constructor, device, train_minutes, checkpoint_path_best, checkpoint_path_last, eval_every_secs, use_character_level: bool = USE_CHARACTER_LEVEL):
    EVAL_ITERS = 2
    training_dataloader, validation_dataloader, tokenizer = getLiveData(BATCH_SIZE, context_length=BLOCK_SIZE)
    PAD_ID = tokenizer.pad_token_id
    if use_character_level:
        PAD_ID = 0
        VOCAB = 256 # 256 characters in the ASCII table
    else:
        VOCAB = tokenizer.vocab_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    # model = dumbGram(vocab_size=VOCAB).to(device)
    model = model_constructor(VOCAB, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}\n")
    
    # For character level, sequences are zero-padded (see _ascii_encode_padded), so ignore index 0;
    # otherwise ignore the tokenizer's real pad token id.
    char_ignore_index = PAD_ID
    if FOCAL_LOSS:
        focal_loss = CharacterFocalLoss(gamma=2.0, ignore_index=char_ignore_index)
    
    def loss_fn(logits, yb):
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits.reshape(-1, logits.shape[-1])
        yb = yb.reshape(-1)
        if FOCAL_LOSS:
            return focal_loss(logits, yb)
        return F.cross_entropy(logits, yb, ignore_index=char_ignore_index)
    model.loss_fn = loss_fn
    print(f"{'Step':>7}  {'Train Loss':>10}  {'Val Loss':>10}  {'Train Acc':>9}  {'Val Acc':>9}  {'PPL':>8}  {'Norm':>8}")
    print("─" * 80)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2000, eta_min=LR / 10)
    if use_character_level:
        def enc(text):
            return torch.tensor(list(text.encode("ascii", errors="ignore")), dtype=torch.long).to(device)
        def dec(tokens):
            vals = tokens.tolist() if isinstance(tokens, torch.Tensor) else tokens
            return bytes(vals).decode("ascii", errors="ignore")
    else:
        def enc(text):
            return torch.tensor(tokenizer.encode(text), dtype=torch.long).to(device)
        def dec(tokens):
            return ''.join(decode(tokenizer, tokens))
    val_data = iter(validation_dataloader)
    train_data = iter(training_dataloader)
    eval_val_data = next(val_data)
    eval_train_data = next(train_data)
    sample_text = "To understand how a cell generates energy, we must first look at..."
    model, optimizer, scheduler, history, best_val_loss = train_loop(model, optimizer, scheduler, train_data, 
            val_data, device, eval_every_secs, train_minutes, checkpoint_path_best, checkpoint_path_last, enc, dec, eval_train_data, eval_val_data, loss_fn,
            sample_text, pad_id=PAD_ID, tokenizer=tokenizer)
    return model, best_val_loss
# ─────────────────────────────────────────────
# Core training function
# ─────────────────────────────────────────────

def train(
    model_constructor,
    train_minutes: float   = TRAIN_MINUTES,
    checkpoint_path_best: str   = CHECKPOINT_PATH_BEST,
    checkpoint_path_last: str   = CHECKPOINT_PATH_LAST,
    eval_every_secs: float = EVAL_EVERY_SECS,
    device: str | None     = None,
    use_shakespeare: bool = True
) -> LociS4DLm:
    """
    Train NeuronML on tiny Shakespeare for `train_minutes` wall-clock minutes.
    Saves the best checkpoint to `checkpoint_path` and returns the best model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if use_shakespeare:
        tran_fn = train_shakespeare
    else:
        tran_fn = train_fineweb
    model, best_val_loss = tran_fn(model_constructor, device, train_minutes, checkpoint_path_best, checkpoint_path_last, eval_every_secs)
    # ── Load and return best model ──────────────────────────────────────────
    if Path(checkpoint_path_best).exists():
        model.load_state_dict(torch.load(checkpoint_path_best, map_location=device))
        print(f"Best model loaded from {checkpoint_path_best}  (val loss: {best_val_loss[0]:.4f})")
    else:
        print("No checkpoint found — returning model from final step.")
    return model


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    best_model = train(
        train_minutes=TRAIN_MINUTES,
        checkpoint_path_best=CHECKPOINT_PATH_BEST,
        checkpoint_path_last=CHECKPOINT_PATH_LAST,
        eval_every_secs=EVAL_EVERY_SECS,
    )
    print(f"\nReturned model type: {type(best_model).__name__}")
    return best_model


if __name__ == "__main__":
    main()