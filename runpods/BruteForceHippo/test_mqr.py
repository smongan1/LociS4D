import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from BruteForceHippo import LociiS4DLm
import torch.nn.functional as F
class MQARDataset(Dataset):
    """
    Generates synthetic Multi-Query Associative Recall (MQAR) data.
    Structure: [KV Pairs ...] -> [Filler/Distractors ...] -> [Queries ...]
    """
    def __init__(self, num_samples=10000, seq_len=128, num_kv_pairs=8, vocab_size=512):
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_kv_pairs = num_kv_pairs
        self.vocab_size = vocab_size

        # Reserve special token IDs
        self.pad_token = 0
        self.kv_vocab_start = 1
        self.kv_vocab_end = vocab_size - 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Sample unique Keys and Random Values
        # Ensure keys and values don't overlap for clean evaluation
        keys = torch.randperm(self.vocab_size // 2)[1:self.num_kv_pairs + 1]
        values = torch.randperm(self.vocab_size // 2)[self.num_kv_pairs + 1 : 2 * self.num_kv_pairs + 1]

        # 2. Construct KV Pairs [K1, V1, K2, V2, ...]
        kv_prefix = []
        for k, v in zip(keys, values):
            kv_prefix.extend([k.item(), v.item()])
        
        # 3. Create Queries (permuted order of keys)
        query_keys = keys[torch.randperm(self.num_kv_pairs)]
        
        # 4. Fill middle sequence with random non-key noise/fillers
        num_fillers = self.seq_len - len(kv_prefix) - len(query_keys)
        fillers = torch.randint(low=self.vocab_size // 2 + 2, high=self.vocab_size, size=(num_fillers,)).tolist()

        # Combine sequence
        input_seq = kv_prefix + fillers + query_keys.tolist()
        
        # 5. Construct Ground Truth Labels
        # We only calculate loss/accuracy on the target tokens immediately following the queries
        labels = [self.pad_token] * len(input_seq)
        
        # Create map of Key -> Value
        kv_map = {k.item(): v.item() for k, v in zip(keys, values)}
        
        # Mark query targets
        query_start_idx = len(kv_prefix) + num_fillers
        for i, q in enumerate(query_keys):
            target_pos = query_start_idx + i
            # The target value should be predicted right at or after the query token
            labels[target_pos] = kv_map[q.item()]

        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

# Example Usage & Evaluation Harness
def evaluate_mqar_model(model, seq_len=256, num_kv_pairs=16, vocab_size=1024, batch_size=32):
    dataset = MQARDataset(num_samples=1000, seq_len=seq_len, num_kv_pairs=num_kv_pairs, vocab_size=vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model.to(device)
    model.eval()

    correct_retrievals = 0
    total_queries = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass: shape [Batch, Seq_Len, Vocab_Size]
            logits, _ = model(inputs) 
            predictions = torch.argmax(logits, dim=-1)

            # Mask out padding/non-query tokens (where target != 0)
            query_mask = targets != 0
            
            correct_retrievals += (predictions[query_mask] == targets[query_mask]).sum().item()
            total_queries += query_mask.sum().item()

    accuracy = (correct_retrievals / total_queries) * 100
    print(f"MQAR Retrieval Accuracy (Seq Len: {seq_len}, Pairs: {num_kv_pairs}): {accuracy:.2f}%")
    return accuracy

def train_mqar_model(model, seq_len=256, num_kv_pairs=16, vocab_size=1024, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.loss_fn = lambda x, y: F.cross_entropy(x.reshape(-1, x.shape[-1]), y.reshape(-1))
    dataset = MQARDataset(num_samples=100, seq_len=seq_len, num_kv_pairs=num_kv_pairs, vocab_size=vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    current_seq_len = seq_len
    acc_steps = 0
    min_acc_steps = 100 
    acc_threshold = 0.98
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    model.train()
    for epoch in range(10000000):  # Boosted slightly to ensure convergence
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            results, _ = model(inputs, y=targets)
            loss = results[0]
            logits = results[1]
            predictions = torch.argmax(logits, dim=-1)
            masked_predictions = predictions[targets != 0]
            masked_targets = targets[targets != 0]
            accuracy = (masked_predictions == masked_targets).sum().item() / len(masked_predictions)
            if accuracy >= acc_threshold:
                acc_steps += 1
            else:
                acc_steps = 0
            if acc_steps >= min_acc_steps:
                if num_kv_pairs < 3:
                    num_kv_pairs += 1
                    print(f"Increasing number of KV pairs to {num_kv_pairs}")
                else:
                    seq_scale = max(1, current_seq_len * .25)
                    current_seq_len =int(current_seq_len + seq_scale)
                    print(f"Increasing sequence length to {current_seq_len}")
                dataset = MQARDataset(num_samples=25, seq_len=current_seq_len, num_kv_pairs=num_kv_pairs, vocab_size=vocab_size)
                print(f"New dataset with sequence length {current_seq_len} and {num_kv_pairs} KV pairs")
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        if epoch % 25 == 0:
            print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, KV Pairs: {num_kv_pairs}, Sequence Length: {current_seq_len}")
        
    return model
if __name__ == "__main__":
    seq_len = 16
    num_kv_pairs = 1
    vocab_size = 1024
    batch_size = 32
    model = LociiS4DLm(vocab_size=vocab_size, embedding_dim=64, model_dim=128, state_dim=128, num_blocks=8, seq_len=5000)
    model = train_mqar_model(model, seq_len=seq_len, num_kv_pairs=num_kv_pairs, vocab_size=vocab_size, batch_size=batch_size)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    evaluate_mqar_model(model, seq_len=seq_len, num_kv_pairs=num_kv_pairs, vocab_size=vocab_size, batch_size=batch_size)