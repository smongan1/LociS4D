import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, IterableDataset

class RemoteOrLocalStreamingDataset_Modified(IterableDataset):
    def __init__(self, streamed_slice, tokenizer_name, max_length=512):
        super().__init__()
        # Accept the pre-sliced stream (.take() or .skip() generator)
        self.dataset = streamed_slice
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.max_length = max_length

    def __iter__(self):
        for example in self.dataset:
            text = example["text"]
            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None
            )
            yield {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"]
            }
    def get_validation_dataset(self):
        for example in self.validation_dataset:
            text = example["text"]
            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None
            )
            yield {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"]
            }

def dynamic_collate_fn(batch, pad_token_id):
    """
    Takes a list of on-the-fly tokenized items and pads them dynamically 
    to the longest sequence length *in this specific batch*.
    """
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_masks = [torch.tensor(item["attention_mask"]) for item in batch]
    
    # Pad sequences dynamically
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    padded_attention_masks = torch.nn.utils.rnn.pad_sequence(
        attention_masks, batch_first=True, padding_value=0
    )
    
    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_masks
    }

def getLiveData(BATCH_SIZE=4, context_length=64):
    # FineWeb-Edu sample configurations (using 'sample-10BT' for testing)
    DATASET_NAME = "HuggingFaceFW/fineweb-edu"
    SUBSET = "sample-10BT" 
    SPLIT = "train"
    TOKENIZER_NAME = "gpt2" # Replace with your target model tokenizer
    
    # 1. Load the base streaming dataset
    full_stream = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

    NUM_VAL_SAMPLES = 2000 

    # 2. Slice the stream dynamically (these return generators)
    val_stream = full_stream.take(NUM_VAL_SAMPLES)
    train_stream = full_stream.skip(NUM_VAL_SAMPLES)

    # 3. Create your IterableDataset instances using your custom class
    # (Pass the sliced streams into your wrapper class instead of loading fresh datasets inside)
    training_dataset = RemoteOrLocalStreamingDataset_Modified(train_stream, tokenizer_name=TOKENIZER_NAME, max_length=context_length)
    validation_dataset = RemoteOrLocalStreamingDataset_Modified(val_stream, tokenizer_name=TOKENIZER_NAME, max_length=context_length)
    # Get the pad token ID from our dataset's tokenizer for collation
    tokenizer = training_dataset.tokenizer
    pad_id = tokenizer.pad_token_id
    
    # Wrap with PyTorch DataLoader for seamless batching
    training_dataloader = DataLoader(
        training_dataset, 
        batch_size=BATCH_SIZE,
        collate_fn=lambda b: dynamic_collate_fn(b, pad_token_id=pad_id)
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=lambda b: dynamic_collate_fn(b, pad_token_id=pad_id)
    )
    
    return training_dataloader, validation_dataloader, tokenizer

def decode(tokenizer, ids):
    decoded_text = tokenizer.batch_decode(
        ids, 
        skip_special_tokens=True  # Removes <|endoftext|> or <pad> tokens
    )
    return decoded_text
# --- Execution Example ---
if __name__ == "__main__":
    
    # print(f"Initializing streaming dataset for {DATASET_NAME}...")
    training_dataloader, validation_dataloader, tokenizer = getLiveData()
    print(tokenizer.vocab_size)
    
    
    print("\nFetching and tokenizing batches on-the-fly...\n")
    
    # Let's pull 3 batches to demonstrate
    for i, batch in enumerate(training_dataloader):
        print(len(batch["input_ids"]))
        if i >= 3:
            break
        print(f"--- Batch {i+1} ---")
        decoded_text = decode(tokenizer, batch["input_ids"])
