import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from litgpt.lora import GPT, merge_lora_weights
from litgpt.data import Alpaca2k, SFTDataset
import litgpt
from datasets import load_dataset

# Model definition (equivalent to LitLLM)
class LLMModel(nn.Module):
    def __init__(self, low_rank=4, rate=0.002, l_alpha=16, l_dropout=0.05):
        super(LLMModel, self).__init__()
        # Lora Model
        self.model = GPT.from_name(
            name="tiny-llama-1.1b",
            lora_r=low_rank,
            lora_alpha=l_alpha,
            lora_dropout=l_dropout,
            lora_query=True,
            lora_key=False,
            lora_value=True
        )
        litgpt.lora.mark_only_lora_as_trainable(self.model)

    def forward(self, input_ids):
        return self.model(input_ids)

# Function to compute loss
def compute_loss(model, input_ids, targets):
    logits = model(input_ids)
    loss = litgpt.utils.chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:])
    return loss

# Training function
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids, targets = batch["input_ids"].to(device), batch["labels"].to(device)
        optimizer.zero_grad()
        loss = compute_loss(model, input_ids, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Validation function
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, targets = batch["input_ids"].to(device), batch["labels"].to(device)
            loss = compute_loss(model, input_ids, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Data setup (similar to LLMDataModule in original code)
class LLMData:
    def __init__(self, repo_id, val_split_fraction=0.05, batch_size=1, max_seq_length=512, num_workers=4):
        self.repo_id = repo_id
        self.val_split_fraction = val_split_fraction
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_workers = num_workers
        self.tokenizer = litgpt.Tokenizer(f"checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    def prepare_data(self):
        dataset = load_dataset(self.repo_id)
        train_val_split = dataset["train"].train_test_split(test_size=self.val_split_fraction)
        train_data = train_val_split["train"]
        val_data = train_val_split["test"]

        self.train_dataset = SFTDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            prompt_style = "alpaca"
        )
        self.val_dataset = SFTDataset(
            data=val_data,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            prompt_style = "alpaca"
        )

    def get_dataloaders(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return train_loader, val_loader

# Distributed Evaluation
def dist_eval(hyperparameters):
    grad_batches = hyperparameters.get("grad_batches", 16)
    rate = hyperparameters.get("learning_rate", 0.002)
    low_rank = hyperparameters.get("lora_rank", 4)
    fast_run = hyperparameters.get("fast_run", True)
    max_steps = 20 if fast_run else 2000

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup data
    data_module = LLMData(repo_id="mhenrichsen/alpaca_2k_test", val_split_fraction=0.2)
    data_module.prepare_data()
    train_loader, val_loader = data_module.get_dataloaders()

    # Initialize model and optimizer
    model = LLMModel(low_rank=low_rank, rate=rate).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=rate, weight_decay=1e-2, betas=(0.9, 0.95))

    # Training loop
    for epoch in range(2):
        print(f"Epoch {epoch + 1}")
        train_loss = train(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss}")

        # Validation after each epoch
        val_loss = validate(model, val_loader, device)
        print(f"Validation Loss: {val_loss}")

    # Merge LoRA weights and return validation loss
    merge_lora_weights(model.model)
    return val_loss

if __name__ == "__main__":
    # Hyper Parameters
    HP = {
        "learning_rate": 0.002,
        "lora_rank": 4,
    }
    validation_loss = dist_eval(HP)
    print(f"Final Validation Loss: {validation_loss}")
