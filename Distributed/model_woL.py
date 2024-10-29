import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from litgpt.lora import GPT, merge_lora_weights
from litgpt.data import SFTDataset
import litgpt
from datasets import load_dataset
import os

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

class LLMModel(nn.Module):
    def __init__(self, low_rank=4, rate=0.002, l_alpha=16, l_dropout=0.05):
        super(LLMModel, self).__init__()
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

    def load_weights(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cuda")
        self.model.load_state_dict(state_dict, strict=False)


def compute_loss(model, input_ids, targets):
    logits = model(input_ids)
    loss = litgpt.utils.chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:])
    return loss


def train(model, dataloader, optimizer, device, max_steps=-1):
    model.train()
    total_loss = 0
    steps = 0
    for batch in dataloader:
        steps += 1
        input_ids, targets = batch["input_ids"].to(device), batch["labels"].to(device)
        optimizer.zero_grad()
        loss = compute_loss(model, input_ids, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if steps == max_steps:
            return total_loss /steps, False
    return total_loss / len(dataloader), True


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, targets = batch["input_ids"].to(device), batch["labels"].to(device)
            loss = compute_loss(model, input_ids, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)


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
            prompt_style="alpaca"
        )
        self.val_dataset = SFTDataset(
            data=val_data,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            prompt_style="alpaca"
        )

    def get_dataloaders(self, world_size, rank):
        train_sampler = DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(self.val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=train_sampler, num_workers=self.num_workers)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=val_sampler, num_workers=self.num_workers)
        return train_loader, val_loader


def setup_distributed(rank, world_size):
    """ Initialize distributed environment """
    os.environ['MASTER_ADDR'] = 'localhost'  # Address of the master process
    os.environ['MASTER_PORT'] = '12355'      # Free port for communication
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    return rank


def cleanup():
    """Clean up distributed process group"""
    dist.destroy_process_group()


def dist_eval(rank, world_size, hyperparameters):
    grad_batches = hyperparameters.get("grad_batches", 16)
    rate = hyperparameters.get("learning_rate", 0.002)
    low_rank = hyperparameters.get("lora_rank", 4)
    fast_run = hyperparameters.get("fast_run", True)
    max_steps = 20 if fast_run else 2000
    n_epochs = hyperparameters.get("n_epochs", 1)

    # Setup distributed
    local_rank = setup_distributed(rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    # Setup data
    data_module = LLMData(repo_id="mhenrichsen/alpaca_2k_test", val_split_fraction=0.2)
    data_module.prepare_data()
    train_loader, val_loader = data_module.get_dataloaders(world_size, local_rank)

    # Initialize model and optimizer
    model = LLMModel(low_rank=low_rank, rate=rate).to(device)
    model = DDP(model, device_ids=[local_rank])

    # Load pre-trained weights
    checkpoint_path = f"checkpoints/{model_id}/lit_model.pth"
    model.module.load_weights(checkpoint_path)
    print(f"Rank {local_rank}, Model weights loaded from {checkpoint_path}")

    optimizer = optim.AdamW(model.parameters(), lr=rate, weight_decay=1e-2, betas=(0.9, 0.95))

    # Training loop
    bool_steps = True
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}")
        if bool_steps:
            train_loss, bool_steps = train(model, train_loader, optimizer, device, max_steps)
            print(f"Rank {local_rank}, Train Loss: {train_loss}")

        # Validation after each epoch
        val_loss = validate(model, val_loader, device)
        print(f"Rank {local_rank}, Validation Loss: {val_loss}")

    # Merge LoRA weights (after DDP cleanup)
    if local_rank == 0:
        merge_lora_weights(model.module.model)

    cleanup()
    return val_loss


def main_worker(rank, world_size, hyperparameters):
    return dist_eval(rank, world_size, hyperparameters)


def main():
    world_size = torch.cuda.device_count()  # Number of available GPUs

    # Hyper Parameters
    HP = {
        "learning_rate": 0.002,
        "lora_rank": 4,
    }

    # Start a process for each GPU
    a = mp.spawn(main_worker, args=(world_size, HP), nprocs=world_size, join=True)
    print(a)

if __name__ == "__main__":
    main()
    print("test")
