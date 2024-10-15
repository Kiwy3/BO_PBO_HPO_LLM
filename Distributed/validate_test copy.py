import os
import json
import torch  # type: ignore
import litgpt  # type: ignore
from litgpt.lora import GPT, merge_lora_weights  # type: ignore
from litgpt.data import Alpaca2k  # type: ignore
import lightning as L  # type: ignore
import torch.distributed as dist  # type: ignore

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

# Model ID for loading the pre-trained model from checkpoints
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def cleanup_distributed_environment():
    """
    Clean up the distributed environment.
    """
    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
        print("dist env destroyed")
    else : 
        print("dist env not init")
    
    if dist.is_initialized():
        print("still init")

# Define a LightningModule for fine-tuning a GPT model with LoRA (Low-Rank Adaptation)
class LitLLM(L.LightningModule):
    def __init__(self, low_rank=4, rate=0.002, l_alpha=16, l_dropout=0.05):
        """
        Initialize the LitLLM model with specified LoRA parameters.
        Args:
        - low_rank (int): LoRA rank, controls the low-rank decomposition.
        - rate (float): Learning rate.
        - l_alpha (int): LoRA scaling factor.
        - l_dropout (float): Dropout rate for LoRA layers.
        """
        super().__init__()
        self.lr = rate
        # Initialize GPT model with LoRA parameters
        self.model = GPT.from_name(
            name="tiny-llama-1.1b",
            lora_r=low_rank,
            lora_alpha=l_alpha,
            lora_dropout=l_dropout,
            lora_query=True,  # Apply LoRA to query weights
            lora_key=False,   # Do not apply LoRA to key weights
            lora_value=True   # Apply LoRA to value weights
        )
        # Mark only LoRA layers as trainable
        litgpt.lora.mark_only_lora_as_trainable(self.model)
        # Initialize an empty list to store validation step outputs
        self.validation_step_outputs = []

    def on_train_start(self):
        """
        Load pre-trained model weights at the start of training.
        """
        # Load the model state dictionary from a checkpoint
        state_dict = torch.load(f"checkpoints/{model_id}/lit_model.pth", mmap=True, weights_only=False)
        # Load the state into the model (strict=False allows some layers to be skipped)
        self.model.load_state_dict(state_dict, strict=False)

    def compute_loss(self, input_ids, targets):
        """
        Compute the cross-entropy loss for the model.
        Args:
        - input_ids: Input token IDs for the model.
        - targets: Target token IDs for the loss calculation.
        """
        # Get model logits (predictions)
        logits = self.model(input_ids)
        # Compute the chunked cross-entropy loss (ignoring the last token)
        loss = litgpt.utils.chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:])
        return loss

    def training_step(self, batch):
        """
        Define the training loop's step. This method is called on each batch.
        Args:
        - batch: The input data batch, containing 'input_ids' and 'labels'.
        """
        input_ids, targets = batch["input_ids"], batch["labels"]
        # Calculate the loss for the current batch
        loss = self.compute_loss(input_ids, targets)
        # Log the training loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        """
        Define the validation step, which is similar to the training step but for validation.
        Args:
        - batch: The validation data batch.
        """
        input_ids, targets = batch["input_ids"], batch["labels"]
        # Calculate the validation loss
        loss = self.compute_loss(input_ids, targets)
        # Append the loss to the validation outputs list
        self.validation_step_outputs.append(loss)
        # Log the validation loss
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        """
        This method is called automatically at the end of each validation epoch. It calculates the
        average validation loss across all processes and logs it.
        """
        # Stack the losses and compute the mean
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        # Perform an all-reduce operation to sum the losses across all devices
        dist.all_reduce(epoch_average, op=dist.ReduceOp.SUM)
        epoch_average /= dist.get_world_size()  # Average the loss across the number of devices
        # Log the averaged validation loss
        self.log("avg_val_loss", epoch_average)
        # Clear the list to free up memory
        self.validation_step_outputs.clear()

        # Commented out to remove the effect of saving the model checkpoint
        # if self.trainer.is_global_zero:
        #     self.trainer.save_checkpoint("checkpoints/best_model.ckpt", weights_only=True)

    def configure_optimizers(self):
        """
        Set up the optimizer and learning rate scheduler.
        """
        # Define AdamW optimizer with weight decay
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.0, betas=(0.9, 0.95))
        # Define a learning rate scheduler that increases linearly for the first 10 steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / 10)
        return [optimizer], [scheduler]

# Function to handle the training process
def train(trainer, data, HP, suffix=""):
    """
    Train the model using the given hyperparameters (HP) and save the checkpoint.
    Args:
    - trainer: PyTorch Lightning Trainer instance.
    - data: Dataset for training and validation.
    - HP: Dictionary containing hyperparameters.
    - suffix: Suffix for the saved checkpoint filename.
    """
    # Get hyperparameters for learning rate and LoRA rank, with default values
    rate = HP.get("learning_rate", 0.002)
    low_rank = HP.get("lora_rank", 4)

    # Initialize the model and start training
    with trainer.init_module(empty_init=True):
        model = LitLLM(low_rank=low_rank, rate=rate)
    # Fit the model using the provided trainer and dataset
    trainer.fit(model, data)
    # Merge the LoRA weights into the main model for saving
    merge_lora_weights(model.model)
    # Save the trained model checkpoint
    trainer.save_checkpoint(f"checkpoints/finetuned{suffix}.ckpt", weights_only=True)

# Function to handle validation
def validate(trainer, data, suffix=""):
    """
    Validate the model using a previously saved checkpoint.
    Args:
    - trainer: PyTorch Lightning Trainer instance.
    - data: Dataset for validation.
    - suffix: Suffix for the checkpoint file.
    """
    # Load the model from the specified checkpoint
    model = LitLLM.load_from_checkpoint(f"checkpoints/finetuned{suffix}.ckpt")

    # Run validation and collect outputs
    outputs = trainer.validate(model, dataloaders=data, verbose=True)

    # Gather results across all processes
    # Assuming 'outputs' is a list of dictionaries, we extract the validation loss
    val_losses = [output['val_loss'] for output in outputs]
    
    # Create a tensor to hold the results
    tensor_out = torch.tensor(val_losses).to("cuda")  # Move to GPU for reduction if applicable

    # Reduce the results to gather them on the main process
    dist.all_reduce(tensor_out, op=dist.ReduceOp.SUM)

    # Average the results (assuming we want the average across devices)
    avg_val_loss = tensor_out / dist.get_world_size()

    # Prepare the final output dictionary
    final_output = [{'val_loss': avg_val_loss.item(), 'avg_val_loss': avg_val_loss.item()}]

    return final_output  # Return the single device output

# Function to train and validate the model using given hyperparameters
def BB_eval(HP):
    """
    Evaluate the model with the given hyperparameters by training and validating.
    Args:
    - HP: Dictionary containing hyperparameters such as 'learning_rate' and 'device_number'.
    """
    # Check if "device_number" is specified in HP, else count available GPUs
    device_count = HP.get("device_number", torch.cuda.device_count())

    # Load the dataset and tokenizer
    data = Alpaca2k(val_split_fraction=0.05)  # Use 5% for validation
    tokenizer = litgpt.Tokenizer(f"checkpoints/{model_id}")
    data.connect(tokenizer, batch_size=1, max_seq_length=512)


    # Create the PyTorch Lightning Trainer with dynamic device count
    trainer = L.Trainer(
        devices=device_count,
        max_epochs=1,               # Train for 1 epoch
        max_steps=50,               # Stop after 50 steps
        accumulate_grad_batches=8,  # Accumulate gradients over 8 batches
        precision="bf16-true",      # Use bfloat16 precision for training
    )

    # Perform validation after training
    return validate(trainer, data, "")

def running():
    # Hyperparameters for evaluation
    with open("HP_config.json") as file:
        HP = json.load(file)
    print("inside HP printing : ",HP)
    # Check if "device_number" is specified in HP, else count available GPUs
    device_count = HP.get("device_number", torch.cuda.device_count())
    #Init the dist env
    dist.init_process_group(
        backend = "nccl",
        rank=0,
        world_size = device_count,
    )
    # Run the evaluation and print the validation result
    out = BB_eval(HP)
    cleanup_distributed_environment()
    return out

if __name__ == "__main__":
    out = running()
    print(out)

