import sys
import json
import torch
import litgpt
from litgpt.lora import GPT, merge_lora_weights
from litgpt.data import Alpaca2k
import lightning as L
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
import numpy as np

import json
from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path
from lightning import LightningDataModule
from litgpt import Tokenizer
from litgpt.data import Alpaca2k, SFTDataset
from litgpt.prompts import PromptStyle
from datasets import load_dataset

model_id = "meta-llama/Meta-Llama-3-8B"
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
bar = False

class LitLLM(L.LightningModule):
    def __init__(self, low_rank=4, rate=0.002, l_alpha=16, l_dropout=0.05,bar = True):
        """
        Initialize the LitLLM model with specified LoRA parameters.

        Args:
        - low_rank (int): LoRA rank, controls the low-rank decomposition.
        - rate (float): Learning rate.
        - l_alpha (int): LoRA scaling factor.
        - l_dropout (float): Dropout rate for LoRA layers.
        - bar (bool): A boolean flag for additional configuration.

        The GPT model is initialized with the provided LoRA parameters, and only LoRA layers are marked as trainable.
        """
        super().__init__()
        # Parameters
        self.lr = rate
        self.bar = bar
        self.validation_step_outputs = []
        # Lora Model
        self.model = GPT.from_name(
            # name list : https://github.com/Lightning-AI/litgpt/blob/main/litgpt/config.py#L164
            name="Llama-3.1-8B{}",
            lora_r=low_rank,
            lora_alpha=l_alpha,
            lora_dropout=l_dropout,
            lora_query=True,
            lora_key=False,
            lora_value=True
        )
        litgpt.lora.mark_only_lora_as_trainable(self.model)
    
    def compute_loss(self, input_ids, targets):
        """
        Compute the cross-entropy loss for the model.

        Args:
        - input_ids: Input token IDs for the model.
        - targets: Target token IDs for the loss calculation.

        Returns:
        - loss: The computed cross-entropy loss.
        """
        if torch.isnan(input_ids).any():
            print("NaN detected in input")

        if torch.isnan(targets).any():
            print("NaN detected in targets")
        
        logits = self.model(input_ids)
        loss = litgpt.utils.chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:])
        
        if torch.isnan(loss).any():
            print("NaN detected in loss")

        return loss


    #------------------------------ Training ------------------------------
    def on_train_start(self):
        """
        Load the pre-trained model weights at the start of training.

        This function is called once at the beginning of training. It loads the pre-trained model weights
        from the specified checkpoint and loads them into the model. The strict=False argument allows
        loading weights even when the model architecture is slightly different.
        """
        state_dict = torch.load(f"checkpoints/{model_id}/lit_model.pth", mmap=True, weights_only=False)
        self.model.load_state_dict(state_dict, strict=False)

    def training_step(self, batch):
        """
        Define the training loop's step. This method is called on each batch.

        Args:
        - batch: The input data batch, containing 'input_ids' and 'labels'.

        Returns:
        - loss: The computed cross-entropy loss for the current batch.
        """
        input_ids, targets = batch["input_ids"], batch["labels"]
        loss = self.compute_loss(input_ids, targets)
        self.log("train_loss", loss, prog_bar=self.bar)
        return loss
    
    def on_train_epoch_end(self):
        """
        Called at the end of each training epoch. This method is a hook
        provided by PyTorch Lightning.

        This implementation does nothing, as manual checkpoint saving is
        not needed with PyTorch Lightning.
        """
        pass  # Disable manual checkpoint saving


    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        This method is a hook provided by PyTorch Lightning and is called
        automatically when the Trainer is initialized. It returns a tuple
        containing the optimizer and the learning rate scheduler.

        The optimizer is an instance of AdamW, with a learning rate of
        self.lr and a weight decay of 1e-2. The betas are set to (0.9, 0.95).

        The learning rate scheduler is an instance of LambdaLR, with a
        schedule defined by the lambda function lambda step: step / 10.
        This schedule will increase the learning rate linearly for the first
        10 steps, and then keep it constant.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-2, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / 10)
        return [optimizer], [scheduler]
    
    #------------------------------ Validate ------------------------------
    def validation_step(self, batch):
        """
        Called at each validation step.

        This method is a hook provided by PyTorch Lightning.

        The input batch is unpacked into input_ids and targets, and the
        loss is computed using the compute_loss method. The loss is then
        logged to the logger with the key "val_loss", and appended to the
        validation_step_outputs list.

        Args:
        - batch: A dictionary containing the input_ids and labels for the
          validation batch.

        Returns:
        - loss: The computed loss for the validation batch.
        """
        input_ids, targets = batch["input_ids"], batch["labels"]
        loss = self.compute_loss(input_ids, targets)
        self.validation_step_outputs.append(loss)
        self.log("val_loss", loss, prog_bar=self.bar)
        return loss
    
    def on_validation_epoch_end(self):
        """
        Called at the end of each validation epoch.

        This method is a hook provided by PyTorch Lightning.

        The method stacks all the validation losses computed in the
        validation_step method, computes the mean of this stack, and logs
        it to the logger with the key "val_loss_avg".

        Args:
        - None

        Returns:
        - None
        """
        loss_total = torch.stack(self.validation_step_outputs)
        self.log("val_loss_avg", loss_total.mean())
        return super().on_validation_epoch_end()


@dataclass
class LLMDataModule(LightningDataModule):
    mask_prompt: bool = False
    val_split_fraction: float = 0.05
    prompt_style: Union[str, PromptStyle] = "alpaca"
    ignore_index: int = -100
    seed: int = 42
    num_workers: int = 4
    download_dir: Path = Path("./data/alpaca2k")
    repo_id: str = field(repr=False, default="mhenrichsen/alpaca_2k_test")
    file_name: str = field(repr=False, default="alpaca2k_data_cleaned_archive.json")

    tokenizer: Optional[Tokenizer] = field(default=None, init=False, repr=False)
    batch_size: int = field(default=1, init=False, repr=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    train_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)
    val_dataset: Optional[SFTDataset] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Initialize the data module.

        This method is called after the data module has been initialized. It sets the
        prompt_style attribute to an instance of PromptStyle if it is given as a string.
        """
        super().__init__()
        if isinstance(self.prompt_style, str):
            self.prompt_style = PromptStyle.from_name(self.prompt_style)

    def connect(self, tokenizer: Optional[Tokenizer] = None, batch_size: int = 1, max_seq_length: Optional[int] = None) -> None:
        """
        Configure the data module with the specified tokenizer, batch size, and maximum sequence length.

        Args:
        - tokenizer (Optional[Tokenizer]): The tokenizer to be used for processing text. If not provided, defaults to None.
        - batch_size (int): The number of samples per batch for training and validation. Defaults to 1.
        - max_seq_length (Optional[int]): The maximum sequence length for tokenized inputs. If None, defaults to -1.

        Returns:
        - None
        """
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = -1 if max_seq_length is None else max_seq_length

    def prepare_data(self) -> None:
        """
        Download the dataset from Hugging Face.

        This method downloads the dataset from Hugging Face and caches it in the specified directory.

        Args:
        - None

        Returns:
        - None
        """
        load_dataset(self.repo_id, cache_dir=self.download_dir)

    def setup(self, stage: str = None) -> None:
        """
        Prepare the data module for training and validation.

        This method loads the dataset from Hugging Face, splits it into training and
        validation sets, and creates SFTDataset instances for training and
        validation.

        Args:
        - stage (str): The stage for which the data module is being prepared. If not
          provided, defaults to None.

        Returns:
        - None
        """
        # Load the dataset
        dataset = load_dataset(self.repo_id, cache_dir=self.download_dir)

        # Split the dataset into training and validation sets
        train_validation_split = dataset["train"].train_test_split(test_size=self.val_split_fraction, seed=self.seed)
        train_data = train_validation_split["train"]
        val_data = train_validation_split["test"]

        # Create SFTDataset instances for training and validation
        self.train_dataset = SFTDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        self.val_dataset = SFTDataset(
            data=val_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )

    def train_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the training dataset.

        This method creates a DataLoader instance with the training dataset and
        returns it.

        Returns:
        - DataLoader: The DataLoader instance for the training dataset.
        """
        out = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(self.seed)
        )
        return out

    def val_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the validation dataset.

        This method creates a DataLoader instance with the validation dataset
        and returns it.

        Returns:
        - DataLoader: The DataLoader instance for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def BB_eval(HP):
    """
    Evaluate the model with the given hyperparameters by training and validating.

    Args:
    - HP: Dictionary containing hyperparameters such as 'learning_rate', 'lora_rank', and 'grad_batches'.

    Returns:
    - validation_loss: The validation loss of the model.
    """

    # Hyper Parameters loading
    grad_batches = HP.get("grad_batches", 16)
    rate = HP.get("learning_rate", 0.002)
    low_rank = HP.get("lora_rank", 4)
    fast_run = HP.get("fast_run", True)

    if fast_run :
        max_steps = 20
    else:
        max_steps = 2000

    # Data module management
    data_module = LLMDataModule(
        val_split_fraction=0.2,  # Adjust as needed
    )
    data_module.connect(
        tokenizer=litgpt.Tokenizer(f"checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        batch_size=1,
        max_seq_length=512
    )
    data_module.prepare_data()
    data_module.setup()

    # Configure Trainer
    trainer = L.Trainer(
            devices=1,
            max_epochs=2,
            max_steps=max_steps,
            accumulate_grad_batches=grad_batches,
            precision="32-true",
            enable_checkpointing=False,
        )
    

    # Generate and train the model
    
    model = LitLLM(low_rank=low_rank, rate=rate)
    trainer.fit(model, datamodule = data_module)

    # Merge and compute validation loss
    merge_lora_weights(model.model)
    out = trainer.validate(model, datamodule = data_module)

    validation_loss = out[0]["val_loss_avg"]

    return validation_loss


if __name__ == "__main__":
    # Hyper Parameters
    HP = {
        "learning_rate": 0.002,
        "lora_rank": 2,
    }

    out = BB_eval(HP)
    print("final output : ",out)
