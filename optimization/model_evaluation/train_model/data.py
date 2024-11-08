from dataclasses import dataclass, field
from typing import Optional, Union

import torch
from torch.utils.data import DataLoader

import lightning as L
from litgpt.prompts import PromptStyle
from litgpt import Tokenizer
from litgpt.data import SFTDataset

from datasets import load_dataset

@dataclass
class LLMDataModule(L.LightningDataModule):
    mask_prompt: bool = False
    val_split_fraction: float = 0.05
    prompt_style: Union[str, PromptStyle] = "alpaca"
    ignore_index: int = -100
    seed: int = 42
    num_workers: int = 4
    repo_id: str = field(repr=False, default="tatsu-lab/alpaca")
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
        load_dataset(self.repo_id)

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
        dataset = load_dataset(self.repo_id)

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
            persistent_workers=True,
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
            persistent_workers=True
        )

