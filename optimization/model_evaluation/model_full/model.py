# Librairies import
import torch
import pytorch_lightning as L
import litgpt
from model_evaluation.model_full.lora import GPT

class LitLLM(L.LightningModule):
    
    def __init__(self, low_rank=4, rate=0.002, l_alpha=16, l_dropout=0.05,weight_decay = 1e-2,bar = False, model_name = "tiny-llama-1.1b", model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initialize the LitLLM model with specified LoRA parameters.

        Args:
            low_rank (int): LoRA rank, controls the low-rank decomposition.
            rate (float): Learning rate.
            l_alpha (int): LoRA scaling factor.
            l_dropout (float): Dropout rate for LoRA layers.
            bar (bool): A boolean flag for additional configuration.
            name (str): The name of the pre-trained model to use.
            model_id (str): The model ID to use for the pre-trained model.

        The GPT model is initialized with the provided LoRA parameters, and only LoRA layers are marked as trainable.
        """
        super().__init__()
        # Parameters
        self.lr = rate
        self.bar = bar
        self.weight_decay = weight_decay
        self.validation_step_outputs = []
        self.old_lora_A = []
        self.model_id = model_id
        # LLM Model
        self.model = GPT.from_name(
            name=model_name,
            lora_r=low_rank,
            lora_alpha=l_alpha,
            lora_dropout=l_dropout,
            lora_query=True,
            lora_key=True,
            lora_value=True
        )
        # Mark only LoRA layers as trainable
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

    def check_lora_b_zeros(self):
        name = self.model.state_dict().keys()
        for n in name:
            if "lora_B" in n:
                if torch.count_nonzero(self.model.state_dict()[n]) == 0:
                    #print(f"lora_b {n} is all zero")
                    pass
                else:
                    print(f"lora_b {n} is not all zero")

    def compare_lora_A(self):
        old = self.old_lora_A.copy()
        for n, p in self.model.named_parameters():
            if "lora_A" in n:
                self.old_lora_A.append(p)

        if old == self.old_lora_A:
            print("lora_A is same")
        else:
            print("lora_A is not same")

    #------------------------------ Training ------------------------------
    def on_train_start(self):
        """
        Load the pre-trained model weights at the start of training.

        This function is called once at the beginning of training. It loads the pre-trained model weights
        from the specified checkpoint and loads them into the model. The strict=False argument allows
        loading weights even when the model architecture is slightly different.
        """
        torch.set_float32_matmul_precision('medium')
        state_dict = torch.load(f"checkpoints/{self.model_id}/lit_model.pth", mmap=True, weights_only=False)
        self.model.load_state_dict(state_dict, strict=False)

        for n, p in self.model.named_parameters():
            if "lora_A" in n:
                self.old_lora_A.append(p)

        

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
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.95))
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
        self.log("val_loss", loss, prog_bar=self.bar, sync_dist=True)
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
        loss_total = torch.stack(self.validation_step_outputs).mean()
        mean = self.all_gather(loss_total)
        self.log("val_loss_avg", torch.mean(mean), sync_dist=True)
        return super().on_validation_epoch_end()
