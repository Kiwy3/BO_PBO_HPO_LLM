import os
import json
import torch
import litgpt
from litgpt.lora import GPT, merge_lora_weights
from litgpt.data import Alpaca2k
import lightning as L
import torch.distributed as dist

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

class LitLLM(L.LightningModule):
    def __init__(self, low_rank=4, rate=0.002, l_alpha=16, l_dropout=0.05):
        super().__init__()
        self.lr = rate
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
        self.validation_step_outputs = []
        self.training_step_outputs = []

    def on_train_start(self):
        state_dict = torch.load(f"checkpoints/{model_id}/lit_model.pth", mmap=True, weights_only=False)
        self.model.load_state_dict(state_dict, strict=False)

    def compute_loss(self, input_ids, targets):
        logits = self.model(input_ids)
        loss = litgpt.utils.chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:])
        return loss

    def training_step(self, batch):
        input_ids, targets = batch["input_ids"], batch["labels"]
        loss = self.compute_loss(input_ids, targets)
        self.training_step_outputs.append(loss)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.training_step_outputs).mean()
        self.log("avg_train_loss", epoch_average)

    def validation_step(self, batch):
        input_ids, targets = batch["input_ids"], batch["labels"]
        loss = self.compute_loss(input_ids, targets)
        self.validation_step_outputs.append(loss)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.0, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / 10)
        return [optimizer], [scheduler]

def train(trainer, data, HP, suffix=""):
    rate = HP.get("learning_rate", 0.002)
    low_rank = HP.get("lora_rank", 4)

    with trainer.init_module(empty_init=True):
        model = LitLLM(low_rank=low_rank, rate=rate)
    trainer.fit(model, data)
    merge_lora_weights(model.model)
    trainer.save_checkpoint(f"checkpoints/finetuned{suffix}.ckpt", weights_only=True)

    return model

def validate(trainer, data, suffix=""):
    rate = HP.get("learning_rate", 0.002)
    low_rank = HP.get("lora_rank", 4)

    with trainer.init_module(empty_init=True):
        model = LitLLM(low_rank=low_rank, rate=rate)
    model = LitLLM.load_from_checkpoint(f"checkpoints/finetuned{suffix}.ckpt")

    trainer.validate(model, dataloaders=data, verbose=True)
    outputs = torch.stack(model.validation_step_outputs)

    return outputs

def BB_eval(HP):
    device_count = HP.get("device_number", torch.cuda.device_count())
    data = Alpaca2k(val_split_fraction=0.05)
    tokenizer = litgpt.Tokenizer(f"checkpoints/{model_id}")
    data.connect(tokenizer, batch_size=1, max_seq_length=512)

    trainer = L.Trainer(
        devices=1,
        max_epochs=1,
        max_steps=20,
        accumulate_grad_batches=8,
        precision="bf16-true",
    )
    x = train(trainer, data, HP)
    epoch_average = torch.stack(x.training_step_outputs)
    print(epoch_average)
    out = validate(trainer, data, "")

    return out

if __name__ == "__main__":
    HP = {
        "learning_rate": 0.002,
        "lora_rank": 4,
    }
    out = BB_eval(HP)
    print(out)
