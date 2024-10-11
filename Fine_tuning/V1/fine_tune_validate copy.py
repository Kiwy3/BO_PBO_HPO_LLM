import os
import torch # type: ignore
import litgpt # type: ignore
from litgpt.lora import GPT, merge_lora_weights # type: ignore
from litgpt.data import Alpaca2k # type: ignore
import lightning as L # type: ignore

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

class LitLLM(L.LightningModule):
    def __init__(self,low_rank,rate,l_alpha = 16, l_dropout = 0.05):
        super().__init__()
        self.lr = rate
        self.model = GPT.from_name(
            name="tiny-llama-1.1b",
            lora_r=low_rank, 
            lora_alpha=l_alpha, 
            lora_dropout=l_dropout, 
            lora_query=True, 
            lora_key=False, 
            lora_value=True,
        )
        litgpt.lora.mark_only_lora_as_trainable(self.model)
    
    def on_train_start(self):
        state_dict = torch.load("checkpoints/"+model_id+"/lit_model.pth", mmap=True,weights_only = False)
        self.model.load_state_dict(state_dict, strict=False)

    def training_step(self, batch):
        input_ids, targets = batch["input_ids"], batch["labels"]
        logits = self.model(input_ids)
        loss = litgpt.utils.chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        warmup_steps = 10
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.0, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup_steps)
        return [optimizer], [scheduler]
    
    def validation_step(self, batch):
        input_ids, targets = batch["input_ids"], batch["labels"]
        logits = self.model(input_ids)
        loss = litgpt.utils.chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:])
        self.log("train_loss", loss, prog_bar=True)
        return loss

def train(trainer,model, data,HP, suffix = ""):
    with trainer.init_module(empty_init=True):
        model = LitLLM(low_rank = 4,rate=0.002)
    trainer.fit(model, data)
    merge_lora_weights(model.model)
    trainer.save_checkpoint("checkpoints/finetuned.ckpt", weights_only=True)



if __name__ == "__main__":
    data = Alpaca2k(val_split_fraction=0.2)
    tokenizer = litgpt.Tokenizer("checkpoints/"+model_id)
    data.connect(tokenizer, batch_size=1, max_seq_length=512)

    trainer = L.Trainer(
        devices=2,
        max_epochs=1,
        max_steps = 50,
        accumulate_grad_batches=8,
        precision="bf16-true",
        #limit_val_batches = 10
    )
    with trainer.init_module(empty_init=True):
        model = LitLLM(low_rank = 4,rate=0.002)
        
train_i = False
if train_i : 
    test = trainer.fit(model, data)
    print(test)

    # Save final checkpoint
    merge_lora_weights(model.model)
    trainer.save_checkpoint("checkpoints/finetuned.ckpt", weights_only=True)
    
validate_i = True
if validate_i : 
    out = trainer.validate(model,dataloaders = data,verbose = True)
    print(out)