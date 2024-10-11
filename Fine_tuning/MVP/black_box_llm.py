#Import librairy
import litgpt
import torch
import lightning as L
from litgpt.lora import GPT, merge_lora_weights
from litgpt.data import Alpaca2k
import math


model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

#Define a LLM class
class LitLLM(L.LightningModule):
    def __init__(self,low_rank=4,lora_alpha=16,lr=0.0002):
        super().__init__()
        self.model = GPT.from_name(
            name="tiny-llama-1.1b",
            #lora_r=4, 
            lora_r=low_rank, 
            #lora_alpha=16,
            lora_alpha=lora_alpha, 
            lora_dropout=0.05, 
            lora_query=True, 
            lora_key=False, 
            lora_value=True,
        )
        self.learning_rate = lr
        litgpt.lora.mark_only_lora_as_trainable(self.model)
    
    def on_train_start(self):
        state_dict = torch.load("checkpoints/"+model_id+"/"+"/lit_model.pth", mmap=True)
        self.model.load_state_dict(state_dict, strict=False)

    def training_step(self, batch):
        input_ids, targets = batch["input_ids"], batch["labels"]
        logits = self.model(input_ids)
        loss = litgpt.utils.chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        warmup_steps = 10
        #optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0002, weight_decay=0.0, betas=(0.9, 0.95))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup_steps)
        return [optimizer], [scheduler]
    
    def validation_step(self, batch):
        input_ids, targets = batch["input_ids"], batch["labels"]
        logits = self.model(input_ids)
        loss = litgpt.utils.chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:])
        self.log("train_loss", loss, prog_bar=True)
        return loss

def loss_function(HP):
    loRA, l_alpha, lr = HP
    lr = math.exp(lr)
    loRA = round(loRA)
    l_alpha = round(l_alpha) 
    model = LitLLM(loRA, l_alpha, lr)

    data = Alpaca2k(val_split_fraction=0.2)
    tokenizer = litgpt.Tokenizer("checkpoints/"+model_id+"/")
    data.connect(tokenizer, batch_size=1, max_seq_length=512)

    trainer = L.Trainer(
        devices=1,
        max_epochs=1,
        max_steps = 10,
        accumulate_grad_batches=8,
        precision="bf16-true",
        limit_val_batches = 10
    )

    #Train the model
    trainer.fit(model, data)
    merge_lora_weights(model.model)
    trainer.save_checkpoint("/kaggle/working/checkpoints/finetuned.ckpt", weights_only=True)
    out = trainer.validate(model,dataloaders = data,verbose = True)
    return out[0]["train_loss"]

a = loss_function((2, 16, 0.002))
print(a)