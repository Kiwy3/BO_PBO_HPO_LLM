import os
import torch # type: ignore
import litgpt # type: ignore
from litgpt.lora import GPT, merge_lora_weights # type: ignore
from litgpt.data import Alpaca2k # type: ignore
import lightning as L # type: ignore
import torch.distributed as dist # type: ignore

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
T = {}

"""Make a LLM class from a Lightning Module"""
class LitLLM(L.LightningModule):
    def __init__(self,low_rank = 4,rate = 0.002,l_alpha = 16, l_dropout = 0.05):
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
        if (dist.get_rank()==12):
            print(batch)
            print(input_ids.shape, targets.shape)
            print(loss)
            print(logits, logits.shape,"\n\n\n")
        self.log("train_loss", loss, prog_bar=True)
        return loss

"""----------------------------------Make functions to call training and evaluation------------------"""
def train(trainer, data,HP, suffix = ""):

    #Check if the rate is in the HP dict
    if "learning_rate" in HP.keys():
        rate = HP["learning_rate"]
    else : rate = 0.002

    #Check if the lora rank is in the HP dict
    if "lora_rank" in HP.keys():
        low_rank = HP["lora_rank"]
    else : low_rank = 4


    with trainer.init_module(empty_init=True):
        model = LitLLM(low_rank = low_rank,rate=rate)
    trainer.fit(model, data)
    merge_lora_weights(model.model)
    trainer.save_checkpoint("checkpoints/finetuned"+suffix+".ckpt", weights_only=True)

def validate(trainer, data,suffix=""):
    model = LitLLM.load_from_checkpoint("checkpoints/finetuned"+suffix+".ckpt")
    out = trainer.validate(model,dataloaders = data,verbose = True)
    return out

def BB_eval(HP):

    data = Alpaca2k(val_split_fraction=0.05)
    tokenizer = litgpt.Tokenizer("checkpoints/"+model_id)
    data.connect(tokenizer, batch_size=1, max_seq_length=512)

    trainer = L.Trainer(
        devices=4,
        max_epochs=1,
        max_steps = 50,
        accumulate_grad_batches=8,
        precision="bf16-true",
        #limit_val_batches = 10
    )

    #train(trainer,data,HP)
    return validate(trainer, data,"")


HP = {"learning_rate" : 0.001,
      "lora_rank" : 4}
val_loss = 0

out = BB_eval(HP)

print(out)


"""
output = [None,None,None,None]
A = dist.all_gather_object(output,out)
print(A)
print(torch.distributed.get_rank())


tens_out = torch.tensor(out[0]["val_loss"]).to("cuda")
print(tens_out)
A = dist.all_reduce(tens_out, op=dist.ReduceOp.SUM)
print(A)"""