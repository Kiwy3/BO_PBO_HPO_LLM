import torch
import lightning as L
import litgpt
from pathlib import Path

# Import Custom libraries
from model_evaluation.train_model import LLM_model, merge_lora_weights, lora_filter
from optimization.model_evaluation.train_model.data import LLMDataModule
from model_evaluation.eval import task_evaluate
from optimization.model_evaluation.utils import quantize_plug


model_dict = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0":"tiny-llama-1.1b",
    "meta-llama/Meta-Llama-3.1-8B":"Llama-3.1-8B",

}

def evaluate(HP):

    """
    Evaluates a model with specified hyperparameters, trains it, and returns the evaluation accuracy.

    Args:
    - HP (dict): A dictionary containing hyperparameters such as 'grad_batches', 'learning_rate',
      'lora_rank', 'lora_dropout', 'lora_alpha', 'device', 'nb_device', 'weight_decay', 'epochs',
      'fast_run', and 'idx'.

    Returns:
    - float: The evaluation accuracy of the model on the "mmlu" task.
    """
    hyperparameters = HP["hyperparameters"]
    experiment = HP["experiment"]

    
    # Hyper Parameters loading
    grad_batches = hyperparameters.get("grad_batches", 4)
    rate = hyperparameters.get("learning_rate", 0.002)
    low_rank = hyperparameters.get("lora_rank", 4)
    lora_dropout = hyperparameters.get("lora_dropout", 0.05)
    lora_alpha = hyperparameters.get("lora_alpha", 16)
    weight_decay = hyperparameters.get("weight_decay", 1e-2)

    model_id = experiment.get("model_id","TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model_name = experiment.get("model_name","tiny-llama-1.1b")
    device = torch.device(
        experiment.get("device", "cuda" if torch.cuda.is_available() else "cpu" ))
    nb_device = experiment.get("nb_device", torch.cuda.device_count())
    epochs = experiment.get("epochs", 1)
    max_steps = 20 if experiment.get("fast_run", True) else 2000
    eval_limit = experiment.get("eval_limit", 50)

    # Set the precision for A100
    torch.set_float32_matmul_precision('medium')

    # Data module management
    data_module = LLMDataModule(
        val_split_fraction=0.2,  # Adjust as needed
)
    data_module.connect(
        tokenizer=litgpt.Tokenizer(f"checkpoints/{model_id}"),
        batch_size=1,
        max_seq_length=512
    )
    data_module.setup()

    # Configure Trainer
    trainer = L.Trainer(
            devices=nb_device,
            max_epochs=epochs,
            max_steps=max_steps,
            strategy="ddp_spawn",
            #strategy="ddp",
            accumulate_grad_batches=grad_batches,
            precision="16-mixed",
            enable_checkpointing=False,
            #plugins=quantize_plug(),
        )
    
    # Generate and train the model
    model = LLM_model(
        low_rank=low_rank, 
        rate=rate,
        l_alpha=lora_alpha,
        l_dropout=lora_dropout,
        weight_decay = weight_decay,
        model_name=model_name,
        model_id=model_id        
        ).to(device)
    
    trainer.fit(model, datamodule = data_module)
    
    #if not trainer.is_global_zero: exit(0)

    # Saving unmerged model
    lora_path = "checkpoints/lora"
    print("Saving before LoRA")
    torch.save({k : v for k, v in model.model.state_dict().items()}
        , Path(lora_path) / "lit_model_full.pth"
    )

    # Merge
    print("Merging Lora Weights")
    merge_lora_weights(model.model)


    # Saving merged model
    print("merging model")
    state_dict = {k.replace("linear.", ""): v for k, v in model.model.state_dict().items() if not lora_filter(k, v)}
    
    #print(state_dict.keys())
    save_path = Path(lora_path) / "lit_model.pth"
    torch.save(state_dict, save_path)
    



    # Evaluate the model
    print("Evaluating model")
    out = task_evaluate(lora_path,
                          tasks="mmlu",
                          limit=eval_limit,
                          force_conversion=True,
                          out_dir="eval/")

    return out["mmlu"]["acc,none"]

if __name__ == "__main__":
    HP = {"fast_run" : True}
    import os
    print(os.getcwd())
    out = evaluate(HP)
    print(out)