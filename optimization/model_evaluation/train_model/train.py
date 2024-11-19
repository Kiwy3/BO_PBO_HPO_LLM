# Classical libraries
from pathlib import Path
import json

# Torch libraries
import torch
import litgpt
import lightning as L

# custom librairies
from model_evaluation.train_model import LLM_model, merge_lora_weights, lora_filter
from model_evaluation.train_model import LLM_model
from model_evaluation.train_model import LLMDataModule
from model_evaluation.utils import load_hyperparameters



def training():

    """
    Train a model with given hyperparameters and experiment settings.

    Args:
    - HP: a dictionary with two keys: "hyperparameters" and "experiment". The first one contains all the hyperparameters
          to be used for training, and the second one contains the experiment settings such as model name, device, and
          number of devices.
    """
    HP = load_hyperparameters()
    hyperparameters = HP["hyperparameters"]
    
    
    # Hyper Parameters loading
    grad_batches = hyperparameters.get("grad_batches", 4)
    rate = hyperparameters.get("learning_rate", 0.002)
    low_rank = hyperparameters.get("lora_rank", 4)
    lora_dropout = hyperparameters.get("lora_dropout", 0.05)
    lora_alpha = hyperparameters.get("lora_alpha", 16)
    weight_decay = hyperparameters.get("weight_decay", 1e-2)

    # experiment parameters
    with open("optimization/config.json") as f:
        experiment = json.load(f)["experiment"]
    model_id = experiment.get("model_id","TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model_name = experiment.get("model_name","tiny-llama-1.1b")
    device = torch.device(
        experiment.get("device", "cuda" if torch.cuda.is_available() else "cpu" ))
    nb_device = experiment.get("nb_device", torch.cuda.device_count())
    nb_device = min(nb_device,torch.cuda.device_count())
    epochs = experiment.get("epochs", 1)
    max_steps = 20 if experiment.get("fast_run", True) else 2000
    strategy = experiment.get("strategy", "ddp_spawn")
    dataset = experiment.get("dataset","tatsu-lab/alpaca")

    # Set the precision for A100
    torch.set_float32_matmul_precision('medium')

    # Data module management
    data_module = LLMDataModule(
        val_split_fraction=0.05,  # Adjust as needed
        repo_id=dataset

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
            strategy=strategy,
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
    
    # Training
    trainer.fit(model, datamodule = data_module)

    # Saving merged model
    print("\t merging and saving")
    merge_lora_weights(model.model)
    state_dict = {k.replace("linear.", ""): v for k, v in model.model.state_dict().items() if not lora_filter(k, v)}
    save_path = Path("checkpoints/lora") / "lit_model.pth"
    torch.save(state_dict, save_path)