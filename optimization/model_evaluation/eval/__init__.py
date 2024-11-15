__all__ = ["evaluate"]

from model_evaluation.eval.hf_eval import convert_and_evaluate as task_evaluate
from model_evaluation.utils import load_hyperparameters, add_results, load_config


def evaluate():
    lora_path = "checkpoints/lora"
    _, _, experiment = load_config()
    tasks = experiment["tasks"]
    eval_limit = experiment["eval_limit"]
    results = task_evaluate(lora_path,
                        tasks=tasks[0] if len(tasks) == 1 else tasks,
                        limit=eval_limit,
                        force_conversion=True,
                        out_dir="eval/")
    res = {}
    for task in tasks:
        res[task] =  results[task]["acc,none"]
    return res