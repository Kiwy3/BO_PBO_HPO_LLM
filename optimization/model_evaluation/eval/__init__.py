__all__ = ["evaluate"]

from model_evaluation.eval.hf_eval import convert_and_evaluate as task_evaluate
from model_evaluation.utils import load_hyperparameters, add_results


def evaluate():
    tasks = ["mmlu"]
    lora_path = "checkpoints/lora"
    HP = load_hyperparameters()["experiment"]
    eval_limit = HP["eval_limit"]
    results = task_evaluate(lora_path,
                        tasks=tasks[0] if len(tasks) == 1 else tasks,
                        limit=eval_limit,
                        force_conversion=True,
                        out_dir="eval/")
    res = {}
    for task in tasks:
        res[task] =  results[task]["acc,none"]
    add_results(results=res,) 
    return res