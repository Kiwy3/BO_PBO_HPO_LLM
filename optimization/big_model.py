from model_evaluation import evaluate

if __name__ == "__main__":
    HP = {"fast_run" : True,
          "model_id" : "meta-llama/Meta-Llama-3.1-8B",
          "nb_device" : 1}
    evaluate(HP)