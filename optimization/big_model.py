from model_evaluation import evaluate

if __name__ == "__main__":
    HP = {"fast_run" : True,
          #"model_id" : "meta-llama/Meta-Llama-3.1-8B",
          "model_id" : "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
          "nb_device" : 2}
    evaluate(HP)