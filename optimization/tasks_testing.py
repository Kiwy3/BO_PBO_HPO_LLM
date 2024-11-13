import json

# custom librairies
from model_evaluation import training, evaluate
from utilities import convert, load_config



#from model_evaluation import evaluate
def evaluation_function(x):
    # convert x into hyperparameters
    hyperparameters = {}
    for i in range(len(hp_def.keys())):
        key = list(hp_def.keys())[i]
        hyperparameters[key] = convert(x,i, hp_def)

    # save hyperparameters
    HP = {"hyperparameters" : hyperparameters}

    # writing in the file   
    with open(export_file, "a+") as outfile:
        json.dump(HP, outfile)
        outfile.write('\n')

    result = evaluate()

    return result["mmlu"]


if __name__ == "__main__":
    export_file = "optimization/export.json"
    hp_def, model_dict, experiment = load_config() 
    experiment["model_name"] = model_dict[experiment["model_id"]]

    evaluation_function([0.004086771438464067, 17, 8, 40, 0.25, 0.25])