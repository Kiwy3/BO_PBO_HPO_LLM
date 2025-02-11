# Bayesian and Partition-Based Optimization for Hyperparameter Optimization of LLM Fine-Tuning

This repository is used for the afornamed article, written for *International Conference on Optimization and Learning (OLA2025)*. It aim to be reproducible, with a brief guide about how to do it. Since hardware needed for experiments is mostly on unix-managed cluster, this guide suppose unix installation.


## First time
When using this repo, there are things to do mostly for the first time, this part adress this. 

## Install the repo
first step, clone the repository and go the the directory : 

```
$ git clone https://github.com/Kiwy3/BO_PBO_HPO_LLM.git
$ cd BO_PBO_HPO_LLM
```

Then, create a virtual environment using python, to install all libraries listed in *requirements.txt*. 

```
$ python -m venv .env
$ source .env/bin/activate
$ pip install -r requirements.txt
```
## Download a model
For the first time, some model need specific steps. Models are downloaded from [HuggingFace](https://huggingface.co/), directly with *litgpt* command lines. 

First, to look at possible model, one can use *litgpt download list*, to obtain the list, and then use *litgpt download model_id* to download this model

```
$ litgpt download list
$ litgpt download meta-llama/Llama-3.2-1B
```

For model like Llama ones, or others with specific access, it's might be necessary to go to [HuggingFace](https://huggingface.co/), and request specific access to the model. 

With this access, it's possible to generate an access token on HuggingFace, to use it with litgpt like this : 
```
$ litgpt download meta-llama/Llama-3.2-1B  --access_token ACCESS_TOKEN
```


## Launch an experiment
Based on grid5000 documentation, there are two ways to launch an experiment : using *passive* or *interactive* launch. 

### Interactive launch

The first step is to reserve computation ressources and activate the virtual environment. The following code is using example from *chuc* cluster.
```
$ oarsub -I -q testing -p chuc -l walltime=10
$ source .env/bin/activate
```
Then, the *cli.py* file is used to launch experiment from Command Line Interface. Example with Bayesian Optimization experiment. 
```
$ python cli.py bo
```

### Passive launch

For passive launch, shell script is used to manage the reservation and activate the virtual environment. This script is *launch.sh*, inside script folder. The script will pass an argument to *cli.py* to launch the experiment.Example with Bayesian Optimization experiment. 

```
$ oarsub -S "script/launch.sh bo"
```

To change experiment configuration, one must change it in *launch.sh* file, such as experiment time, or virtual environment path. 

## Make another experiment

To try another experiment, the first step is to describe the experiment with the creation of a file *exp_name.py*. This file import core class, and optimization algorithms from *algorithm* folder. Example for creating another Bayesian Optimization experiment, with a *bayesian.py* file.

```Python
''' bayesian.py '''

#import all packages
from hpo.core import ModelEval, SearchSpace
from hpo.algorithm import BoGp

# define a function with the experiment
def exp1():
    exp_name = "bo_exp1"

    # Create the search space
    search_space = {          
            "lora_rank" : {"min" : 2,"max" : 64,"type" : "int"},
            "lora_alpha" : {"min" : 1,"max" : 64,"type" : "int"},
            "lora_dropout" : {"min" : 0,"max" : 0.5,"type" : "float"},
            "learning_rate" : {"min" : -10,"max" : -1,"type" : "log"},
            "weight_decay" : {"min" : -5,"max" : -1,"type" : "log"}
        }

    space = SearchSpace(
        variables="search_space",
        savefile=f"{exp_name}.json") #used for saving all solutions

    # Create the objective function
    evaluator = ModelEval(
        search_space = space, 
        dev_run = "",
        experiment_name=exp_name,
        model_id="meta-llama/Llama-3.2-1B" # model inside litgpt lib
    )

    # Optimization algorithm class
    bo = BoGp( 
        #with some algorithm, other argument might be necessary
        space=space,
        maximizer=True, # whether it's a maximization or a minization
        obj_fun=evaluator.train_and_evaluate
    )
    
    # launch hyperparameter optimization
    bo.run(
        budget=50, #total budget
        init_budget=10
    )

```

Then, the experiment must be added in *cli.py*, with a librairy import and a key to launch experiment
```Python
from hpo.experiments import bayesian

'''
lines already in cli.py
...
'''

elif arg == "bo_exp1" : 
    bayesian.exp1()

else : #already in cli.py 
    print(f"Unknown argument : {arg}")

```

## Add another optimization algorithm 

TO DO


## Possibles errors

### Generating examples inside validation loop
During some test, I had trouble with litgpt finetune, because of one line generating example that was not working. 
To deal with it, you can open the *litgpt/finetune/lora.py* file inside the virtual env, and comment out line 371 (*generate_example(fabric, model, tokenizer, eval, data)*). 

```
$ cd .env/lib/python3.9/site-packages/litgpt/finetune/
$ emacs lora.py # or other editor
```
