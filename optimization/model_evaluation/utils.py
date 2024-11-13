import json

def quantize_plug(quantize = "bnb.nf4", precision="16-true"):
    import torch
    from lightning.pytorch.plugins.precision.bitsandbytes import BitsandbytesPrecision
    dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
    plugins = BitsandbytesPrecision(quantize[4:], dtype)
    return plugins

def load_hyperparameters():
    with open("optimization/export.json","r") as file:
        
        for line in file:
            pass
    return json.loads(line)

def add_results(results, file = "optimization/export.json"):
    

    with open(file, 'r+') as f:
        lines = f.readlines()
        last_line = lines[-1]
        last_line_data = json.loads(last_line)
        last_line_data['results'] = results
        lines[-1] = json.dumps(last_line_data) + '\n'
        f.seek(0)
        f.writelines(lines)
        f.truncate()

def load_config():
    with open("optimization/config.json") as f:
        config = json.load(f)
    return config["hyperparameters"], config["models"], config["experiment"]