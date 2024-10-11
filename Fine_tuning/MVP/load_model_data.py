#Load model
import litgpt
from litgpt.scripts.download import download_from_hub

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#model = download_from_hub(model_id)

#Download model
try :
    os.listdir("checkpoints/"+model_id+"/")
except : 
    download_from_hub(model_id)

from litgpt.data import Alpaca2k
data = Alpaca2k(val_split_fraction=0.2)
