from transformers import DistilBertTokenizer

EPOCHS = 5
MAX_LEN = 512
DEVICE = "cpu"
RANDOM_SEED = 123
MODEL_PATH = "./saved_model"
tokenizer = DistilBertTokenizer.from_pretrained(
    'distilbert-base-turkish-cased', 
    do_lower_case=True
)
