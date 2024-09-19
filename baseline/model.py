# model.py
from transformers import T5ForConditionalGeneration, T5Tokenizer

class T5Model:
    def __init__(self, model_name='t5-base'):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer
