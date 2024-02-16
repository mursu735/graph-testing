# The code below is a modified version of the forllowing file.
# sentence-transformers: examples/unsupervised_learning/SimCSE/train_simcse_from_file.py

from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, InputExample
import logging
from datetime import datetime
import sys
import os
import spacy
import helpers

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# Training parameters
model_name = 'distilroberta-base'
train_batch_size = 128
max_seq_length = 32
num_epochs = 1

sentences = []

files = helpers.natural_sort(os.listdir("input/Chapters"))
#files = ["1.txt"]

nlp = spacy.load('en_core_web_sm')

print(files)
for file in files:
    with open(f"input/Chapters/{file}", encoding="utf-8") as file:
        text = file.read()
    #paragraphs = text.split("\n\n")
    #paragraphs = [s.strip().replace("\n", " ") for s in paragraphs]
    #paragraphs = list(filter(None, paragraphs))
    chapter = nlp(text).sents
    for sentence in chapter:
        #print(sent.text.strip(), "\n----------------------")
        sentences.append(sentence.text.strip())

print(sentences)
model_output_path = f'dump/en_simsce_80days-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
model_save_path = "dump/en_simcse_80days"


# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

################# Read the train corpus  #################
train_samples = []
for sentence in sentences:
    sentence = sentence.strip()
    train_samples.append(InputExample(texts=[sentence, sentence]))

logging.info("Train sentences: {}".format(len(train_samples)))

# We train our model using the MultipleNegativesRankingLoss
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)
train_loss = losses.MultipleNegativesRankingLoss(model)

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          optimizer_params={'lr': 5e-5},
          checkpoint_path=model_output_path,
          output_path=model_save_path,
          show_progress_bar=True,
          use_amp=False  # Set to True, if your GPU supports FP16 cores
          )
