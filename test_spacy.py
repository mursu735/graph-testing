import spacy
import en_core_web_trf
import en_core_web_sm
import re

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")
#nlp = en_core_web_trf.load()
line = '["Moby Dick", "could\'ve", "Ahab\'s obsession"]'
line = line.replace("[", "").replace("]", "").replace('"', '').replace("\n", "")
my_list = re.split(r" |'", line)
print(my_list)
doc = nlp(line)
print([(w.text.replace("'", ""), w.pos_) for w in doc])

#line = '["Bildad", "hires", "Ishmael"]'
#line = line.replace("[", "").replace("]", "").replace('"', '').replace("\n", "")
#doc = nlp(line)
#print([(w.text, w.pos_) for w in doc])

#for ent in doc.ents:
#    print(ent.text, ent.label_)