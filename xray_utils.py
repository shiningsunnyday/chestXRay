import pandas as pd
import numpy as np
import gensim

def get_from_csv(entries_path, split):

    data = pd.read_csv(f'{entries_path}/{split}_entries.csv')[['label','xray_paths','text']]
    # Adjusting labels to fit with Snorkel MeTaL labeling convention (0 reserved for abstain)
    data['label'][data['label']==0] = 2
    perc_pos = sum(data['label']==1)/len(data)
    print(f'{len(data)} {split} examples: {100*perc_pos:0.1f}% Abnormal')
        
    return data

def read_corpus(reports):
    for i, line in enumerate(reports):
        tokens = gensim.utils.simple_preprocess(line)
        yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
        
def report2vec(reports, file=None, file_save="report2vec.model"):
   
    if file:
        return gensim.utils.SaveLoad.load(file)
    
    corpus = list(read_corpus(reports))
    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=1, epochs=40)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(file_save)
    return model
