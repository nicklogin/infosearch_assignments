import json
import numpy as np
import tensorflow as tf
import pickle

from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

import sys
sys.path.append('./simple_elmo/')
from simple_elmo.elmo_helpers import load_elmo_embeddings, get_elmo_vectors

from pymorphy2 import MorphAnalyzer
from pymorphy2.tokenizers import simple_word_tokenize

## using hint from
## https://stackoverflow.com/questions/43203215/map-unique-strings-to-integers-in-python
def extract_texts(df):
    return list(i for i in set(df['question2']) if type(i) == str)

def dump_json(obj, fn):
    with open(fn, 'w', encoding='utf-8') as outp:
        json.dump(obj, outp, ensure_ascii=False)

def load_json(fn):
    with open(fn, 'r', encoding='utf-8') as inp:
        obj = json.load(inp)
    return obj

def dump(obj, fn):
    with open(fn, 'wb') as fout:
        pickle.dump(obj, fout)

def load(fn):
    with open(fn, 'rb') as fin:
        obj = pickle.load(fin)
    return obj

class Lemmatizer(MorphAnalyzer):
    def __call__(self, string):
        return [w.normal_form for w in [super(Lemmatizer, self).parse(word)[0] for word in simple_word_tokenize(string)] if w.tag.POS]

class BMVectorSearch:
    def __init__(self, texts, lemmatizer, k=2, b=0.75,
                 use_pretrained=False, index_file=None,
                 tf_file=None, idf_file=None, vocab_file=None):
        self.k = k
        self.b = b
        self.docs = []
        self.lemmatizer = lemmatizer
        if use_pretrained:
            self.docs = np.load(texts, allow_pickle=True)
            self.index = load_npz(index_file)
            self.tf_matrix = load_npz(tf_file)
            self.idf = np.load(idf_file, allow_pickle=True)
            self.vectorizer = CountVectorizer(tokenizer=self.lemmatizer)
            self.vectorizer.vocabulary_ = load(vocab_file)
        else:
            self.update_index(texts)
    
    def update_index(self, docs):
        self.docs = np.array(list(self.docs) + docs)
        self.vectorizer = CountVectorizer(tokenizer=self.lemmatizer)
        self.tf_matrix = self.vectorizer.fit_transform(self.docs)
        ## counting non-zero values in sparse matrix row
        ## using only sparse matrix operations:
        ## dense matrix would be too big - 300+ GB (vs. ~500MB in sparse format)
        df = self.tf_matrix.getnnz(axis=1)
        N = df.sum()
        idf_transform = np.vectorize(lambda x: (N-x+0.5)/(x+0.5))
        self.idf = np.log(idf_transform(df))
        self.set_hyperparams()
    
    def set_hyperparams(self, k=None, b=None):
        if k is None:
            k = self.k
        else:
            self.k = k
        if b is None:
            b = self.b
        else:
            self.b = b
        doc_lengths = np.array(self.tf_matrix.sum(axis=0))[0]
        avgdl = doc_lengths.mean()
        length_ratios = np.array([doc_lengths[col]/avgdl for col in self.tf_matrix.nonzero()[1]])
        idfs = np.array([self.idf[row] for row in self.tf_matrix.nonzero()[0]])
        new_data = idfs * self.tf_matrix.data * (k+1) / (self.tf_matrix.data + k*(1-b+b*length_ratios))
        self.index = csr_matrix((new_data, self.tf_matrix.indices, self.tf_matrix.indptr),
                                shape=self.tf_matrix.shape)
        self.index = self.index.transpose(copy=True)
    
    def search(self, query, return_similarities=False, return_indices=False, n_results=5):
        vector = self.vectorizer.transform([query])
        similarities = vector * self.index
        similarities = np.array(similarities.todense())[0]
        order = np.argsort(similarities)[::-1].tolist()[:n_results]
        
        if return_similarities:
            if return_indices:
                return similarities[order], order
            return similarities[order], self.docs[order]
        
        if return_indices:
            return order
        return self.docs[order]
    
    def multiple_search(self, query):
        vector = self.vectorizer.transform(query)
        return vector * self.index
    
    def dump(self, index_file, text_collection_file, tf_file, idf_file, vocab_file):
        save_npz(index_file, self.index)
        save_npz(tf_file, self.tf_matrix)
        self.idf.dump(idf_file)
        self.docs.dump(text_collection_file)
        dump(self.vectorizer.vocabulary_, vocab_file)


class TFSearch:
    def __init__(self, texts, lemmatizer,
                 use_pretrained=False, index_file=None,
                 vocab_file=None):
        self.lemmatize = lemmatizer
        self.vectorizer = CountVectorizer(tokenizer=self.lemmatize)
        if use_pretrained:
            self.docs = np.load(texts, allow_pickle=True)
            self.index = load_npz(index_file)
            self.vectorizer.vocabulary_ = load(vocab_file)
        else:
            self.index = normalize(self.vectorizer.fit_transform(texts))
            self.index = self.index.transpose(copy=True)
            self.docs = np.array(texts)
    
    def search(self, query, return_similarities=False, return_indices=False, n_results=5):
        vector = normalize(self.vectorizer.transform([query]))
        similarities = vector * self.index
        similarities = np.array(similarities.todense())[0]
        order = np.argsort(similarities)[::-1].tolist()[:n_results]
        
        if return_similarities:
            if return_indices:
                return similarities[order], order
            return similarities[order], self.docs[order]
        
        if return_indices:
            return order
        return self.docs[order]
    

    def dump(self, index_file, text_collection_file, vocab_file):
        save_npz(index_file, self.index)
        self.docs.dump(text_collection_file)
        dump(self.vectorizer.vocabulary_, vocab_file)

class IndexSearch:
    '''Base class for search models using word and sentence vectors'''
    def search(self, text, return_similarities=False, n_results=5):
        query = normalize(self.transform(text)[np.newaxis])
        sims = np.dot(query, self.index)
        top_texts = np.argsort(sims[0]).tolist()[::-1]
        top_texts = top_texts[:n_results]
        if return_similarities:
            return sims[0][top_texts], self.texts[top_texts]
        else:
            return self.texts[top_texts]
   
    def dump(self, index_file, text_collection_file):
        self.index.dump(index_file)
        self.texts.dump(text_collection_file)

class W2VSearch(IndexSearch):
    def __init__(self, model, texts, lemmatizer,
                use_pretrained=False, index_file=None):
        self.model = model
        self.lemmatize = lemmatizer
        if use_pretrained:
            if index_file:
                self.index = np.load(index_file, allow_pickle=True)
                if type(texts) == str:
                    self.texts = np.load(texts, allow_pickle=True)
                else:
                    self.texts = np.array(texts)
            else:
                raise ValueError("No index file provided")
        else:
            self.texts = np.array(texts)
            self.index = np.zeros(shape=(len(texts),model.vector_size))
            for text_id, text in enumerate(texts):
                if text:
                    self.index[text_id] = self.transform(text)
            self.texts = self.texts[~np.isnan(self.index).any(axis=1)]
            self.index = self.index[~np.isnan(self.index).any(axis=1)]
            self.index = normalize(self.index)
            self.index = self.index.transpose() 
    
    def transform(self, text):
        vecs = [self.model[word] for word in self.lemmatize(text) if word in self.model]
        return np.mean(vecs, axis=0)

class ElmoSearch(IndexSearch):
    def __init__(self, texts, elmo_path, lemmatizer, batch_size=200,
                use_pretrained=False, index_file=None):
        self.elmo_path = elmo_path
        self.lemmatize = lemmatizer
        tf.reset_default_graph()
        self.batcher, self.scids, self.esi = load_elmo_embeddings(elmo_path)
        if use_pretrained:
            if index_file:
                self.index = np.load(index_file, allow_pickle=True)
                if type(texts) == str:
                    self.texts = np.load(texts, allow_pickle=True)
                else:
                    self.texts = np.array(texts)
            else:
                raise ValueError("No index file provided")
        else:
            texts = sorted(texts, key=len, reverse=True)
            self.index = self.multiple_transform(texts, batch_size=batch_size)
            self.texts = np.array(texts)
            self.texts = self.texts[~np.isnan(self.index).any(axis=1)]
            self.index = self.index[~np.isnan(self.index).any(axis=1)]
            self.index = normalize(self.index)
            self.index = self.index.transpose()
    
    def multiple_transform(self, texts, batch_size=200):
        index = np.zeros(shape=(len(texts), self.esi['weighted_op'].shape[2].value))
        for i in range(0, len(texts), batch_size):
            j = min(i+batch_size, len(texts))
            text_batch = [self.lemmatize(text) for text in texts[i:j]]
            print("max sentence length - ", max([len(i) for i in text_batch]))
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                elmo_vectors = get_elmo_vectors(
                sess, text_batch, self.batcher, self.scids, self.esi)
            index[i:j] = np.mean(elmo_vectors, axis=1)
            del elmo_vectors
        return index
   
    def transform(self, text):
        return self.multiple_transform([text])[0]