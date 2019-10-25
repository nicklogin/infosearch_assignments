from search_models import extract_texts, BMVectorSearch, TFSearch, Lemmatizer
import pandas as pd
from time import localtime

time_start = localtime()
    
data_path = "C:\\Users\\k1l77\\infosearch\\infosearch_assignments\\data\\quora_question_pairs_rus.csv"

texts = extract_texts(pd.read_csv(data_path))

lemmatizer =  Lemmatizer()

tf_model = TFSearch(texts, lemmatizer)

print('TF Search Model ready, saving...')

tf_model.dump(index_file='./tf_data/index.npz',
text_collection_file='./tf_data/texts.npy',
vocab_file='./tf_data/vocab.json')

print('Successfully saved TF Search Model')

bm_model = BMVectorSearch(texts, lemmatizer)

print('BM25 Model ready, saving...')

bm_model.dump(index_file='./bm25_data/index.npz',
text_collection_file='./bm25_data/texts.npy',
tf_file='./bm25_data/tf_matrix.npz',
idf_file='./bm25_data/idf_matrix.npz',
vocab_file='./bm25_data/vocab.json')

print('Successfully saved BM25 Model')

print("Time consumed:", localtime() - time_start)