from flask import Flask, request, render_template, redirect

from search_models import TFSearch, BMVectorSearch, W2VSearch, ElmoSearch, load_json, Lemmatizer
import os
import logging
import logging.config

from html import unescape

from gensim.models import KeyedVectors

logging.config.fileConfig("logconfig")
search_config = load_json("search_config.json")

def select_seq(model_name):
    seq = ['', '', '', '']
    if model_name == 'tf':
        seq[0] == 'selected'
    elif model_name == 'bm25':
        seq[1] == 'selected'
    elif model_name == 'ft':
        seq[2] == 'selected'
    elif model_name == 'elmo':
        seq[3] == 'selected'
    return seq

def search_with_model(model, text, n_results=100):
    if not text.strip():
        raise Exception("Empty search query")
    n_results = int(n_results)
    kwargs = {'use_pretrained': True}
    lemmatizer = Lemmatizer()
    kwargs['lemmatizer'] = lemmatizer
    if model == 'tf':
        path = search_config['tf_data_path']
        kwargs['texts'] = os.path.join(path, 'texts.npy')
        kwargs['index_file'] = os.path.join(path, 'index.npz')
        kwargs['vocab_file'] = os.path.join(path, 'vocab.pkl')
        search_model = TFSearch(**kwargs)
        ##print(search_model.vectorizer.transform([text]).shape)
        ##print(search_model.index.shape)
    elif model == 'bm25':
        path = search_config['bm25_data_path']
        kwargs['texts'] = os.path.join(path, 'texts.npy')
        kwargs['index_file'] = os.path.join(path, 'index.npz')
        kwargs['vocab_file'] = os.path.join(path, 'vocab.pkl')
        kwargs['tf_file'] = os.path.join(path, 'tf_matrix.npz')
        kwargs['idf_file'] = os.path.join(path, 'idf_matrix.npy')
        search_model = BMVectorSearch(**kwargs)
    elif model == 'ft':
        path = search_config['ft_data_path']
        kwargs['model'] = KeyedVectors.load(search_config['ft_path'])
        kwargs['texts'] = os.path.join(path, 'texts.npy')
        kwargs['index_file'] = os.path.join(path, 'index.npy')
        search_model = W2VSearch(**kwargs)
    elif model == 'elmo':
        path = search_config['elmo_data_path']
        kwargs['elmo_path'] = search_config['elmo_path']
        kwargs['texts'] = os.path.join(path, 'texts.npy')
        kwargs['index_file'] = os.path.join(path, 'index.npy')
        search_model = ElmoSearch(**kwargs)
    else:
        raise Exception
    return search_model.search(text, return_similarities=True, n_results=n_results)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def display_index():
    logger = logging.getLogger("myApp")
    query = 'пример текста для поиска'
    model_name = "tf"
    n_res = 10
    #print(logger)
    if request.args:
        param_string = ','.join([f'"{k}":"{unescape(v)}"' for k,v in request.args.items()])
        print(param_string)
        logger.info('search with params: '+param_string)
        try:
            results = [(float(i),j) for i,j in zip(*search_with_model(**request.args))]
            logger.info("request ended with success")
            query = request.args['text']
            n_res = request.args['n_results']
            model_name = request.args['model']
        except Exception as e:
            logger.exception(e)
            results = [('', "Invalid search query")]
    else:
        logger.info("start page opened")
        results = [('', "Empty search query")]
    return render_template('index.html', results=results, query=query, n_res=n_res, model_name=model_name)

if __name__ == '__main__':
    logger = logging.getLogger("myApp")
    logger.info("service started")
    app.run()
