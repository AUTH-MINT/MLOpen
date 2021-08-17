import os
import pandas

from mlopenapp.pipelines import vectorization as vct,\
    text_preprocessing as tpp, \
    logistic_regression as lr, \
    metrics as mtr
from mlopenapp.utils import io_handler as io
from mlopenapp.utils import plotter, data_handler

# These will be replaced by user input
train_paths = [
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir,
                     'data/user_data/train/pos/'),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir,
                     'data/user_data/train/neg/')
]

train_sentiments = [1, 0]

test_paths = [
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir,
                     'data/user_data/test/pos/'),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir,
                     'data/user_data/test/neg/')
]

test_sentiments = [1, 0]


def get_params(run=True):
    if not run:
        params = {"train - pos samples": ("upload"), "train - neg samples": ("upload"),
                  "test - pos samples": ("upload"), "test - neg samples": ("upload"), }
        return params
    else:
        return ""


def train(inpt, params=None):
    """
    Creates a model based on a train and a test dataframes, and calculates model metrics
    """
    print("Preparing Data. . .")
    df_test = pandas.DataFrame()
    df_train = pandas.DataFrame()
    for key, val in params.items():
        if "test" in str(key):
            df_test = pandas.concat(df_test, data_handler.read_from_file(val))
        if "train" in str(key):
            df_train = pandas.concat(df_train, data_handler.read_from_file(val))
    tfidf = {}
    vector = {}
    corpus = df_train["text"].tolist()
    tfidf["text"] = vct.fit_tf_idf(corpus)
    vector["text"] = vct.tf_idf(corpus, tfidf["text"])
    # TODO: add pos/neg frequencies method
    # freqs = frq.build_freqs(corpus, df['sentiment'].tolist())
    # x_pn = [frq.statement_to_freq(txt, freqs) for txt in corpus]
    # x_posneg = frq.get_posneg(corpus, freqs)
    s_a_model = lr.log_reg(vector["text"], df_train['sentiment'].tolist())
    test_corpus = df_test["text"].tolist()
    test_sentiment = df_test['sentiment'].tolist()
    mtr.get_model_metrics(tfidf["text"], s_a_model, test_corpus, test_sentiment, True)
    models = [(s_a_model, "logreg_model")]
    args = [(tfidf["text"], "tfidf_vect")]
    io.save_pipeline(models, args, os.path.basename(__file__))
    return tfidf["text"], s_a_model


def run_pipeline(input, model, args, params=None):
    """
    Predicts the sentiment of a list of text statements
    """
    preds = {'data': [], 'columns': [], 'graphs': None}
    pos = 0
    input.open("r")
    for statement in input.readlines():
        temp = predict_text(statement, args['tfidf_vect'], model, False)
        preds['data'].append([str(temp[0]), str(temp[1])])
        if temp[1] == 1:
            pos += 1
    x = [x for x in range(20)]
    y = [x for x in range(20)]
    preds['columns'] = ['Statement', 'Sentiment']
    div = plotter.pie_plot_from_lists(values=[pos, len(preds['data']) - pos], labels=['Positive', 'Negative'],
                 title="Number of Positive and Negative Reviews")
    preds['graphs'] = [div]
    return preds


def predict_text(text, tfidf, model, processed=False):
    """
    Predicts the sentiment of a single text statement
    """
    original = text
    if not processed:
        text = tpp.process_text(text)
    transformed_text = tfidf.transform([text])
    prediction = model.predict(transformed_text)
    return (original, prediction[0])


