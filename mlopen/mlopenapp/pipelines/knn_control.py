import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from mlopenapp.utils import io_handler as io


def get_params(run=True):
    if run:
        params = {"k": ("integer", {"default": 5}), "data": ("file"), "column_names": ("string")}
        return params
    else:
        return ""


def train(inpt, params):
    raise Exception("knn has no separate training phase. Run the pipeline with the training "
                    + "set in the 'Select data' field.")


def run_pipeline(input, model, args, params=None):
    input.open("r")
    # Assign colum names to the dataset
    if "column_names" in params and len(params['column_names']) > 0:
        names = params['column_names'].split(",")
    else:
        names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

    # Read dataset to pandas dataframe
    dataset = pd.read_csv(input, names=names)

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 4].values

    # Train - Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # Normalize attributes
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = KNeighborsClassifier(n_neighbors=int(params.get('k', 5)))
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    preds = {}
    data = [[str(x[0]), str(x[1]), str(x[2]), str(x[3]), y, z] for x, y, z in zip(X_test, y_pred, y_test)]
    preds['data'] = data
    preds['columns'] = list(dataset.columns.values)[:-1] + ['Predicted Class', 'Actual Class']
    preds['graphs'] = None
    preds['text'] = """
    <div>This is a standard implementation of the <b>knn algorithm</b>.</div>
    <div>From wikipedia:</div>
    <p>In <a href="https:/www.wikipedia.com/wiki/Statistics" title="Statistics">statistics</a>,
    the <b><i>k</i>-nearest neighbors algorithm</b> (<b><i>k</i>-NN</b>)
    is a <a href="https:/www.wikipedia.com/wiki/Non-parametric_statistics" class="mw-redirect" 
    title="Non-parametric statistics">non-parametric</a> <a href="https:/www.wikipedia.com/wiki/Classification" 
    title="Classification">classification</a> method first developed by 
    <a href="https:/www.wikipedia.com/wiki/Evelyn_Fix" title="Evelyn Fix">Evelyn Fix</a> and 
    <a href="https:/www.wikipedia.com/wiki/Joseph_Lawson_Hodges_Jr." title="Joseph Lawson Hodges Jr.">Joseph Hodges</a>
    in 1951,<sup id="cite_ref-1" class="reference"><a href="#cite_note-1">&#91;1&#93;
    </a></sup> and later expanded by <a href="https:/www.wikipedia.com/wiki/Thomas_M._Cover" title="Thomas M. Cover">
    Thomas Cover</a>. It is used for 
    <a href="https:/www.wikipedia.com/wiki/Statistical_classification" title="Statistical classification">
    classification</a> and <a href="https:/www.wikipedia.com/wiki/Regression_analysis" 
    title="Regression analysis">regression</a>. In both cases, the input consists of the 
    <i>k</i> closest training examples in <a href="https:/www.wikipedia.com/wiki/Data_set"
    title="Data set">data set</a>. The output depends on whether <i>k</i>-NN 
    is used for classification or regression.</p>
    """
    return preds

