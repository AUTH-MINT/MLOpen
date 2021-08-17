# mlopen

mlopen is a django web application for uploading and running machine learning pipelines.

## Installation

TODO

## Usage
### Creating pipelines to be trained and/or run in the mlopen platform

The mlopen platform does not require much from a pipeline developer in order for them to be able to import their pipelines. Almost any python3 machine learning pipeline can easily be converted to be used by mlopen in a few easy steps.

Each pipeline consists of the following elements:
- The **control** file, which is a main python file that contains the base code to train and run the pipeline. (**mandatory**)
- Support files to be used by the main python file (**optional**)

The control file must always implement **2 base functions**, `train` and `run_pipeline`. Even if your pipeline does not include training a model, it is still best practice to create a function and have it use `pass` or log a message that this pipeline does not need/provide training.

#### The `train` function

The `train` function takes two optional arguments, `input` and `params`:

```python
def train(inpt, params=None)
```

The train function can use the `input` file (which will be chosen in the platform before training the pipeline by the user) to train the model and save it into the db.

Once the model and any additional files (like vocabularies, vectorizers and so on) are created, mlopen provides the function `save_pipeline` to save them to the db, in order for them to be retrieved every time a user wants to run the pipeline. This function exists in the `utils` module of mlopen, so to use it it's enough to import it in your control file:

```python
from mlopenapp.utils import io_handler as io
```

and then use it to save your files, like this:
```python
io.save_pipeline(models, args, os.path.basename(__file__))
```
where models is a list contained the trained model objects (usually just one) and args is a list containing a list of other objects needed for running the pipeline.

So, tl;dr: `train` uses the input and any other items defined by the creator of the pipeline to train a model and create any support objects, and then saves them using the `save_pipeline` function.

#### The `run_pipeline` function

The `run_pipeline` function is called when a user attempts to run a pipeline on their own data to acquire results. The pipeline should already be trained (or lack the need for training, as is the case with the knn algorithm for example). 

```python
def run_pipeline(input, model, args, params=None)
```
Here, the `model` and `args` (if present) arguments are automatically provided by the mlopen platform, based on the corresponding items saved on the database during the training phase. The `args` argument is in the form of a dictionary, with the object names as keys (the names used to save the objects during the training phase) and the objects themselves as values. If, for example, a user saves a tf-idf vectorizer inside the `train` function as
```python
(tfidf, "tfidf_vect")
```
it can be retrieved in the `run_pipelines` function like this:
```python
args['tfidf_vect']
``` 
The `run_pipeline` function uses the `input` argument, provided by the user via the mlopen interface, to run the pipeline and get results.

The function returns a dictionary which must be named `preds`: this dictionary contains 4 optional keys:`data`, `columns`, `graphs` and `text`. If the creator of the pipelines only wants to display some graphs, for example, they can skip creating all other keys and just return a `preds` dictionary with a list of graphs as the value to the dicts sole key, `graphs`.

`data`: a list of lists or tuples representing the rows and columns of a table

`columns`: the names of the columns for the table provided via `data`

`graphs`: a list of plotly graphs to be displayed by the platform. The mlopen platform provides convenience functions for scatter and pie charts via the `plotter` module in `utils`, but users can also create their own plotly graphs, turn them to html using `plotly.offline.plot` with the `output_type='div'` argument and add them to `preds['graphs'].

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
