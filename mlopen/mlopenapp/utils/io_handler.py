import sys
import os
import pickle
import datetime
from ..models import models
from django.core.files import File
from .. import constants


def save(arg_object, name, save_to_db=False, type=None):
    try:
        output = open(constants.FILE_DIRS[type] + '/' + name + '.pkl', 'wb')
        pickle.dump(arg_object, output, pickle.HIGHEST_PROTOCOL)
        output.close()
        if save_to_db:
            output = open(constants.FILE_DIRS[type] + '/' + name + '.pkl', 'rb')
            filefield, _ = constants.FILE_TYPES[type].objects.get_or_create(
                name=name,
                defaults={
                    'created_at': datetime.datetime.now(),
                    'updated_at': datetime.datetime.now(),
                    'file': File(output)
                }
            )
            filefield.save()
            print(filefield)
            output.close()
            return filefield
        return True
    except Exception as e:
        print(e)
        return False


def load(name, type):
    try:
        with open(os.path.join(constants.FILE_DIRS[type], name), 'rb') as input:
            ret = pickle.load(input)
        return ret
    except Exception as e:
        print(e)
        return False


def save_pipeline(models, args, name):
    pip_models = []
    pip_args = []
    for model in models:
        temp = save(model[0], model[1], True, 'model')
        if type(temp) == bool:
            return False
        pip_models.append(temp)
    for arg in args:
        temp = save(arg[0], arg[1], True, 'arg')
        if type(temp) == bool:
            return False
        pip_args.append(temp)
    name = name[:-3] if name.endswith(".py") else name
    pipeline, _ = constants.FILE_TYPES['pipeline'].objects.get_or_create(
        control=name,
        defaults={
            'name': name,
            'created_at': datetime.datetime.now(),
            'updated_at': datetime.datetime.now()
        }
    )
    pipeline.save()
    for model in pip_models:
        pipeline.ml_models.add(model)
    for arg in pip_args:
        pipeline.ml_args.add(arg)
    pipeline.save()


def get_pipeline_list():
    pipeline_list = []
    for filename in os.listdir(constants.CONTROL_DIR):
        if filename.endswith("control.py"):
            pipeline_list.append(filename[:-3])
    return pipeline_list


def save_pipeline_file(f):
    with open(os.path.join(constants.CONTROL_DIR, f.name), 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


def load_pipeline(pipeline):
    try:
        model_qs = pipeline.get_models()
        args_qs = pipeline.get_args()
        model = None
        args = {}
        for m in model_qs:
            model = pickle.load(m.file.open('rb'))
        for ar in args_qs:
            args[ar.name] = pickle.load(ar.file.open('rb'))
        return model, args
    except Exception as e:
        print(e)
        return False


def save_pipeline_files(pipeline, files):
    if not files:
        return True
    if pipeline.endswith(".py"):
        pipeline = pipeline[:-3]
    dir_name = os.path.join(constants.CONTROL_DIR, pipeline)
    try:
        # Create target Directory
        os.mkdir(dir_name)
    except FileExistsError:
        return False, "Directory already exists."
    for f in files:
        with open(os.path.join(dir_name, f.name), 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)
    return True, "Directory and files successfully created."
