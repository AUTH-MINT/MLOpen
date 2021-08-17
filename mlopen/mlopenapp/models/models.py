import os
import django
from django.db import models
from .storages import LOCAL_STORAGE
from django.core.files.storage import FileSystemStorage
from django.conf import settings

FILE_DIRS = {
    "model": os.path.join(LOCAL_STORAGE, 'models'),
    "arg": os.path.join(LOCAL_STORAGE, 'args'),
    "pipeline": os.path.join(LOCAL_STORAGE, 'pipelines')
}


class MLModel(models.Model):
    """
    Model for machine learning models
    """
    name = models.CharField(max_length=200, db_index=True)
    created_at = models.DateTimeField(null=True)
    updated_at = models.DateTimeField(null=True)
    file = models.FileField(storage=FileSystemStorage(location=FILE_DIRS['model']))


class MLArgs(models.Model):
    """
    Model for machine learning helper objects (such as vectorizers)
    """
    name = models.CharField(max_length=200, db_index=True)
    created_at = models.DateTimeField(null=True)
    updated_at = models.DateTimeField(null=True)
    file = models.FileField(storage=FileSystemStorage(location=FILE_DIRS['arg']))


class MLPipeline(models.Model):
    """
    Model for machine learning pipelines.

    Includes all files and data needed for the application to assemble a pipeline.
    A pipeline uses all related models and objects, accepts user data as input and
    returns the results.
    """
    name = models.CharField(max_length=200, db_index=True)
    control = models.CharField(max_length=200, null=False, default="control")
    created_at = models.DateTimeField(null=True)
    updated_at = models.DateTimeField(null=True)
    ml_models = models.ManyToManyField(MLModel)
    ml_args = models.ManyToManyField(MLArgs)

    def get_models(self):
        return self.ml_models.all()

    def get_args(self):
        return self.ml_args.all()

    def __str__(self):
        return self.control
