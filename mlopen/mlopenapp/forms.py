from django import forms
from .utils import io_handler as io
from .models import MLPipeline, InputFile
from . import constants


class UploadFileForm(forms.Form):
    name = forms.CharField(max_length=50)
    file = forms.FileField()


class ImportPipelineForm(forms.Form):
    file = forms.FileField()
    support_files = forms.FileField(required=False,
                                    widget=forms.ClearableFileInput(
                                        attrs={'multiple': True, 'value': "YOLO"}))


class PipelineSelectForm(forms.Form):
    type = forms.CharField(required=False)
    pipelines = forms.ModelChoiceField(queryset=MLPipeline.objects.all())
    input = forms.ModelChoiceField(queryset=InputFile.objects.all(), required=False)


class UploadForm(forms.ModelForm):
    class Meta:
        model = InputFile
        fields = [
        'name',
        'created_at',
        'file'
        ]
