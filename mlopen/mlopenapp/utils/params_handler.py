from django import forms
from ..constants import PARAM_MAP
from ..models import InputFile


def field_params(type, params):
    if type in ["integer", "float"]:
        blank = True
        default = 0.0
        if "blank" in params:
            blank = params["blank"]
        if "default" in params:
            default = params["default"]
        return PARAM_MAP[type](required=blank, initial=default)
    elif type == "choice":
        if "choices" in params:
            choices = []
            for choice in params["choices"]:
                choices.append((choice, choice))
            return PARAM_MAP[type](choices=choices)
    elif type == "file":
        return PARAM_MAP[type](queryset=InputFile.objects.all(), required=False)
    elif type == "upload":
        return PARAM_MAP[type](required=False)


def get_params_form(params):
    paramform = forms.Form()
    for name, type in params.items():
        if isinstance(type, tuple):
            try:
                if len(type) > 1:
                    paramform.fields[name] = field_params(type[0], type[1])
                else:
                    paramform.fields[name] = field_params(type[0], {})
            except Exception as e:
                print(e)
        else:
            try:
                if type == "file":
                    paramform.fields[name] = PARAM_MAP[type](queryset=InputFile.objects.all(), required=False)
                else:
                    paramform.fields[name] = PARAM_MAP[type]()
            except Exception as e:
                print(e)
    return paramform
