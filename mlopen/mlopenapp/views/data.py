from datetime import datetime
from mlopenapp.forms import UploadFileForm
from django.views.generic.base import TemplateView
from django.views.generic.edit import FormView
from django.shortcuts import render, redirect
import mlopenapp.models as m


class DataView(TemplateView, FormView):
    template_name = "data.html"
    form_class = UploadFileForm
    success_url = '/data/'
    relative_url = "data"
    FILE_TYPES = ['csv', 'txt']

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["title"] = "Input File Upload"
        context["file_title"] = "input"
        context['template'] = "data.html"
        return context

    def post(self, request, *args, **kwargs):
        form = self.form_class(request.POST, request.FILES)
        print("FORM IS")
        print(form)
        if form.is_valid():
            temp = m.InputFile()
            temp.name = form.cleaned_data['name']
            temp.file = request.FILES['file']
            temp.created_at = datetime.now()
            temp.save()
            return redirect(self.success_url)
        else:
            return render(request, self.template_name, {'form': form})

    def form_valid(self, form):
        return super().form_valid(form)
