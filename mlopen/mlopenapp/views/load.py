from datetime import datetime
from django.views.generic.base import TemplateView
from django.views.generic.edit import FormView
from django.shortcuts import render, redirect
from mlopenapp.forms import ImportPipelineForm
import mlopenapp.models as m

from ..utils import io_handler as io


class ImportView(TemplateView, FormView):
    template_name = "data.html"
    form_class = ImportPipelineForm
    success_url = '/import/'
    fail_url = '/import_fail/'
    relative_url = "import"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["title"] = "Upload Pipeline Files and Support Files"
        context["file_title"] = "control"
        context['template'] = "import.html"
        return context

    def post(self, request, *args, **kwargs):
        form = self.form_class(request.POST, request.FILES)
        if form.is_valid():
            try:
                pipeline = request.FILES['file'].name
                if not pipeline.endswith("control.py"):
                    return redirect(self.fail_url)
                files = request.FILES.getlist('support_files')
                io.save_pipeline_files(pipeline, files)
                io.save_pipeline_file(request.FILES['file'])

                temp = m.MLPipeline()
                temp.control = pipeline[:-3]
                temp.created_at = datetime.now()
                temp.save()
            except:
                return redirect(self.fail_url)
            return redirect(self.success_url)
        else:
            return render(request, self.template_name, {'form': form})

    def form_valid(self, form):
        return super().form_valid(form)
