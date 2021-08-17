from django.urls import path
from django.conf.urls import url

from . import views

app_name = "mlopenapp"
urlpatterns = [
    url(
        r'^$',
        views.IndexView.as_view(),
        name="base"
    ),
    url(
        r'^import/$',
        views.ImportView.as_view(),
        name="import"
    ),
    url(
            r'^data/$',
            views.DataView.as_view(),
            name="data"
        ),
    url(
            r'^pipelines/$',
            views.PipelineView.as_view(),
            name="pipelines"
        ),
]
