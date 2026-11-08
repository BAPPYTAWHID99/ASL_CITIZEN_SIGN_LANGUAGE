"""URL routes for the ASL translator app."""

from django.urls import path
from django.views.generic import TemplateView

from .views import predict


urlpatterns = [
    path("api/predict/", predict, name="predict"),
    path("", TemplateView.as_view(template_name="translator/index.html"), name="index"),
]
