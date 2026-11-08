"""Root URL configuration for the ASL application."""

from django.contrib import admin
from django.urls import include, path


urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("translator.urls")),
]
