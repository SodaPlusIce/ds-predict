from django.urls import path

from .views import predict,train,download

urlpatterns = [
    path("predict/", predict, name="predict"),
    path("train/", train, name="train"),
    path("download/", download, name="download"),
]