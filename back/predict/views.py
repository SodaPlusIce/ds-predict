from django.http import HttpResponse

def predict(request):
    if request.method=='POST':
        # 拿到用户上传的csv里的数据
        print(request.body)
        # 待集成算法
        # xxx
        # 将matplotlib画的图base64编码发送给前端
        return HttpResponse("photo")