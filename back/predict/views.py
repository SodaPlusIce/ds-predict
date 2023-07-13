from django.http import HttpResponse
import matplotlib
matplotlib.use("agg")  # 设置 Matplotlib 后端为 agg
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def predict(request):
    if request.method == "POST":
        # 拿到用户上传的csv里的数据
        print(request.body)
        # 待集成算法
        # xxx
        # 创建一个 Matplotlib 图形
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.plot([1, 2, 3], [4, 5, 6])
        ax.set_xlabel("X Label")
        ax.set_ylabel("Y Label")
        ax.set_title("Title")
        canvas = FigureCanvas(fig)

        # 将 Matplotlib 图形保存为 PNG 格式的二进制数据
        buffer = io.BytesIO()
        canvas.print_png(buffer)
        buffer.seek(0)
        image_data = buffer.getvalue()
        buffer.close()

        # 将 PNG 图像数据转换为 Base64 编码的字符串
        base64_data = base64.b64encode(image_data).decode("latin1")

        # 返回 Base64 编码的字符串
        return HttpResponse(base64_data)
        # return HttpResponse("photo")
