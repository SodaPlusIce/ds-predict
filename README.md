back:
安装django，pymysql
建议conda环境下运行
/back路径下执行 `python manage.py runserver`

front:
/front目录下执行npm install
执行npm start启动项目

db:
docker pull mysql

图表展示

业务逻辑：
前端拿到上传的csv文件，解析成数组
以json格式传到后端
后端解析，然后在接口里跑模型
前端显示等待中
跑完模型把结果图返回给前
前显示出来

前端一个容器，通过status变量判断是现实加载组件还是图表组件
status 0未上传文件1训练中2图表和下载按钮