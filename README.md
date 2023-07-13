back:
- 建议conda环境下运行
- 安装django，pymysql
- /back路径下执行 `python manage.py runserver`启动项目
- 算法结合需在back/predict/views.py中code
- urls配置路由，back/back/urls.py配置总路由规则

front:
- /front目录下执行npm install
- 执行npm start启动项目

db:
- docker里开一个mysql
- db名叫ds-predict
- 表自行设计

业务逻辑：
- 前端拿到上传的csv文件，解析成数组
- 调用/predict接口，数据给到后端
- 后端拿到数据，然后跑模型
- 前端显示等待中
- 跑完模型把结果图通过base64编码返回给前
- 前端显示出来

tips：
- 前端是clone的一个模版改的
- 核心文件在front/front-react/src/pages/home/Workspace.tsx中

todo:
- [ ] 修改各种图标
- [ ] 修改登录时的文字显示
- [ ] 集成算法
- [ ] 添加下载模型功能