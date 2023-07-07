from django.http import HttpResponse
import pymysql

def homePageView(request):
    # 打开数据库连接
    db = pymysql.connect(host='localhost',
                         user='root',
                         password='123456',
                         database='ds-predict')
     
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
     
    # 使用 execute()  方法执行 SQL 查询 
    sql = 'select * from user where username = 111'
    cursor.execute(sql)
     
    # 使用 fetchone() 方法获取单条数据.
    data = cursor.fetchone()
     
    print(data)
     
    # 关闭数据库连接
    db.close()
    return HttpResponse("Hello, World!")