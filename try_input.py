
from connect_database import conn_mongodb,conn_postgresql,conn_sqlalchemy
from read_data import read_mongodata, read_given_time
from transfer_data import tf_postgresql
import datetime


start_time = datetime.datetime(2021, 6, 16, 17, 14, 42, 966000)
end_time = datetime.datetime(2021, 6, 16, 17, 14, 42, 991000)
mongo_host = "127.0.0.1"
mongo_port = 27017
mongo_user = ""
mongo_password = ""
mongo_dbname = "weibo"
mongo_tablename = "likes"

post_host = "localhost"
post_port = "5432"
post_user = "postgres"
post_password = "wsad950428"
post_db_name = "QIANYI"
post_table_name = "l"

conn_mon_db = conn_mongodb(mongo_host,mongo_port,mongo_user,mongo_password,mongo_dbname)

df = read_given_time(conn_mon_db,mongo_tablename,start_time,end_time)



conn_sql = conn_sqlalchemy(post_user,post_password,post_host,post_port,post_db_name)


tf_postgresql(conn_sql,df,post_table_name)










