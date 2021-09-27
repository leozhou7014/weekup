import pymongo
import psycopg2
import sqlalchemy
from sqlalchemy import exc

def conn_mongodb(host,port, user, password, db_name):
    """
    make a connection with database in mongodb

    pymongo.MongoClient: create a new connection with mongodb

    connection error return 1
    """
    try:
        if user and password:
            url = 'mongodb://%s:%s@%s:%s/%s' % (user, password, host, port, db_name)
            client = pymongo.MongoClient(url)
            conn_mon_db = client[db_name]
        else:
            client  = pymongo.MongoClient(host,port)
            conn_mon_db = client[db_name]

    except (Exception,pymongo.errors.ConnectionFailure) as e:
        return 1
    return conn_mon_db

def close_mongodb(conn_mon_db):
    """
    close mongodb
    """
    conn_mon_db.close()



def conn_postgresql(db_name, user_name, password_num, host_num, port_num):
    """
    use psycopg2 toolkit to connect with postgresql database

    psycopg2.connect() : create a new connection with postgresql
    conn.cursor() : create a new cursor

    """
    conn = psycopg2.connect(database=db_name, user=user_name, password=password_num, host=host_num, port=port_num)
    cursor = conn.cursor()
    return conn, cursor


def close_postgresql(conn,cursor):
    """
    cursor.close(): Close cursor
    conn.close() : close database
    """
    cursor.close()
    conn.close()

def conn_sqlalchemy(user,password,host,port,db_name):
    """
    make a connection to postgresql use sqlalchemy toolkit


    """
    try:
        url = 'postgresql://%s:%s@%s:%s/%s' %(user,password,host,port,db_name)
        engine = sqlalchemy.create_engine(url)

        conn_sql = engine.connect()

    except (Exception, exc.DisconnectionError) as e:
        return 1

    return conn_sql

def close_sqlalchemy(conn_sql):
    """
    close connection
    """
    conn_sql.close()


