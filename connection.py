import sqlalchemy
import psycopg2
from sqlalchemy import exc


def conn_sqlalchemy(user, password, host, port, db_name):
    """
    make a connection to postgresql use sqlalchemy toolkit

    """
    try:
        url = 'postgresql://%s:%s@%s:%s/%s' % (user, password, host, port, db_name)
        engine = sqlalchemy.create_engine(url)



    except (Exception, exc.DisconnectionError) as e:
        return 1

    return engine






