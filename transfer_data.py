
import psycopg2
# def tf_to_wb_likes(data,conn,cursor):
#     """
#     use executemany to insert data into postgresql wb_like table
#
#     """
#     try:
#         data_list = list(data)
#         cursor.executemany("""INSERT INTO likes VALUES(
#                                      %(_id)s,
#                                      %(tweet_id)s,
#                                      %(user_id)s,
#                                      %(tweet_user_id)s,
#                                      %(nick_name)s,
#                                      %(created_at)s,
#                                      %(created_at_timestamp)s,
#                                      %(crawl_at)s,
#                                      %(crawl_at_timestamp)s);
#              """, ({**r} for r in data_list))
#     except(Exception,psycopg2.DatabaseError) as e:
#         return 1
#     conn.commit()


def tf_postgresql(conn,df,table_name):
    """
    pd.DataFrame to sql method insert DataFrame into postgresql's table


    conn : a connection with postgresql using sqlalchemy toolkit
    df : dataframe from read_mongodb
    table_name : the table we want to insert the data
    if_exists = 'append' : if the table is already exist, add data to the table
    chunksize : if we insert large data into database, we need to split it into small size
    """
    df.to_sql(table_name,con = conn, if_exists = 'append', index = False)













