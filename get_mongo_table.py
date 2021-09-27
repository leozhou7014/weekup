import pymongo

def get_mongo_table(mongo_db_host, mongo_db_name, mongo_db_table, condition, content):
    '''
    从mongo db数据库获取表
    :param mongo_db_host: mongodb地址
    :param mongo_db_name: mongodb数据库名称
    :param mongo_db_table: mongodb表名
    :param condition: 查询条件
    :param content: 查询内容
    :return: 拉取到的表（pymongo.cursor.Cursor形式）
    '''
    db = pymongo.MongoClient(mongo_db_host)[mongo_db_name]
    db_mongo_table = db[mongo_db_table].find(condition, content)    # 数据库中拉取帖子，pymongo.cursor.Cursor形式
    return db_mongo_table