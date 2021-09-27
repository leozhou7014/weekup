# --*-- coding: utf-8 --*--

import pandas as pd
import pymongo
from timer import func_timer
import threading
import queue as Queue

def get_liveId(engine, output=False):
    '''
    获取非水军ID
    :param engine: 数据库
    :return: 非水军ID: DataFrame
    '''
    sql = 'SELECT _id FROM public."2th100w_water_remove" where not is_water'
    id_series = pd.read_sql(sql, engine)
    id_series = id_series.astype(str)
    if output:
        id_series.to_csv('微博去水用户ID.csv', encoding='utf-8-sig', index=False)
    return id_series

def get_mongoURL(file='D:\jzp\数据库地址.txt', i=1):
    '''
    从数据库地址文件中获取mongourl地址
    :param file: 存放MongoDB地址文件路径
    :param i: 地址在文件中的行数（从0开始）
    :return: url: str
    '''
    f = open(file, 'r', encoding='utf-8')
    url_str = f.read()
    f.close()
    url = url_str.split('\n')[i]
    return url

##　多线程
class myThread(threading.Thread):
    def __init__(self, name, q, table, Id_list, content, size=1000, attr='user_id', condition='$in'):
        threading.Thread.__init__(self)
        self.name = name  # 线程名称
        self.q = q  # 从队列中获取的内容
        self.size = size
        self.attr = attr
        self.condition = condition
        self.content = content
        self.Id_list = Id_list
        self.table = table

    def run(self):   # 改写run方法
        while True:
            try:
                self.get_tweet()
            except:
                break

    def get_tweet(self):
        global tweet_Df
        i = self.q.get(timeout=2)
        try:
            data = pd.DataFrame(self.table.find({self.attr: {self.condition: self.Id_list[i * self.size: (i + 1) * self.size]}}, self.content))
            # 如果tweet_Df不存在则创建
            try:
                tweet_Df = pd.concat([tweet_Df, data], axis=0)
            except:
                tweet_Df = data
        except:
            print(self.q.qsize(), self.name, url, 'error')


@func_timer
def main(Id_list, content, table, attr='user_id', condition='$in', thread_no=10, batch_no=10, batch_size=1000):
    '''
    多线程获取mongodb数据
    :param Id_list: ID参数列表
    :param content: 获取的内容: dict
    :param table: mongoDB 数据表
    :param attr: 字段名
    :param condition: 字段条件: '$in'
    :param thread_no: 线程数
    :param batch_no: 批次数
    :param batch_size: 批次大小
    :return: DataFrame
    '''
    global tweet_Df
    thread_list = [str(i) for i in range(thread_no)]
    workQueue = Queue.Queue(10)
    threads = []
    for t_name in thread_list:
        thread = myThread(name=t_name, q=workQueue, table=table, attr=attr, condition=condition, Id_list=Id_list,
                          content=content, size=batch_size)
        thread.start()
        threads.append(thread)

    for i in range(batch_no):   # 放入队列
        workQueue.put(i)

    for t in threads:
        t.join()

    df = tweet_Df.copy()
    del tweet_Df    # 释放变量，防止冗余

    return df

if __name__ == '__main__':
    Id_series = pd.read_csv('D:\jzp\文本聚类\微博去水用户ID.csv', encoding='utf-8-sig')
    url = get_mongoURL()
    Id_series['_id'] = Id_series['_id'].astype(str)
    Id_list = list(Id_series['_id'])
    db_weibo = pymongo.MongoClient(url)['weibo']
    content = {'user_id':1, 'content':1}
    tweet_Df = main(Id_list, content, db_weibo, batch_size=10)
    # tweet_Df.to_csv('tweet_Df.csv', encoding='utf-8-sig')