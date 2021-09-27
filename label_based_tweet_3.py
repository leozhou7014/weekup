import numpy as np
import pandas as pd
import time
import sqlalchemy
import pymongo
from tqdm import tqdm
from sqlalchemy import exc
import warnings
import sys
warnings.filterwarnings('ignore')
sys.path.append('D:/datasets/qianyi/common')
import connect_database
from connect_database import conn_postgresql
import clean_seg
from clean_seg import clean_text

def get_dewater_user(post_db_table_name, post_host, post_port, post_user, post_password, post_db_name):
    '''
    获取非水军用户
    :param post_db_table_name: pgsql的表名
    :param post_host: 服务器地址
    :param post_port: 请求端口
    :param post_user: 用户名
    :param post_password: 密码
    :param post_db_name:数据库名称
    :return: 非水军用户的用户ID(DataFrame形式)
    '''
    engine = conn_postgresql(post_db_name, post_user, post_password, post_host, post_port)[0]  # 连接数据库
    sql = 'select _id from "{table_name}" where not is_water'.format(table_name=post_db_table_name)  # 获取非水军的用户ID
    df_dewater = pd.read_sql(sql, engine).astype(str)
    return df_dewater

def get_mongo_tweet(mongo_db_host, mongo_db_name, mongo_db_table, condition, content):
    '''
    从mongo db数据库获取用户的帖子
    :param df_dewater: 非水军用户的用户ID
    :param mongo_db_host: mongodb地址
    :param mongo_db_name: mongodb数据库名称
    :param mongo_db_table: mongodb表名
    :param condition: 查询条件
    :param content: 查询内容
    :return: 拉取到的用户帖子（pymongo.cursor.Cursor形式）
    '''
    db = pymongo.MongoClient(mongo_db_host)[mongo_db_name]
    db_tweet = db[mongo_db_table].find(condition, content)  # 数据库中拉取帖子，pymongo.cursor.Cursor形式
    return db_tweet

def tweet_clean(df_dewater_tweet):
    '''
    对提取到的帖子进行清洗
    :param df_dewater_tweet: 非水军用户的帖子
    :return: 清洗后的非水军用户帖子（DataFrame形式）
    '''
    df_clean = clean_text(df_dewater_tweet)
    df_clean['content_clean_length'] = df_clean['content_clean'].apply(lambda x: len(x))
    df_select = df_clean[(df_clean['content_clean_length'] > 1)]  # 筛选掉清洗后帖子长度小于等于1的帖子
    df_select = df_select.dropna(axis=0, subset=['content'])  # 删除缺失值
    df_select = df_select.reset_index(drop=True)
    return df_select

def read_relate_keywords(kind_file_path):
    '''
    读取关键词文件
    :param kind_file_path: 关键词文件所在的路径
    :return: 关键词列表
    '''
    df_kind = pd.read_csv(kind_file_path, header=None)  # 获得关键词库
    df_kind.columns = ['kind']  # 列名命名
    kind_list = list(df_kind['kind'])  # 转换成列表形式
    return kind_list

def tweet_select_by_keyword(df_dewater_clean,kind_list,param1=True):
    '''
    通过关键词匹配用户帖子
    :param df_dewater_clean: 清洗后的非水军帖子
    :param kind_list: 关键词对应的列表
    :param param1: True为匹配，False为筛除
    :return:
    '''
    df_relate = df_dewater_clean[df_dewater_clean.content_clean.str.contains(kind_list[0])]
    kind_length = len(kind_list)
    for i in tqdm(range(1, kind_length)):
        if param1:
            df_layout = df_dewater_clean[df_dewater_clean.content_clean.str.contains(kind_list[i])]
        else:
            df_layout = df_dewater_clean[-df_dewater_clean.content_clean.str.contains(kind_list[i])]
        df_relate = pd.concat([df_relate, df_layout])
    df_relate = df_relate.drop_duplicates()
    df_relate = df_relate.reset_index(drop=True)
    return df_relate

def select_like_index(df_tweet_filter,good_review_list):
    '''
    筛选出用户好评帖
    :param df_tweet_filter:筛选后用户的帖子
    :param good_review_list: 用户好评列表
    :return: 用户好评的帖子对应的索引，列表形式
    '''
    length_good_review = len(good_review_list)
    list_good_review = [i for j in range(length_good_review) for i in range(len(df_tweet_filter)) \
                        if good_review_list[j] in df_tweet_filter['content_clean'][i]]
    df_tweet_like = df_tweet_filter.iloc[list_good_review]
    df_tweet_like = df_tweet_like.reset_index(drop=True)
    kind_like_list = list(set(df_tweet_like.user_id))  # 对该类产品有好评的用户
    return kind_like_list

def select_num_index(df_tweet_filter):
    '''
    筛选出多次出现相关内容的帖子
    :param df_tweet_filter: 筛选后用户的帖子
    :return: 多次出现相关内容的帖子的索引，列表形式
    '''
    kind_num = df_tweet_filter.user_id.value_counts()  # 对同一用户相关帖子数的统计
    df_kind_num = pd.DataFrame(kind_num)
    df_kind_num = df_kind_num.rename(columns={'user_id': 'num'})
    df_kind_num['user_id'] = df_kind_num.index
    df_kind_num = df_kind_num.reset_index(drop=True)
    df_kind_sel = df_kind_num[df_kind_num['num'] >= 3][df_kind_num['num'] <= 50]  # 根据出现帖子数做进一步筛选 选择[3,50]范围内的用户
    kind_num_list = list(df_kind_sel.user_id)  # 多次出现该相关内容的帖子的用户的ID
    return kind_num_list

def select_final_id(list_1,list_2,columns_id_name):
    '''
    确定最后筛选出的用户的ID
    :param list_1: 用户好评帖/多次出现相关内容帖
    :param list_2: 用户好评帖/多次出现相关内容帖
    :param columns_id_name: 最后生成的DataFrame的列名
    :return: 用户ID，DataFrame形式
    '''
    list_final_id = list_1 + list_2
    list_final_id = list(set(list_final_id))
    df_kind_final = pd.DataFrame(list_final_id, columns=['{}'.format(columns_id_name)])
    return df_kind_final

if __name__ == '__main__':
    mongo_db_host = 'mongodb://userreadwrite:qianyi7785@s-uf655e99dcae9134.mongodb.rds.aliyuncs.com:3717,s-uf6a7793442edc04.mongodb.rds.aliyuncs.com:3717/admin?authSource=admin&readPreference=primary&appname=MongoDB%20Compass&ssl=false'
    mongo_db_name = 'weibo'
    mongo_db_table = 'tweet_user'
    #     condition = {'user_id':{'$in':list_dewater_[i]}}
    content = {'_id': 1, 'user_id': 1, 'created_at': 1, 'content': 1, 'nick_name': 1}

    post_db_table_name = "2th100w_water_remove"
    post_host = '10.8.10.23'
    post_port = '5432'
    post_user = 'postgres'
    post_password = 'qianyi'
    post_db_name = 'wb_data'

    df_kind_final_ = {}                #定义空字典 存储最后的ID
    df_dewater_ = {}                   #定义空字典 存储切片后的用户ID
    list_dewater_ = {}                 #定义空字典  切片后的用户ID

    print(f'step1:获取非水军用户:')
    print('-' * 50)
    df_dewater = get_dewater_user(post_db_table_name, post_host, post_port, post_user, post_password,
                                  post_db_name)  # 从第二个100w用户中获取非水军用户
    for i in range(5):
        df_dewater_[i] = df_dewater.iloc[100000 * i:100000 * i + 100000]
        list_dewater_[i] = list(df_dewater_[i]._id)       # 将用户ID分为5份，以列表形式存储

    for i in range(5):
        t_1 = time.time()

        print('step2:开始拉取帖子:')
        print('-' * 50)
        condition = {'user_id': {'$in': list_dewater_[i]}}           #依次获取10万个用户的帖子（约500万条），通过切片后的非水军用户ID查找帖子
        db_dewater_tweet = get_mongo_tweet(mongo_db_host,
                                           mongo_db_name,
                                           mongo_db_table,
                                           condition,
                                           content)

        t_2 = time.time()
        print(f'帖子拉取完毕，用时{round(t_2 - t_1, 2)}s')
        df_dewater_tweet = pd.DataFrame(db_dewater_tweet)          #转DataFrame形式，运行时间较长（4-6min） 清洗后约350万条帖子
        t_3 = time.time()
        print(f'转为DataFrame形式，用时{round(t_3 - t_2, 2)}s')

        print('step3:开始清洗帖子:')
        print('-' * 50)
        df_select = tweet_clean(df_dewater_tweet)                  #清洗帖子，运行时间略长（60s-110s）
        t_4 = time.time()
        print(f'帖子清洗完毕，用时{round(t_4 - t_3, 2)}s')

        find_user_id_num = len(df_select.user_id.unique())
        find_tweet_num = len(df_select)                           #匹配到的用户数目及其发帖数目

        print(f'匹配到的用户的数目：{find_user_id_num}')
        print(f'匹配到的用户的帖子数目：{find_tweet_num}')

        print('step4:关键词匹配相关帖子:')
        print('-' * 50)

        kind_file_path = 'D:/datasets/file/manicure_kind.txt'
        kind_list = read_relate_keywords(kind_file_path)

        df_tweet_match = tweet_select_by_keyword(df_select, kind_list, param1=True)

        print('step5:获得相对应的用户ID:')
        print('-' * 50)

        list_filter_word = ['转评', '评论', '抽奖', '原博', '直播', '中奖', '揪', '包邮', '优惠券']  # 去除广告营销帖、直播抽奖帖
        df_tweet_filter = tweet_select_by_keyword(df_tweet_match, list_filter_word, param1=False)

        good_review_list = ['喜欢', '不错', '好用', '好看', '最爱', '绝了', '不愧']  # 通过关键词匹配用户好评帖子
        like_list = select_like_index(df_tweet_filter, good_review_list)
        num_list = select_num_index(df_tweet_filter)
        columns_id_name = 'manicure_user_id'
        df_kind_final_[i] = select_final_id(like_list, num_list, columns_id_name)
        print(f'第{i + 1}次筛选后的用户的user_id:{df_kind_final_[i]}')

    df_con = pd.concat([df_kind_final_[0], df_kind_final_[1], df_kind_final_[2], df_kind_final_[3], df_kind_final_[4]])
    df_con = df_con.reset_index(drop=True)
    print(f'最后得到的用户的user_id:{df_con}')
#     df_con.to_csv('D:/datasets/generate_file/manicure_user_id_2th100w.csv',index=False)