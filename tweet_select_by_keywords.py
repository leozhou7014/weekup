import pandas as pd
from tqdm import tqdm

def tweet_select_by_keywords(df_before_match, kind_list,column_name, param1=True):
    '''
    通过关键词匹配用户帖子
    :param df_before_match: 匹配之前的表格
    :param kind_list: 关键词对应的列表
    :param column_name： 从表格中的某一列进行匹配
    :param param1: True为匹配，False为筛除
    :return: 匹配之后的表格（DataFrame形式）
    '''
    df_relate = df_before_match[df_before_match[f'{column_name}'].str.contains(kind_list[0])]        #包含kind_list第一个元素的帖子
    kind_length = len(kind_list)
    for i in (tqdm(range(1, kind_length))):
        df_layout = df_before_match[df_before_match[f'{column_name}'].str.contains(kind_list[i])]    #包含kind_list第i+1个元素的帖子
        df_relate = pd.concat([df_relate, df_layout])                                                #将所有包含关键词的帖子的DataFrame拼接起来
        df_relate = df_relate.drop_duplicates()                                                      #去重
        df_relate = df_relate.reset_index(drop=True)                                                 #重新排列索引
    if param1:
        df_select_by_keywords = df_relate                                                            #参数param1为True，筛选包含关键词的帖子
    else:
        df_select_by_keywords = df_before_match[-df_before_match[f'{column_name}'].isin(df_relate[f'{column_name}'])]
        df_select_by_keywords = df_select_by_keywords.reset_index(drop=True)                         #参数param1为False，筛选不包含关键词的帖子
    return df_select_by_keywords

# 实例
# df_tweet = pd.read_csv('D:/datasets/qianyi/date/20210920-20210926/10w_user_contain_place_tweet.csv')
# df_before_match = df_tweet
# kind_list = ['北京','上海','天津','重庆']
# column_name = 'content'
# param1 = True
# df_1 = tweet_select_by_keywords(df_before_match,kind_list,column_name)                             #默认筛选包含关键词的帖子
#
# param2 = False
# df_2= tweet_select_by_keywords(df_before_match,kind_list,column_name,param1=param2)
#
# print(df_tweet)
# print(df_1)
# print(df_2)