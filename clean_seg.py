import jieba
from jieba import posseg
import re
from LAC import LAC
import pandas as pd


def stopwordslist():
    """
    read stopwords file
    :return: stopwords
    """
    stopwords = [line.strip() for line in open('stopwords.txt' ,encoding='UTF-8').readlines()]
    return stopwords


def seg_text(sentence):
    seg_t = jieba.cut(sentence)
    stopwords = stopwordslist()
    outstr = []
    for i in seg_t:
        if i not in stopwords and len(i ) >1:
            outstr.append(i)
    return outstr

def word_lac(text):
    """"
    just remain noun in all words. Clean documents using stopwords,jieba seg and the length for words should
    longer than 2.
    return: cleaned word list

    """
    stopwords = stopwordslist()
    r = jieba.posseg.cut(text)
    a = []
    for w in r:
        if w.flag in ['n' ,'ns' ,'nr' ,'nz'] and w.word not in stopwords and len(w.word ) >1:
            a.append(w.word)
        else:
            pass
    return a

def clean_text(df,param1 = True,param2 = True):
    """
    固定删除文本中的日期，时间，邮件地址，网址，“转发微博”，“显示地图”，“分享图片”
    “XXX的微博视频”，含“谦毅”的帖子，选择删除表情，标签
    :param df: dataframe, the content we need to clean is df['content']
    :param param1: 是否删除文中的【】中的表情
    :param param2: 是否删除文中#键间的文字
    :return: 清洗过后的dataframe文本
    """
    df = df.astype(str)  # 将所有元素转化为str类型
    df['content_clean'] = df['content'].str.replace(r"\d+/\d+/\d+|", '') \
                                 .str.replace(r"[0-2]?[0-9]:[0-6][0-9]", '') \
                                 .str.replace(r"[\w]+@[\.\w]+", '') \
                                 .str.replace(u"@(.*?):|/@(.*) /", '') \
                                 .str.replace(r"([hH][tT]{2}[pP]://|[hH][tT]{2}[pP][sS]://|[wW]{3}."
                                              r"|[wW][aA][pP].|[fF][tT][pP].|[fF][iI][lL][eE].)[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]",'')\
                                 .str.replace(u"谦毅:(.*?)|/转发微博/g|/显示地图/g|/分享图片/g",'')\
                                 .str.replace(u"#\S*?的微博视频|:\S*?的微博视频",'')\
                                 .str.replace(u'\[.*?]|#| \S*? 显示地图|转发微博|分享图片| \S*?的微博视频|//@.*?:|@\S*? |@\S*?$','')

    if param1:
        df['content_clean'] = df['content_clean'].str.replace(u"\\[.*?]",'') # 删除【】和【】中的表情
    if param2:
        df['content_clean'] = df['content_clean'].str.replace(u"\\#.*?#",'') # 删除#号和#间的话题

    return df




def lac_text_baidu(sentence, tag_list, baidu_lac_model):
    stopwords = stopwordslist()
    baidu_lac_model.load_customization('mydict.txt')
    lac_t = baidu_lac_model.run(sentence)
    data = {'word': lac_t[0], 'tag': lac_t[1]}
    df = pd.DataFrame(data)
    df1 = df['word'].loc[df['tag'].isin(tag_list)]
    outstr = []
    for i in df1:
        if i not in stopwords and len(i) > 1:
            outstr.append(i)
    return outstr





