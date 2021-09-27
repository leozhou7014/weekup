import jieba
import pandas as pd
from LAC import LAC
from jieba import posseg
from stopwords import *
from gensim.models import Phrases, Word2Vec
import numpy as np


def seg_text(sentence, jieba_model=True):
    """
    使用jieba/lac工具进行分词
    :param sentence: 待分词的句子
    :param jieba_model:指定分词方法：jieba/lac，默认为jieba
    :return: 分词结果
    """

    if jieba_model:
        seg_sentence = jieba.cut(sentence)
    else:
        lac_seg = LAC(mode='seg')
        seg_sentence = lac_seg.run(sentence)

    out_word = [word for word in seg_sentence if word not in STOPWORDS and len(word) > 1]
    return out_word


def tag_text_lac(sentence, tag_list):
    """
    使用LAC方法进行词性选择，所得单词列表去除属于停词表的词，且单词长度大于1

    :param sentence: 待分词的句子
    :param tag_list: 指定所需要的词性列表
    :return:分词结果
    """
    lac_tag = LAC(mode='lac')
    tag_text = lac_tag.run(sentence)
    df = pd.DataFrame({'word': tag_text[0], 'tag': tag_text[1]})
    df1 = df['word'].loc[df['tag'].isin(tag_list)]
    out_word = [word for word in df1 if word not in STOPWORDS and len(word) > 1]
    return out_word


def tag_text_jieba(sentence, tag_list):
    """"
    使用jieba进行词性选择,所得单词列表去除属于停词表的词，且单词长度大于1

    :param sentence: 待分词的句子
    :param tag_list: 指定所需要的词性列表
    :return:分词结果
    """
    tag_text = jieba.posseg.cut(sentence)
    out_word = [word for word in tag_text if
                word.flag in tag_list and word.word not in STOPWORDS and len(word.word) > 1]
    return out_word


def clean_text(df, param1=False, param2=False):
    """
    固定删除文本中的日期，时间，邮件地址，网址，“转发微博”，“显示地图”，“分享图片”
    “XXX的微博视频”，含“谦毅”的帖子，选择删除表情，标签
    :param df: dataframe, the content we need to clean
    :param param1: 是否删除文中的[]中的表情
    :param param2: 是否删除文中#键间的文字
    :return: 清洗过后的dataframe文本
    """
    df = df.astype(str)  # 将所有元素转化为str类型
    df = df.str.replace(r"\d+/\d+/\d+", ' ') \
        .str.replace(r"[0-2]?[0-9]:[0-6][0-9]", ' ') \
        .str.replace(r"[\w]+@[\.\w]+", ' ') \
        .str.replace(r"@(.*?):|/@(.*) /", ' ') \
        .str.replace(r"([hH][tT]{2}[pP]://|[hH][tT]{2}[pP][sS]://|[wW]{3}." +
                     r"|[wW][aA][pP].|[fF][tT][pP].|[fF][iI][lL][eE].)[-A-Za-z0-9+&@#/%?=~_|!:,"
                     r".;]+[-A-Za-z0-9+&@#/%=~_|]", ' ') \
        .str.replace(u"谦毅:(.*?)|/转发微博/g|/显示地图/g|/分享图片/g", ' ') \
        .str.replace(r"#\S*?的微博视频|:\S*?的微博视频", ' ') \
        .str.replace(r'\[.*?]|#| \S*? 显示地图|转发微博|分享图片| \S*?的微博视频|//@.*?:|@\S*? |@\S*?$', ' ') \
        .str.replace(r'\s+', ' ')
    if param1:
        df = df.str.replace(u"\\[.*?]", '')  # 删除【】和【】中的表情
    if param2:
        df = df.str.replace(u"\\#.*?#", '')  # 删除#号和#间的话题

    return df


def doc_vec(df):
    """
    word2Vec方法生成词向量
    """
    bigram_trans = Phrases(df)
    word2Vec_model = Word2Vec(bigram_trans[df], min_count=1, vector_size=20, epochs=200)  # 生成word2Vec模型
    feature = np.array([])  # 改成数组
    for tokens in df:
        zero_vector = np.zeros(word2Vec_model.vector_size)
        vectors = np.array([])  # 改成数组
        for token in tokens:
            if token in word2Vec_model.wv:
                vectors.append(word2Vec_model.wv[token])
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            feature.append(avg_vec)
        else:
            feature.append(zero_vector)
    return feature
