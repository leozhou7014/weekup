#coding:utf-8
'''

'''
import os 
#dir_path = os.path.dirname(os.path.realpath(__file__))
stop_txt = os.getcwd()+'/stop.txt'

def stopwordslist(path):
    """
    read stopwords file
    :return: stopwords
    """
    stopwords = [line.strip() for line in open(path, encoding='UTF-8').readlines()]
    return stopwords

STOPWORDS = stopwordslist(stop_txt)