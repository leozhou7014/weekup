def tweet_clean_select(df, param1 = True, param2 = True):
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

    df['content_clean_length'] = df['content_clean'].apply(lambda x: len(x))
    df_select = df[(df['content_clean_length'] > 1)]  # 筛选掉清洗后帖子长度小于等于1的帖子
    df_select = df_select.dropna(axis=0, subset=['content'])  # 删除缺失值
    df_select = df_select.reset_index(drop=True)
    return df_select

