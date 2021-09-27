import collections

import pandas as pd
import datetime

import pyecharts
from pyecharts.globals import ThemeType, SymbolType

from stopwords import *
from pyecharts import options as opts
from pyecharts.charts import Pie, Map, WordCloud


def get_age(birthday, age_cut=True):
    """
    去除生日中的中文（星座），空值
    返回用户年龄
    :param birthday: 用户生日series数据
    :param age_cut: 是否对年龄进行分级
    :return: 用户年龄series数据
    """
    birthday = birthday.apply(lambda x: pd.to_datetime(x, errors='coerce'))
    birthday = birthday.dropna().reset_index(drop=True)
    today_year = datetime.datetime.now().year  # 现在的年
    age = [today_year - birth.year for birth in birthday]
    age = pd.DataFrame(age, columns=['age'])  # 用户年龄
    if age_cut:
        labels = ['少年', '青年1', '青年2', '青年3', '青年4', '青年5', '中年', '老年', 'other']
        bins = [0, 15, 20, 25, 30, 35, 40, 50, 70, 200]
        age = pd.cut(age, bins=bins, labels=labels, right=False)
    return age


def draw_pie(df, title):
    """
    分布饼图
    :param df: 某标签series数据
    :param title: 饼图标题
    :return: 饼图配置
    """

    df_count = df.value_counts()
    df_index = df_count.index.tolist()
    df_value = df_count.values.tolist()

    pie = (
        Pie()
        .add("", [list(z) for z in zip(df_index, df_value)])
        .set_global_opts
        (
            title_opts=opts.TitleOpts(title=f"{title}", pos_left='center'),
            legend_opts=opts.LegendOpts(pos_left='10%', pos_top='middle', orient='vertical')  # 图例
        )
        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}:{d}%"))  # 数据标签格式
    )

    return pie


def get_province(all_province):
    """
    返回省份正确的信息
    :param all_province:用户省份未清洗series数据
    :return: 用户正确省份信息series数据
    """
    province = [pro for pro in all_province if pro not in ['其他', 'None', '海外']]
    province = pd.DataFrame(province, columns=['province'])

    return province


def draw_map(province, title):
    """
    地理分布图
    :param province: 省份信息series数据
    :param title: 地理分布图名称
    :return: 地理分布图配置
    """
    province_count = province.value_counts().reset_index()
    province_values = province_count.iloc[:, 1].tolist()
    province_index = province_count.loc[:, 'province'].tolist()
    max_num = max(province_values)
    list1 = [[province_index[i], province_values[i]] for i in range(len(province_index))]

    province_map = (
        Map()
        .add("", list1, maptype="china")
        .set_global_opts(
            title_opts=opts.TitleOpts(title=f"{title}"),
            visualmap_opts=opts.VisualMapOpts(max_=max_num)  # 最大数据范围
        )
    )
    return province_map


def get_label(label, seg_label=False):
    """
    清洗用户的标签信息
    :param label: 标签信息series数据
    :return: 清洗后的用户标签信息
    """
    label_drop = label.dropna()
    label_cut = label_drop.apply(lambda x: x.split(','))
    label_list = list(chain.from_iterable(label_cut.values))
    if seg_label:
        seg_label_list = [jieba.cut(word) for word in label_list]
        seg_label_list = list(chain.from_iterable(seg_label_list))
        label_list = [word for word in seg_label_list if word not in STOPWORDS]

    return label_list


def draw_wordcloud(wordlist, title):
    """
    词云图
    :param wordlist: 待生成词云的词语列表
    :param title: 词云标题
    :return:词云配置
    """

    word_counts = collections.Counter(wordlist)
    word_count_topn = word_counts.most_common(100)

    word_cloud = (
        WordCloud(init_opts=opts.InitOpts(width='1000px', height='600px', theme=ThemeType.MACARONS))
        .add(
             '词频',
             data_pair=word_count_topn,
             word_size_range=[15, 108], textstyle_opts=opts.TextStyleOpts(font_family='cursive'),
             shape=SymbolType.DIAMOND)

        .set_global_opts(
            title_opts=opts.TitleOpts(f'{title}'),
            toolbox_opts=pyecharts.options.ToolboxOpts(is_show=False),
            tooltip_opts=opts.TooltipOpts(is_show=True, background_color='red', border_color='yellow'))

    )
    return word_cloud
