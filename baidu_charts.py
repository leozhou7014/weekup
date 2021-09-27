from pyecharts.charts import Pie
from pyecharts import options as opts
import pandas as pd
from pyecharts.globals import SymbolType
from pyecharts.globals import ThemeType,CurrentConfig
from pyecharts.charts import WordCloud
from pyecharts import options as opts
import jieba.analyse as analyse


def draw_sentiment_pie(df_senta,title):
    num = df_senta.value_counts()
    labels = ['正面情绪帖子', '负面情绪帖子']
    values = [int(num[1]), int(num[0])]


    pie_emotion_reba = (
        Pie()
            .add("", [list(z) for z in zip(labels, values)])
            .set_colors(["blue", "green"])
            .set_global_opts
            (
            title_opts=opts.TitleOpts(title=title, pos_left='center'),
            legend_opts=opts.LegendOpts(pos_left='10%', pos_top='middle', orient='vertical')
        )
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}:{d}%"))

    )
    return pie_emotion_reba


def draw_tfidf_wordcloud(result_list,top_n,title):
    keywords = analyse.extract_tags(str(result_list),topK = top_n, withWeight = True)
    word1 = WordCloud(init_opts=opts.InitOpts(width='1000px', height='600px', theme=ThemeType.MACARONS))
    word1.add('词频', data_pair=keywords,
              word_size_range=[15, 108], textstyle_opts=opts.TextStyleOpts(font_family='cursive'),
              shape=SymbolType.DIAMOND)
    word1.set_global_opts(title_opts=opts.TitleOpts(title),
                          toolbox_opts=opts.ToolboxOpts(is_show=False),
                          tooltip_opts=opts.TooltipOpts(is_show=True, background_color='red', border_color='yellow'))
    word1.render("%s.html" %(title))
    return word1