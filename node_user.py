import networkx as nx
import pandas as pd


def find_topn_node(df,topn,label_node,label_edge):
   """

   return top n kol user_id in fan chart
   :param df: fan chart(fan_id & followed_id)
   :param topn: top n user you want to know
   :param label_node: node label
   :param label_edge: edge label
   :return: dataframe type top n kol, columns = ['user_id','followed_num']
   """
   g = nx.DiGraph()  # direct graph
   for i, row in df.iterrows():
      g.add_node(row['fan_id'], label= label_node)
      g.add_node(row['followed_id'], label= label_node)
      g.add_edge(row['fan_id'], row['followed_id'], label= label_edge)

   top_list = pd.DataFrame(list(sorted(g.in_degree, key=lambda x: x[1], reverse=True)))
   top_list_n = pd.DataFrame({'user_id':top_list[0],'followed_num':top_list[1]}).iloc[0:topn]

   return top_list_n


def fill_nickname_fannum(df,table_name,engine):
   """
   fill the nick_name and fan_num for the given user_id
   :param df: dataframe dataset has column 'user_id'
   :param engine: connection with postgresql
   :param table_name: user info table,which contains fan_num and nick_name
   :return: df with user's fan_num and nick_name
   """
   user_list = list(df)
   sql = f'select u._id,u.nick_name,u.fan_num from {table_name} u where u._id = any(array{user_list})'

   df_kol = pd.read_sql(sql,engine)

   return df_kol

