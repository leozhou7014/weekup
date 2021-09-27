import pandas as pd


def get_wb_id(user_id,engine):
    """
    get all tweet_id for user
    return: user's tweet list
    """
    sql = 'select t.tweet_id,t.user_id from wb_tweet_dynamic t where t.user_id = %s '%(user_id)
    df = pd.read_sql(sql,engine)
    df = df.drop_duplicates(subset = 'tweet_id',keep  = 'first',inplace = False)
    return df


def get_user_id(tweet_id, engine):
    """
    get all user_id for repost,comment and like under the unique tweet_id
    """
    repost_id = "select r.tweet_id,r.user_id from wb_repost r where r.tweet_id = '(%s)'" % (tweet_id)
    comment_id = "select r.tweet_id,r.user_id from wb_comment r where r.tweet_id = '%s'" % (tweet_id)
    like_id = "select r.tweet_id,r.user_id from wb_like r where r.tweet_id = '%s'" % (tweet_id)
    repost = pd.read_sql(repost_id, engine)
    comment = pd.read_sql(comment_id, engine)
    like = pd.read_sql(like_id, engine)

    all_user = pd.concat([repost, comment, like])
    return all_user


def get_user_id(tweet_id, engine):
    """
    get all user_id for repost,comment and like under the unique tweet_id
    return: tweet's user_id list
    """
    repost_id = "select r.tweet_id,r.user_id from wb_repost r where r.tweet_id = '(%s)'" % (tweet_id)
    comment_id = "select r.tweet_id,r.user_id from wb_comment r where r.tweet_id = '%s'" % (tweet_id)
    like_id = "select r.tweet_id,r.user_id from wb_like r where r.tweet_id = '%s'" % (tweet_id)
    repost = pd.read_sql(repost_id, engine)
    comment = pd.read_sql(comment_id, engine)
    like = pd.read_sql(like_id, engine)

    all_user = pd.concat([repost, comment, like])
    return all_user
