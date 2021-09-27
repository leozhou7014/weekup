import paddlehub as hub


def senta_baidu(sentence,senta):
    input_dict = {"text":list(sentence)}
    results = senta.sentiment_classify(data = input_dict)
    return results[0]['sentiment_label']