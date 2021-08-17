from sklearn.linear_model import LogisticRegression

def log_reg(x, y):
    """
    Create a model
    """
    sentiment_model = LogisticRegression()
    sentiment_model.fit(x, y)
    return sentiment_model
