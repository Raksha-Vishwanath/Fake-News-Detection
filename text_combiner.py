from sklearn.base import BaseEstimator, TransformerMixin
class TextCombiner(BaseEstimator, TransformerMixin):
    def __init__(self, text_features):
        self.text_features = text_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.text_features].apply(lambda x: ' '.join(x.astype(str)), axis=1)