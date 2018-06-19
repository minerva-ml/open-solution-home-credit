from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from toolkit.sklearn_transformers.models import SklearnClassifier


def get_sklearn_classifier(ClassifierClass, normalize=False, **kwargs):

    class SklearnBinaryClassifier(SklearnClassifier):
        def transform(self, X, y=None, target=1, **kwargs):
            prediction = self.estimator.predict_proba(X)[:, target]
            return {SklearnClassifier.RESULT_KEY: prediction}

    if normalize:
        return SklearnBinaryClassifier(Pipeline([('standarizer', StandardScaler()),
                                                 ('classifier', ClassifierClass(**kwargs))]))

    return SklearnBinaryClassifier(ClassifierClass(**kwargs))
