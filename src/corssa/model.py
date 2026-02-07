from sklearn.metrics import classification_report, confusion_matrix
from catboost import CatBoostClassifier

class CORSSA(CatBoostClassifier):

    def __init__(self, *args, **kwargs):

        default_kwargs = {
            'loss_function': 'MultiClass',
            'cat_features': ['aa'],
            'eval_metric': 'Accuracy'
        }

        super().__init__(
            *args,
            **{**default_kwargs, **kwargs}
        )

    def grid_search(
            self,
            *args,
            **kwargs
        ):

        default_kwargs = {
            'loss_function': 'MultiClass',
            'cat_features': ['aa'],
            'eval_metric': 'Accuracy'
        }

        super().grid_search(
            *args,
            **{**default_kwargs, **kwargs}
        )