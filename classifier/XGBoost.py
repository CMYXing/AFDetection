import random
import pickle

import numpy as np
import xgboost
from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()


def load_classifier(path=None):
    classifier = XGBoost()

    # if exists, load the pre-trained model
    if path is not None:
        classifier.boost = pickle.load(open(path, 'rb'))

    return classifier


class XGBoost(object):
    """
    XGBoost classifier,
    the probability order is ['A', 'N', 'O', '~']
    """
    def __init__(self):
        my_seed = random.randint(0, 1000)

        self.param = {'learning_rate': 0.1,
                      'eta': 0.1,
                      'num_class': 4,
                      'objective': 'multi:softprob',
                      'eval_metric':'mlogloss'}
        self.param['max_depth'] = 10
        self.param['subsample'] = 0.85
        self.param['colsample_bytree'] = 0.85
        self.param['min_child_weight'] = 4
        self.param['seed'] = my_seed
        self.param['n_jobs'] = -1

        self.num_round = 500
        self.boost = None
        self.pred = None

    def fit(self, train_data, train_label):
        train_label = le.fit_transform(train_label)
        dtrain = xgboost.DMatrix(train_data, label=train_label)
        self.boost = xgboost.train(self.param, dtrain, num_boost_round=self.num_round)

    def predict_prob(self, test_data):
        dtest = xgboost.DMatrix(test_data)
        self.pred = self.boost.predict(dtest)
        return self.pred

    def predict(self, test_data):
        pred_prob = self.predict_prob(test_data)
        pred_num = np.argmax(pred_prob, axis=1)
        try:
            pred = le.inverse_transform(pred_num)
            return pred
        except:
            return pred_num
