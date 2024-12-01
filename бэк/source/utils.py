import datetime
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_log_error,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier


RANDOM_STATE = 47

now = datetime.datetime.now().strftime('%Y-%m-%d_%H%M')
SUBMISSION_PATH = f'BezShansov.csv'


def get_results():
    data = pd.read_csv('train.csv')
    submission = pd.read_csv('test.csv')
    train, test = train_test_split(data, test_size=0.5, random_state=RANDOM_STATE)

    remove_features = train.columns[train.columns.str.startswith('__')].tolist()
    continuous_features = list(set(train.dtypes[train.dtypes != 'object'].index.tolist())
                               - set(remove_features))
    X_train = train[continuous_features].fillna(0.)
    X_test = test[continuous_features].fillna(0.)
    X_sub = submission[continuous_features].fillna(0.)
    reg_model = HistGradientBoostingRegressor(
        random_state=RANDOM_STATE,
        max_iter=100,
        learning_rate=0.1,
        l2_regularization=1.0,
        max_leaf_nodes=45,
        min_samples_leaf=1,
    )

    reg_model.fit(X_train, train['__price_doc'])

    train['__price_predict'] = reg_model.predict(X_train)
    test['__price_predict'] = reg_model.predict(X_test)
    submission['__price_predict'] = reg_model.predict(X_sub)
    train.loc[train['__price_predict'] < 0.1, '__price_predict'] = 0.1
    test.loc[test['__price_predict'] < 0.1, '__price_predict'] = 0.1
    submission.loc[submission['__price_predict'] < 0.1, '__price_predict'] = 0.1
    clf_model = HistGradientBoostingClassifier(
        loss='log_loss',
        random_state=RANDOM_STATE,
        max_iter=100,
        learning_rate=0.1,
        l2_regularization=1.0,
        max_leaf_nodes=25,
        min_samples_leaf=1,
    )

    clf_model.fit(X_train, train['__churn'])

    train['__churn_prob'] = clf_model.predict_proba(X_train)[:, 1]
    test['__churn_prob'] = clf_model.predict_proba(X_test)[:, 1]
    submission['__churn_prob'] = clf_model.predict_proba(X_sub)[:, 1]

    mean_price = test['__price_predict'].mean()

    def alg(x: pd.DataFrame):
        priority = 0
        if x['__churn_prob'] <= 0.2:
            priority += 1
            if x['__price_predict'] >= mean_price:
                priority += 1
            else:
                priority = priority
        elif x['__churn_prob'] >= 0.2:
            priority -= 1

        x['__priority'] = 0

        if priority < 0:
            x['__priority'] = -1
        else:
            x['__priority'] = x['__price_predict'] + priority

        return x['__priority']

    train['__priority'] = train.apply(alg, axis=1)
    test['__priority'] = test.apply(alg, axis=1)
    submission['__priority'] = submission.apply(alg, axis=1)

    score = pd.concat([
        pd.Series(calc_all_metrics(test), name='test'),
    ], axis=1)

    mysub = submission[['__price_predict', '__churn_prob', '__priority']]
    mysub.to_csv(SUBMISSION_PATH, index=False)

    if mysub.shape != (9988, 3):
        raise ValueError('Неправильный размер submission файла')

    METRICS_DESC = {
        "total_profit": "Итоговая полученная прибыль (Ключевая метрика), млн руб.",
        "issue_amount": "Итоговая выданная сумма (25 000 максимум), млн руб.",
        "bad_loans": "Доля выданных кредитов с задолженностью, %",
        "churn_auc": "Метрика ROC AUC по модели предсказания задолженности",
        "price_nmsle": "Метрика Negative Mean Squared Logarithmic Error по модели предсказания стоимости",
    }
    score['desc'] = score.index.map(METRICS_DESC)

    score_list = score.reset_index().to_dict(orient='records')

    return score_list


def calc_all_metrics(data: Any) -> Dict[str, float]:
    def is_credit_issued(x: Any):
        ratio = x['__price_predict'] / x['__price_doc']
        if x['__priority'] <= 0:
            value = 0.0
        elif 0.9 < ratio < 1.0:
            value = x['__price_predict']
        elif 1.0 <= ratio < 1.1:
            value = x['__price_doc']
        else:
            value = 0.0

        return value

    def calc_profit(x: pd.DataFrame) -> np.array:
        if x['is_credit'] == 0.0:
            return 0.0
        if x['__churn'] == 1:
            return -x['debt'] * 2.0
        if x['debt'] < 5:
            return x['debt'] * 0.3
        if x['debt'] < 9:
            return x['debt'] * 0.4
        if x['debt'] >= 9:
            return x['debt'] * 0.5

    max_account = 25e3

    s = (
        data[['__priority', '__churn', '__churn_prob', '__price_doc', '__price_predict']]
        .sort_values('__priority', ascending=False)
        .copy(True)
    )

    s['debt'] = s.apply(is_credit_issued, axis=1)
    s['debt_cum'] = s['debt'].cumsum()
    s['is_credit'] = 0
    s.loc[(s['debt'] > 0) & (s['debt_cum'] <= max_account), 'is_credit'] = 1
    s['profit'] = s.apply(calc_profit, axis=1)

    total_profit = round(s['profit'].sum(), 2)
    good_credits_count = int(s['is_credit'].sum())
    good_credits_debt = int(s[s['is_credit'] == 1]['debt'].sum())
    bad_credits_count = s[s['is_credit'] == 1]['__churn'].sum()

    return {
        'total_profit': int(total_profit),
        'issue_amount': good_credits_debt,
        'bad_loans': round(bad_credits_count / (good_credits_count + bad_credits_count) * 100.0, 1),
        'churn_auc': round(roc_auc_score(y_true=s['__churn'], y_score=s['__churn_prob']), 3),
        'price_nmsle': round(
            -mean_squared_log_error(y_true=s['__price_doc'], y_pred=s['__price_predict']),
            3,
        ),
    }
