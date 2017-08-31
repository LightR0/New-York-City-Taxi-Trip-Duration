import  math
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn import model_selection

input = 'D:/Data/Kaggle/Taxi/'
train = pd.read_csv(input + 'df_train.csv')
test = pd.read_csv(input + 'df_test.csv')
df_s_train = pd.read_csv(input + 'df_stacking_train.csv')
df_s_test = pd.read_csv(input + 'df_stacking_test.csv')
print(train.shape,test.shape)

train = pd.concat([train,df_s_train['oof_predictions_lgb']],axis=1)
test = pd.concat([test,df_s_test['oof_predictions_lgb']],axis=1)
print(train.shape,test.shape)

feature_names = list(train.columns)
do_not_use_for_training = ['id', 'log_trip_duration', 'pickup_datetime', 'dropoff_datetime', 'trip_duration', 'check_trip_duration',
                           'pickup_date', 'avg_speed_h', 'avg_speed_m', 'pickup_lat_bin', 'pickup_long_bin',
                           'center_lat_bin', 'center_long_bin', 'pickup_dt_bin', 'pickup_datetime_group']
feature_names = [f for f in train.columns if f not in do_not_use_for_training]
y = np.log(train['trip_duration'].values + 1)

xgb_pars = {'min_child_weight': 75, 'eta': 0.1,  'max_depth': 10, 'colsample_bytree': 0.3,
            'subsample': 1.0,'booster' : 'gbtree', 'silent': 1,'eval_metric': 'rmse', 'objective': 'reg:linear'}
dtest = xgb.DMatrix(test[feature_names].values)
nfolds = 5
predictions_xgb = np.zeros((test.shape[0], nfolds))
X = train[feature_names].values
kf = model_selection.KFold(n_splits=nfolds, shuffle=True, random_state=2016)
oof_predictions_xgb = np.zeros(X.shape[0])
for fold, (ind_tr, ind_te) in enumerate(kf.split(train)):
    d_train = xgb.DMatrix(train[feature_names].values[ind_tr], y[ind_tr])
    d_valid = xgb.DMatrix(train[feature_names].values[ind_te], y[ind_te])
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    model = xgb.train(xgb_pars, d_train, 1000, watchlist, early_stopping_rounds=50, maximize=False, verbose_eval=False)
    pred = model.predict(dtest)
    oof_predictions_xgb[ind_te] = model.predict(d_valid)
    predictions_xgb[:, fold] = pred
    print ('Fold %d: Score %f'%(fold, model.best_score))
predictions_xgb = predictions_xgb.mean(axis=1)

print('Test shape OK.') if test.shape[0] == predictions_xgb.shape[0] else print('Oops')
test['trip_duration'] = np.exp(predictions_xgb) - 1
test[['id', 'trip_duration']].to_csv(input + 'xgb_cv_stacking_submission.csv', index=False)
