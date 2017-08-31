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

train = pd.concat([train,df_s_train['oof_predictions_xgb']],axis=1)
test = pd.concat([test,df_s_test['oof_predictions_xgb']],axis=1)
print(train.shape,test.shape)

feature_names = list(train.columns)
do_not_use_for_training = ['id', 'log_trip_duration', 'pickup_datetime', 'dropoff_datetime', 'trip_duration', 'check_trip_duration',
                           'pickup_date', 'avg_speed_h', 'avg_speed_m', 'pickup_lat_bin', 'pickup_long_bin',
                           'center_lat_bin', 'center_long_bin', 'pickup_dt_bin', 'pickup_datetime_group']
feature_names = [f for f in train.columns if f not in do_not_use_for_training]
y = np.log(train['trip_duration'].values + 1)

def lgb_rmsle_score(preds, dtrain):
    labels = np.exp(dtrain.get_label())
    preds = np.exp(preds.clip(min=0))
    return 'rmsle', np.sqrt(np.mean(np.square(np.log1p(preds)-np.log1p(labels)))), False
lgb_params = {
    'learning_rate': 0.1, # try 0.2
    'max_depth': 8,
    'num_leaves': 75,
    'objective': 'regression',
    #'metric': {'rmse'},
    'feature_fraction': 0.9,
    'bagging_fraction': 0.5,
    #'bagging_freq': 5,
    'max_bin': 200}       # 1000
def dummy_rmsle_score(preds, y):
    return np.sqrt(np.mean(np.square(np.log1p(np.exp(preds))-np.log1p(np.exp(y)))))
nfolds = 5
predictions_lgb = np.zeros((test.shape[0], nfolds))
X = train[feature_names].values
kf = model_selection.KFold(n_splits=nfolds, shuffle=True, random_state=2016)
oof_predictions_lgb = np.zeros(X.shape[0])
for fold, (ind_tr, ind_te) in enumerate(kf.split(train)):
    d_train = lgb.Dataset(train[feature_names].values[ind_tr], y[ind_tr])
    d_valid = train[feature_names].values[ind_te]
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    model_lgb = lgb.train(lgb_params, d_train, feval=lgb_rmsle_score, num_boost_round=1672)
    pred = model_lgb.predict(test[feature_names].values)
    oof_predictions_lgb[ind_te] =  model_lgb.predict(d_valid)
    predictions_lgb[:, fold] = pred
    print ('Fold %d: Score %f'%(fold, dummy_rmsle_score(oof_predictions_lgb[ind_te], y[ind_te])))
predictions_lgb = predictions_lgb.mean(axis=1)

print('Test shape OK.') if test.shape[0] == predictions_lgb.shape[0] else print('Oops')
test['trip_duration'] = np.exp(predictions_lgb) - 1
test[['id', 'trip_duration']].to_csv(input + 'gbm_cv_stacking_submission.csv', index=False)
