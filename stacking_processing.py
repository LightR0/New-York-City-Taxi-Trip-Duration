import  math
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn import model_selection

input = 'D:/Data/Kaggle/Taxi/'
train = pd.read_csv(input + 'df_train.csv')
test = pd.read_csv(input + 'df_test.csv')
feature_names = list(train.columns)
do_not_use_for_training = ['id', 'log_trip_duration', 'pickup_datetime', 'dropoff_datetime', 'trip_duration', 'check_trip_duration',
                           'pickup_date', 'avg_speed_h', 'avg_speed_m', 'pickup_lat_bin', 'pickup_long_bin',
                           'center_lat_bin', 'center_long_bin', 'pickup_dt_bin', 'pickup_datetime_group']
feature_names = [f for f in train.columns if f not in do_not_use_for_training]
y = np.log(train['trip_duration'].values + 1)

#lgb 5折cv
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

#xgb 5折cv
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
    pred = model.predict(dtest)#, ntree_limit=model.best_ntree_limit)
    oof_predictions_xgb[ind_te] = model.predict(d_valid)#, ntree_limit=model.best_ntree_limit)
    predictions_xgb[:, fold] = pred
    print ('Fold %d: Score %f'%(fold, model.best_score))
predictions_xgb = predictions_xgb.mean(axis=1)

print(train[feature_names].values.shape,test[feature_names].values.shape,
     oof_predictions_xgb.shape,oof_predictions_lgb.shape,
     predictions_xgb.shape,predictions_lgb.shape)

#把lgb和xgb预测训练集和测试集的输出作为一列特征分别保存
df_stacking_train = pd.DataFrame()
df_stacking_test = pd.DataFrame()
df_stacking_train['oof_predictions_xgb'] = oof_predictions_xgb
df_stacking_train['oof_predictions_lgb'] = oof_predictions_lgb
df_stacking_test['oof_predictions_xgb'] = predictions_xgb
df_stacking_test['oof_predictions_lgb'] = predictions_lgb
df_stacking_train.to_csv(input + 'df_stacking_train.csv',index=False)
df_stacking_test.to_csv(input + 'df_stacking_test.csv',index=False)
