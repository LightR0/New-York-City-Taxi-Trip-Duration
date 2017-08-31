# New-York-City-Taxi-Trip-Duration
主要是修改Kernelhttps://www.kaggle.com/gaborfodor/from-eda-to-the-top-lb-0-367<br>
Stacking:https://github.com/freelzy/Tencent_Social_Ads
修改部分：调参，Stacking，线性加权<br>
pre_processing.py：特征工程<br>
stacking_processing.py：xgb和lgb5折cv对训练集和测试集分别预测，作为两列特征保存<br>
Stacking_lgb.py：把xgb 5折cv生成的一列特征加到原始特征上，用lgb进行5折cv<br>
Stacking_xgb.py：把lgb 5折cv生成的一列特征加到原始特征上，用xgb进行5折cv<br>
Line_stacking.py：线性加权<br>
模型：Xgboost，lightGBM<br>
最优成绩：Public 0.37210，(xgb 5折cv)*0.6+(lgb stacking)*0.4 <br>
说明：最优成绩并不是xgb_stacking与lgb_stacking线性加权得到，而是xgb 5折cv与lgb_stacking线性加权得到<br>
目的：保存代码<br>
致谢：感谢Kernel部分各位选手的开源方案，受益匪浅
