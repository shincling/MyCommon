import xgboost as xgb
# read in data
dtrain = xgb.DMatrix('similarity_withLap_0607')
dtest = xgb.DMatrix('similarity_withLap_0607')
# specify parameters via map
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)
f_out=open('similarity_lap_result_0607','a')
for each in preds:
    f_out.write(str(each))
    f_out.write('\n')
