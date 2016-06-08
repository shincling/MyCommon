import xgboost as xgb
# read in data
dtrain = xgb.DMatrix('similarity_withLap_0607')
dtest = xgb.DMatrix('similarity_withLap_0607')
# specify parameters via map
param = {'max_depth':50, 'eta':1, 'silent':0, 'objective':'binary:logistic' }
num_round = 80
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)
f_out=open('similarity_lap_result_0607','a')
for each in preds:
    f_out.write(str(each))
    f_out.write('\n')
