from scipy.stats import boxcox
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Lasso
from scipy.stats import boxcox
from scipy.special import inv_boxcox

def LassoCrossVal(DataX, DataY, NumVars, Catvars, Demo):
    if Demo is True:
        NumTransformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                         ('scaler', StandardScaler())])
        CatTransfomer = Pipeline(
            steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                   ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        PreProcesser = ColumnTransformer(
            transformers=[
                ('num', NumTransformer, NumVars),
                ('cat', CatTransfomer, Catvars)])
    else:
        NumTransformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                         ('scaler', StandardScaler())])
        PreProcesser = ColumnTransformer(
            transformers=[
                ('num', NumTransformer, NumVars),
            ])
    LassoModel = Pipeline(steps=[('preprocessor', PreProcesser),
                           ('lassocv', LassoCV(cv=5, n_jobs=7, max_iter=10000, verbose=True))])
    LassoModel.fit(DataX, DataY)
    return LassoModel



# XGB training instance 
def BuildModel(DataX, DataY, NumVars, Catvars, Demo, Discrete):
    if Demo is True:
        NumTransformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])
        CatTransfomer = Pipeline(
            steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        PreProcesser = ColumnTransformer(
            transformers =[
                ('num', NumTransformer, NumVars),
                ('cat', CatTransfomer, Catvars)])
    else:
        NumTransformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                         ('scaler', StandardScaler())])
        PreProcesser = ColumnTransformer(
            transformers=[
                ('num', NumTransformer, NumVars),
                ])
    
    if Discrete is True:
        param_grid = {'xgb__objective': ['binary:logisitic'],
                      'xgb__maxdepth': np.arange(2, 10, step=1),
                      'xgb__n_estimators': np.arange(0, 1100, step=100),
                      'xgb__reg_alpha': np.linspace(1e-5, 100, num=10),
                      'xgb__gamma': [i/10.0 for i in range(0, 5)],
                      'xgb__eval_metric':['auc']}
        pipe = Pipeline(steps=[('preprocessor', PreProcesser),
                               ('xgb', XGBClassifier())])


    else:
        param_grid = {'xgb__objective': ['reg:squarederror'],
                      'xgb__maxdepth': np.arange(2, 10, step=1),
                      'xgb__n_estimators': np.arange(0, 1100, step=100),
                      'xgb__reg_alpha': np.linspace(1e-5, 100, num=10),
                      'xgb__gamma': [i/10.0 for i in range(0, 5)],
                      'xgb__eval_metric': ['mae'],
                      }
        pipe = Pipeline(steps=[('preprocessor', PreProcesser),
                            ('xgb', XGBRegressor())])
    ModelTrain = GridSearchCV(pipe, param_grid,
                          cv=5, verbose=1, error_score='raise', n_jobs=7)
    ModelTrain.fit(DataX, DataY)
    print("Best parameter (CV score=%0.3f):" % ModelTrain.best_score_)
    print(ModelTrain.best_params_)
    return ModelTrain

# if __name__ == "__main__":
#     import multiprocessing as mp
#     mp.set_start_method('forkserver')
DataSet = pd.read_csv(
    '/media/labcomp/HDD2/SOMA_SHAHRAD_DEC_2017/PSG.RDI_ML_TRAINING_Apr_01_2019.txt')
PSGVar = 'PSG.RDI'
DataSet.loc[:, "Gender"]=DataSet.Gender.str.lower().values
## try to nomalize the response var
ResponseVar = boxcox(DataSet.loc[:, PSGVar].values+1)[0]
BoxLambda = boxcox(DataSet.loc[:, PSGVar].values+1)[1]
plt.hist(ResponseVar)
NumVars = DataSet.iloc[:, 2:].select_dtypes(include='float').columns.to_list()
Catvars = DataSet.iloc[:, 2:].select_dtypes(exclude='float').columns.to_list()

ProtVars = DataSet.iloc[:, 2:]
## build the model
testX, trainX, testY, trainY = train_test_split(
    ProtVars, ResponseVar, train_size=0.2, shuffle=True)
## Baseline RDI model
dummy_mean = DummyRegressor(strategy='mean')
dummy_mean.fit(trainX, trainY)
PredDummy = dummy_mean.predict(testX)
print('MAE of BaselineModel Dummy  is %0.3f' %
      mean_absolute_error(PredDummy, testY))


## XGB reressor with gridsearchcv
RDIModel = BuildModel(DataX=trainX, DataY=trainY, NumVars=NumVars, Catvars=Catvars, Demo=True, Discrete=False)
RDIModelPred = RDIModel.predict(testX)
print('MAE of GridSearchCV XGB regressor is %0.3f' %
      mean_absolute_error(RDIModelPred, testY))
plt.plot(RDIModelPred, testY, 'ro')
plt.ylabel('Predicted RDI')
plt.xlabel('True RDI')
plt.show()

### Lassomodel
LassoCVModelNoAgeSexBMI = LassoCrossVal(DataX=trainX, DataY=trainY,
                      NumVars=NumVars, Catvars=Catvars, Demo=False)

LassoCVPredNoAgeSexBMI = LassoCVModelNoAgeSexBMI.predict(testX)
print('MAE of LassoCV regressor is %0.3f' %
      mean_absolute_error(LassoCVPredNoAgeSexBMI, testY))
plt.plot(LassoCVPredNoAgeSexBMI, testY, 'ro')
plt.ylabel('Predicted RDI')
plt.xlabel('True RDI')
plt.show()
LassoCVModelAgeSexBMI = LassoCrossVal(DataX=trainX, DataY=trainY,
                                        NumVars=NumVars, Catvars=Catvars, Demo=True)

LassoCVPredAgeSexBMI = LassoCVModelAgeSexBMI.predict(testX)
print('MAE of LassoCV regressor is %0.3f' %
      mean_absolute_error(LassoCVPredAgeSexBMI, testY))

plt.plot(LassoCVPredAgeSexBMI, testY, 'ro')
plt.ylabel('Predicted RDI')
plt.xlabel('True RDI')
plt.show()
ConvPred = inv_boxcox(LassoCVPredAgeSexBMI, BoxLambda)+1
ConvTrue = inv_boxcox(testY, BoxLambda)+1
plt.plot(ConvPred, ConvTrue, 'ro')
## Lasso only with age sex and bmi
NumVarsDemo = ['Age', 'BMI', 'Leptin']










# # build a model to impute BMI for the missing BMI
# MissingBmi = DataSet.index[DataSet.BMI.isna()]
# BMIData = DataSet.drop(MissingBmi)
# BMIDataY = BMIData.BMI.values
# BMIDataX = BMIData.iloc[:, 5:].values
# ProtVars = BMIData.columns[5:]
# testX, trainX, testY, trainY = train_test_split(BMIDataX, BMIDataY, train_size=0.3, shuffle=True)
# trainX.shape
# testX.shape

# ### fit a baseline model
# dummy_mean = DummyRegressor(strategy='mean')
# dummy_mean.fit(trainX, trainY)
# PredDummy = dummy_mean.predict(testX)
# print('MAE of BaselineModel Dummy  is %0.3f' % mean_absolute_error(PredDummy, testY))

# BestModel = BuildModel(DataX=trainX, DataY=trainY)
# BestModel.predict(testX)
# pred = BestModel.predict(testX)
# print('MAE GridSearchCV is %0.3f' % mean_absolute_error(pred, testY))
# ### plot the error
# plt.plot(pred, testY, 'ro')
# plt.ylabel('Predicted BMI')
# plt.xlabel('True BMI')
# plt.show()
# ### predict on the missing BMI 
# MissingBMIPredXGB=BestModel.predict(DataSet.iloc[MissingBmi, 5:].values)

# ## try the lasso model to see if it can be improved
# LassoMod=LassoCrossVal(testX=testX, trainX=trainX, testY=testY, trainY=trainY)
# predLasso = LassoMod.predict(StandardScaler().fit_transform(testX))
# coef_ser = pd.Series(LassoMod.coef_, index=ProtVars)
# coef_imp = coef_ser.index[coef_ser != 0]
# print('MAE LassoCV is %0.3f' % mean_absolute_error(predLasso, testY))

# ### plot the error
# plt.plot(predLasso, testY, 'ro')
# plt.ylabel('Predicted BMI')
# plt.xlabel('True BMI')
# plt.show()
# # clf = XGBRegressor()
# # clf.fit(BMIDataX, BMIDataY)
# ## if import coefficiecnts are selected ?
# BMIDataXImp = BMIData.loc[:, coef_imp].values
# testX, trainX, testY, trainY = train_test_split(
#     BMIDataXImp, BMIDataY, train_size=0.3, shuffle=True)

# LassoModImp = LassoCrossVal(testX=testX, trainX=trainX,
#                          testY=testY, trainY=trainY)

# predLassoImp = LassoModImp.predict(StandardScaler().fit_transform(testX))
# print('MAE LassoCV with most Imp Coef is %0.3f' % mean_absolute_error(predLassoImp, testY))

# DataSet.loc[MissingBmi, 'BMI'] = MissingBMIPredXGB


# ### now train the RDI to see how it performs with age gender and bMI


# le = LabelEncoder()
# Gender = DataSet.Gender.str.lower().values
# le.fit(Gender)
# DataSet.loc[:, "Gender"]=le.transform(Gender)
# ## Missing ages should be dropped
# AgeMissin = DataSet.index[DataSet.Age.isna()]
# RDIData=DataSet.drop(AgeMissin)
# RDIData.reset_index(inplace=True, drop=True)
# RDIDataTrain=RDIData.iloc[:, 5:].values
# RDIy = RDIData.iloc[:, 0].values

# testX, trainX, testY, trainY = train_test_split(
#     RDIDataTrain, RDIy, train_size=0.3, shuffle=True)
# ## make the baseline model
# dummy_mean = DummyRegressor(strategy='mean')
# dummy_mean.fit(trainX, trainY)
# PredDummy=dummy_mean.predict(testX)

# RDI_XGBreg=BuildModel(DataX=trainX, DataY=trainY)
# predXGB = RDI_XGBreg.predict(testX)


# LassoMod = LassoCrossVal(testX=testX, trainX=trainX,
#                          testY=testY, trainY=trainY)
# predLasso = LassoMod.predict(StandardScaler().fit_transform(testX))
# coef_ser_rdi = pd.Series(LassoMod.coef_, index=ProtVars)
# coef_imp_rdi = coef_ser.index[coef_ser_rdi != 0]


# print('MAE GridSearchCV is %0.3f' % mean_absolute_error(predXGB, testY))
# print('MAE LassoCV is %0.3f' % mean_absolute_error(predLasso, testY))
# print('MAE Dummy is %0.3f' % mean_absolute_error(PredDummy, testY))
# #MAE GridSearchCV is 14.898
# plt.plot(predXGB, testY, 'ro')
# plt.ylabel('Predicted RDI')
# plt.xlabel('True RDI')
# plt.show()

# ## use only with important coefs
# RDIDataXImp = RDIData.loc[:, coef_imp_rdi].values
# testX, trainX, testY, trainY = train_test_split(
#     RDIDataXImp, RDIy, train_size=0.3, shuffle=True)

# LassoModImpRDI = LassoCrossVal(testX=testX, trainX=trainX,
#                             testY=testY, trainY=trainY)

# predLassoImp = LassoModImpRDI.predict(StandardScaler().fit_transform(testX))
# print('MAE LassoCV with most Imp Coef is %0.3f' %
#       mean_absolute_error(predLassoImp, testY))

# RDI_XGBregImp = BuildModel(DataX=trainX, DataY=trainY)
# predXGBImp = RDI_XGBreg.predict(testX)

