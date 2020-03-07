import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from model.wpda import WPDA
from model.wcda import WCDA
from model.fixed_dose import FixedDose
from loader.warfarin_loader import WarfarinLoader
from evaluation.evaluation import Evaluation
from model.model import Model
import pandas as pd
from data.testModel import testModel
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Get data
wf = WarfarinLoader()
model = testModel(True)

# Prepare data
model.featurize(wf)
model.prepare_XY()
X = model.get_X()
y = model.get_Y()
print(X.shape)
#print(X[0])
X = pd.DataFrame(data=X,columns=model.feature_columns)




#apply SelectKBest class to extract top 10 best features
n = len(model.feature_columns)
bestfeatures = SelectKBest(score_func=chi2, k=n)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Features','Score']  #naming the dataframe columns
print(featureScores.nlargest(n,'Score'))  #print 10 best features
print(list(featureScores.nlargest(n,'Score')["Features"]))

"""
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


data=X
data["warfarin dose"]=y
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
"""
