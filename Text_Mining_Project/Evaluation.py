
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import model_selection

#######################Evaluation in Justice Level ##########################################

#Getting the data frame and the features frame resulting from the preprocessing
feature_df_gnb = pd.read_csv("/home/ahlem/Text_Mining_Project/preprocessed_data_justiceLevel.csv", encoding="ISO-8859-1").astype('int')

feature_data_X = feature_df_gnb[['term', 'natural_court', 'argument_month', 'decision_month',
                                                            'decision_delay', 'justice',
                                                            'petitioner', 'respondent', 'jurisdiction', 'adminAction',
                                                            'caseSource',
                                                            'caseOrigin', 'certReason', 'lc_case_decision',
                                                            'issue', 'issueArea', 'lawType', 'lawSupp',
                                                            'petitionerState', 'respondentState',
                                                            'partyWinning', 'majVotes']]
target_data_Y = feature_df_gnb["justice_level_decision"]

#comparision
models = []
seed=7
models.append(('AdaBoostClassifier', AdaBoostClassifier()))
models.append(('GaussianNB', GaussianNB()))
models.append(('RandomForestClassifier', RandomForestClassifier()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=5, random_state=seed)
	cv_results = model_selection.cross_val_score(model, feature_data_X, target_data_Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)



#boxplot algorithm comparison

fig = plt.figure()
fig.suptitle('Model evaluation for Justice level')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


#####################Evaluation in Case level#######################################
feature_df_gnb_case = pd.read_csv("/home/ahlem/Text_Mining_Project/preprocessed_data_caseLevel.csv", encoding = "ISO-8859-1").astype(int)
feature_data_X_case = feature_df_gnb_case[['term', 'natural_court', 'argument_month', 'decision_month',
                                                            'decision_delay', 'justice',
                                                            'petitioner', 'respondent', 'jurisdiction', 'adminAction',
                                                            'caseSource',
                                                            'caseOrigin', 'certReason', 'lc_case_decision',
                                                            'issue', 'issueArea', 'lawType', 'lawSupp',
                                                            'petitionerState', 'respondentState',
                                                            'partyWinning', 'majVotes']]
target_data_Y_case = feature_df_gnb_case["case_level_decision"]

#comparision
models = []
seed=7
models.append(('AdaBoostClassifier', AdaBoostClassifier()))
models.append(('GaussianNB', GaussianNB()))
models.append(('RandomForestClassifier', RandomForestClassifier()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=5, random_state=seed)
	cv_results = model_selection.cross_val_score(model, feature_data_X_case, target_data_Y_case, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Model evaluation for Case level')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
