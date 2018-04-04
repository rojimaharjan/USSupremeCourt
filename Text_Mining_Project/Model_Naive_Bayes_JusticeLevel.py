#Imports
import numpy
from Methods import *
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
import sys
import warnings
import matplotlib.pyplot as plt




#Getting the data frame and the features frame resulting from the preprocessing
feature_df_gnb = pd.read_csv("/home/ahlem/Text_Mining_Project/preprocessed_data_justiceLevel.csv", encoding = "ISO-8859-1").astype(int)

#getting the raw data for plotting the classes according to case_name
# raw_justice_data=pd1.read_csv(r'C:\Users\Admin\Documents\TMP\data.csv',encoding = "ISO-8859-1")

# Setup training time period
min_training_years = 5
term_range = range(feature_df_gnb["term"].min() + min_training_years,
                   feature_df_gnb["term"].max() + 1)

 # Setup model


model = None
term_count = 0
class_counts = pd.DataFrame(columns=['term','class0','class1','class-1'])

for term in term_range:
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    # Diagnostic output
    print("Term: {0}".format(term))
    term_count += 1

    # Setup train and test periods
    train_index_gnb = (feature_df_gnb.loc[:, "term"] < term).values
    test_index_gnb = (feature_df_gnb.loc[:, "term"] == term).values

    # Setup train data
    feature_data_train_X = feature_df_gnb.loc[train_index_gnb,['term', 'natural_court', 'argument_month', 'decision_month',
               'decision_delay', 'justice',
               'petitioner', 'respondent', 'jurisdiction', 'adminAction', 'caseSource',
               'caseOrigin', 'certReason', 'lc_case_decision',
               'issue', 'issueArea', 'lawType', 'lawSupp', 'petitionerState', 'respondentState',
               'partyWinning', 'majVotes']]
    target_data_train_Y = feature_df_gnb.loc[train_index_gnb, "justice_level_decision"]

    # Setup test data
    feature_data_test_X = feature_df_gnb.loc[test_index_gnb ,['term', 'natural_court', 'argument_month', 'decision_month',
               'decision_delay', 'justice',
               'petitioner', 'respondent', 'jurisdiction', 'adminAction', 'caseSource',
               'caseOrigin', 'certReason', 'lc_case_decision',
               'issue', 'issueArea', 'lawType', 'lawSupp', 'petitionerState', 'respondentState',
               'partyWinning', 'majVotes']]
    target_data_test_Y = feature_df_gnb.loc[test_index_gnb, "justice_level_decision"]

    #build the model
    model=GaussianNB()
    # Fit the model
    model.fit(feature_data_train_X,
          target_data_train_Y)

     # Perform predictions
    feature_df_gnb.loc[test_index_gnb, "gnb_predicted"] = model.predict(feature_data_test_X)
    current_term_data = feature_df_gnb.loc[feature_df_gnb["term"] == term]
    count_0 = len(current_term_data.loc[current_term_data["gnb_predicted"] == 0].index)
    count_1 = len(current_term_data.loc[current_term_data["gnb_predicted"] == 1].index)
    count_other = len(current_term_data.loc[current_term_data["gnb_predicted"] == -1].index)
    df = pd.DataFrame([[term, count_0, count_1, count_other]], columns=['term', 'class0', 'class1', 'class-1'])
    class_counts = class_counts.append(df, ignore_index=True)


    # mean accuracy for test data and labels
    mean = model.score(feature_data_test_X, target_data_test_Y)
    print(mean)


    # Store scores per class
    scores = model.predict_proba(feature_data_test_X)
    feature_df_gnb.loc[test_index_gnb, "gnb_predicted_score_other"] = scores[:, 0]
    feature_df_gnb.loc[test_index_gnb, "gnb_predicted_score_affirm"] = scores[:, 1]
    feature_df_gnb.loc[test_index_gnb, "gnb_predicted_score_reverse"] = scores[:, 2]


# Evaluation
evaluation_index_gnb = feature_df_gnb.loc[:, "term"].isin(term_range)
target_actual_gnb = feature_df_gnb.loc[evaluation_index_gnb, "justice_level_decision"]
target_predicted_gnb = feature_df_gnb.loc[evaluation_index_gnb, "gnb_predicted"]
feature_df_gnb.loc[:, "gnb_correct"] = numpy.nan
feature_df_gnb.loc[evaluation_index_gnb, "gnb_correct"] = (target_actual_gnb == target_predicted_gnb).astype(float)

# Evaluate the model
print("Gaussian Naive Bayes")
print("="*50)
print(classification_report(target_actual_gnb, target_predicted_gnb,digits=2))
print(confusion_matrix(target_actual_gnb, target_predicted_gnb))
print(accuracy_score(target_actual_gnb, target_predicted_gnb))
print("="*50)

cnf_matrix = confusion_matrix(target_actual_gnb, target_predicted_gnb)
np.set_printoptions(precision=2)

#Plot non-normalized confusion matrix
plt.figure()
class_names= ["other","affirm","reverse"]
plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')


#plot the values of the classes given the term for justice centric
plot_class_count=pd.DataFrame(class_counts,columns=["term","class0","class1","class-1"])
fig,ax = plt.subplots()
plot_class_count.plot(x="term", y=["class0", "class1", "class-1"], ax=ax, ls="--")
plt.show()





