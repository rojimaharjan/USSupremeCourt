#Imports
import numpy
import pandas
import sklearn.ensemble
from Methods import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sys
import warnings



#Getting the preprocessed data

data = pandas.read_csv("/home/ahlem/Text_Mining_Project/preprocessed_data_caseLevel.csv", encoding="ISO-8859-1").astype('int')

####################building the model based on the method used in the reference paper#####################################

# Setup training time period
min_training_years = 5
term_range = range(data["term"].min() + min_training_years,
                   data["term"].max() + 1)

# Setting the parameters of the growing random forest

# Number of trees to grow per term
trees_per_term = 10

# Number of trees to begin with
initial_trees = min_training_years * trees_per_term


# Setup model
m0 = None
m = None
term_count = 0
class_counts = pandas.DataFrame(columns=['term','class0','class1','class-1'])


for term in term_range:
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    # Diagnostic output
    print("Term: {0}".format(term))
    term_count += 1

    # Setup train and test periods
    train_index = (data.loc[:, "term"] < term).values
    test_index = (data.loc[:, "term"] == term).values

    # Setup train data
    feature_data_train = data.loc[train_index, ['term', 'natural_court', 'argument_month', 'decision_month',
                   'decision_delay', 'justice',
                   'petitioner', 'respondent', 'jurisdiction', 'adminAction', 'caseSource',
                   'caseOrigin', 'certReason', 'lc_case_decision',
                   'issue', 'issueArea', 'lawType', 'lawSupp', 'petitionerState', 'respondentState',
                   'partyWinning', 'majVotes']]
    target_data_train = data.loc[train_index, "case_level_decision"]

    # Setup test data
    feature_data_test = data.loc[test_index, ['term', 'natural_court', 'argument_month', 'decision_month',
                   'decision_delay', 'justice',
                   'petitioner', 'respondent', 'jurisdiction', 'adminAction', 'caseSource',
                   'caseOrigin', 'certReason', 'lc_case_decision',
                   'issue', 'issueArea', 'lawType', 'lawSupp', 'petitionerState', 'respondentState',
                   'partyWinning', 'majVotes']]
    target_data_test = data.loc[test_index, "case_level_decision"]

    # Check if we should rebuild the model based on the changement of the natural court
    if set(data.loc[data.loc[:, "term"] == (term - 1), "justice"].unique()) != \
            set(data.loc[data.loc[:, "term"] == (term), "justice"].unique()):
        # natural Court change; trigger forest fire
        print(
            "Natural court change; rebuilding with {0} trees".format(initial_trees + (term_count * trees_per_term)))

        m0 = None
        m = None

    # Build the model
    if not m:
        # Grow an initial forest and adaboost random forest model
        m0 = sklearn.ensemble.RandomForestClassifier(n_estimators=initial_trees + (term_count * trees_per_term),
                                                    class_weight="balanced_subsample",
                                                    warm_start=True,
                                                    n_jobs=-1)
        m = sklearn.ensemble.AdaBoostClassifier(base_estimator=m0, n_estimators=2, learning_rate=1,
                                                algorithm="SAMME")
    else:
        # Grow the forest by increasing the number of trees (requires warm_start=True)
        m.set_params(base_estimator=m0.set_params(n_estimators=initial_trees + (term_count * trees_per_term)))

    # Fit the adaboost random forest model
    m.fit(feature_data_train,
          target_data_train)

    # Perform forest predictions
    data.loc[test_index, "arf_predicted"] = m.predict(feature_data_test)
    current_term_data = data.loc[data["term"] == term]
    count_0 = len(current_term_data.loc[current_term_data["arf_predicted"] == 0].index)
    count_1 = len(current_term_data.loc[current_term_data["arf_predicted"] == 1].index)
    count_other = len(current_term_data.loc[current_term_data["arf_predicted"] == -1].index)
    df = pandas.DataFrame([[term, count_0, count_1, count_other]], columns=['term', 'class0', 'class1', 'class-1'])
    class_counts = class_counts.append(df, ignore_index=True)


# Evaluation range
evaluation_index = data.loc[:, "term"].isin(term_range)
target_actual = data.loc[evaluation_index, "case_level_decision"]
target_predicted = data.loc[evaluation_index, "arf_predicted"]

# Evaluate the model
print("Adaboost Random forest  model evaluation metrics")
print("="*32)
print(sklearn.metrics.classification_report(target_actual, target_predicted))
print(sklearn.metrics.confusion_matrix(target_actual, target_predicted))
print(sklearn.metrics.accuracy_score(target_actual, target_predicted))
print("="*32)

cnf_matrix = confusion_matrix(target_actual, target_predicted)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
class_names= ["other","affirm","reverse"]
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.show()

#plot the values of the classes given the term for justice centric
plot_class_count=pandas.DataFrame(class_counts,columns=["term","class0","class1","class-1"])
fig,ax = plt.subplots()
plot_class_count.plot(x="term", y=["class0", "class1", "class-1"], ax=ax, ls="--")
plt.show()
