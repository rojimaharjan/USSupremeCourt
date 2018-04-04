# Imports
import datetime
import numpy
import pandas
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
import sys
import warnings

# Methods

def get_outcome_map():
    """
    Get the outcome map to convert an SCDB outcome into
    an affirm/reverse/other mapping.

    Rows correspond to vote types.  Columns correspond to disposition types.
    Matrix Element values correspond to:
    * 0: affirm --> no change in precedent decision
    * 1: reverse --> change in precedent decision
    * -1: other --> any decision other than affirm or reverse

    """

    """
    Create the map in reference to the paper related to our topic:
    A general approach for predicting the behavior of the Supreme Court of the United
    States
    
    """

    outcome_map = pandas.DataFrame([[-1, 0, 1, 1, 1, 0, 1, -1, -1, -1, -1],
                                    [-1, 1, 0, 0, 0, 1, 0, -1, -1, -1, -1],
                                    [-1, 0, 1, 1, 1, 0, 1, -1, -1, -1, -1],
                                    [-1, 0, 1, 1, 1, 0, 1, -1, -1, -1, -1],
                                    [-1, 0, 1, 1, 1, 0, 1, -1, -1, -1, -1],
                                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                    [-1, 0, 0, 0, -1, 0, -1, -1, -1, -1, -1]])

    outcome_map.columns = range(1, 12)
    outcome_map.index = range(1, 9)

    return outcome_map


def get_outcome(vote, disposition, outcome_map=None):
    """
    Return the outcome code based on outcome map.
    """
    if not outcome_map:
        SCDB_OUTCOME_MAP = get_outcome_map()
        outcome_map = SCDB_OUTCOME_MAP

    if pandas.isnull(vote) or pandas.isnull(disposition):
        return -1

    return outcome_map.loc[int(vote), int(disposition)]


def get_date(value):
    """
    Get date value from SCDB string format.
    """
    try:
        return datetime.datetime.strptime(value, "%m/%d/%Y").date()
    except:
        return None


def get_date_month(value):
    """
    Get month from date.
    """
    try:
        return value.month
    except:
        return -1

def as_column_vector(values):
    # Return values as column vector
    return numpy.array(values, ndmin=2).T

# Compute the feature importance

def feature_importance(data):
    X = data[['term', 'natural_court', 'argument_month', 'decision_month',
                   'decision_delay', 'justice',
                   'petitioner', 'respondent', 'jurisdiction', 'adminAction', 'caseSource',
                   'caseOrigin', 'lcDisagreement', 'certReason', 'lc_case_decision',
                   'issue', 'issueArea', 'lawType', 'lawSupp', 'petitionerState', 'respondentState',
                   'partyWinning', 'majVotes']]
    Y = data.loc[:, "justice_level_decision"]

    # Build the classifier
    model = XGBClassifier()
    model.fit(X, Y)

    # plot feature importance
    plot_importance(model)
    pyplot.show()


def preprocess_data_justiceLevel(data):

    """
    Preprocess the SCDB:"supreme court database" data.

    """

    # Define Labels. Here we have two labels: The justice_level decision
    outcome_map = get_outcome_map()
    justice_level_decision = pandas.DataFrame(data.loc[:, ("vote", "caseDisposition")] \
        .apply(lambda row: get_outcome(row["vote"], row["caseDisposition"]), axis=1))

    # Get chronological variables
    term = as_column_vector(data.loc[:, "term"].fillna(-1))
    natural_court= as_column_vector(data.loc[:, "naturalCourt"].fillna(-1))

    # Get argument and decision dates
    argument_date_raw = data.loc[:, "dateArgument"].apply(get_date)
    decision_date_raw = data.loc[:, "dateDecision"].apply(get_date)

    argument_month_raw = argument_date_raw.apply(get_date_month).astype(int).values
    decision_month_raw = decision_date_raw.apply(get_date_month).astype(int).values

    decision_delay = ((decision_date_raw - argument_date_raw) / numpy.timedelta64(1, 'W')) \
        .fillna(-1) \
        .astype(int)

    # Get justice identification variables
    justice = as_column_vector(data.loc[:, "justice"].fillna(-1))

    # Get petitioner and respondent id and state
    petitioner = as_column_vector(data.loc[:, "petitioner"].fillna(-1))
    respondent = as_column_vector(data.loc[:, "respondent"].fillna(-1))
    petitionerState = as_column_vector(data.loc[:, "petitionerState"].fillna(-1))
    respondentState = as_column_vector(data.loc[:, "respondentState"].fillna(-1))

    # Get the case information
    jurisdiction = as_column_vector(data.loc[:, "jurisdiction"].fillna(-1))
    adminAction = as_column_vector(data.loc[:, "respondent"].fillna(-1))
    caseSource = as_column_vector(data.loc[:, "caseSource"].fillna(-1))
    caseOrigin = as_column_vector(data.loc[:, "caseOrigin"].fillna(-1))

    certReason = as_column_vector(data.loc[:, "certReason"].fillna(-1))
    issue = as_column_vector(data.loc[:, "issue"].fillna(-1))
    issueArea = as_column_vector(data.loc[:, "issueArea"].fillna(-1))
    lawType = as_column_vector(data.loc[:, "lawType"].fillna(-1))
    lawSupp = as_column_vector(data.loc[:, "lawSupp"].fillna(-1))

    # Get the lower court decision information
    lc_case_decision = pandas.DataFrame(outcome_map.loc[1, data.loc[:, "lcDisposition"]].values).fillna(-1)
    #lcDisagreement = as_column_vector(data.loc[:, "lcDisagreement"].fillna(-1))

    # Get information about the final outcome of the case
    partyWinning = as_column_vector(data.loc[:, "partyWinning"].fillna(-1))
    majVotes = as_column_vector(data.loc[:, "majVotes"].fillna(-1))


    # Create the final data for the justice level decision
    preprocessed_data = numpy.hstack((term, natural_court,as_column_vector(argument_month_raw),as_column_vector(decision_month_raw),
                                 as_column_vector(decision_delay), justice,
                                 petitioner, respondent,jurisdiction, adminAction, caseSource,
                                 caseOrigin,#lcDisagreement,
                                    certReason, lc_case_decision,
                                 issue, issueArea, lawType, lawSupp, petitionerState, respondentState,
                                 partyWinning,majVotes,justice_level_decision))

    preprocessed_data_labels = ["term"]
    preprocessed_data_labels.append("natural_court")
    preprocessed_data_labels.append("argument_month")
    preprocessed_data_labels.append("decision_month")
    preprocessed_data_labels.append("decision_delay")
    preprocessed_data_labels.append("justice")
    preprocessed_data_labels.append("petitioner")
    preprocessed_data_labels.append("respondent")
    preprocessed_data_labels.append("jurisdiction")
    preprocessed_data_labels.append("adminAction")
    preprocessed_data_labels.append("caseSource")
    preprocessed_data_labels.append("caseOrigin")
    #preprocessed_data_labels.append("lcDisagreement")
    preprocessed_data_labels.append("certReason")
    preprocessed_data_labels.append("lc_case_decision")
    preprocessed_data_labels.append("issue")
    preprocessed_data_labels.append("issueArea")
    preprocessed_data_labels.append("lawType")
    preprocessed_data_labels.append("lawSupp")
    preprocessed_data_labels.append("petitionerState")
    preprocessed_data_labels.append("respondentState")
    preprocessed_data_labels.append("partyWinning")
    preprocessed_data_labels.append("majVotes")
    preprocessed_data_labels.append("justice_level_decision")

    preprocessed_df_justice = pandas.DataFrame(preprocessed_data,
                                  columns=preprocessed_data_labels)

    # At last, return
    return preprocessed_df_justice

def preprocess_data_caseLevel(data, df):

    # Define Labels. Here we have two labels: The case_level decision
    outcome_map = get_outcome_map()
    case_level_decision = outcome_map.loc[1, data.loc[:, "caseDisposition"]].values
    case_level_decision = pandas.DataFrame(case_level_decision)
    preprocessed_df_case = df
    preprocessed_df_case['justice_level_decision']= case_level_decision
    preprocessed_df_case.rename(columns={'justice_level_decision': 'case_level_decision'}, inplace = True)
    preprocessed_df_case = preprocessed_df_case[numpy.isfinite(df['case_level_decision'])]

    return preprocessed_df_case

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Perform preprocessing
data = pandas.read_csv("/home/ahlem/TMP/Database/SCDB_2017_01_justiceCentered_Citation.csv", encoding="ISO-8859-1")
preprocessed_data_justice= preprocess_data_justiceLevel(data)
preprocessed_data_justice.to_csv("/home/ahlem/Text_Mining_Project/preprocessed_data_justiceLevel.csv",index=False)
#feature_importance(data)
preprocessed_data_case= preprocess_data_caseLevel(data,preprocessed_data_justice)
preprocessed_data_case.to_csv("/home/ahlem/Text_Mining_Project/preprocessed_data_caseLevel.csv",index=False)



