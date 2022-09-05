######### Libraries and Some Settings #########

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

###################  METHODS  #######################

# to determine the outlier tresholds
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# to determine categorical, numeric, categorical but cardinal variables
def grab_col_names(dataframe, cat_th=10, car_th=20):
    
    """
    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included in categorical variables.
    Parameters
    ------
        dataframe: dataframe
                The dataframe from which variable names are to be retrieved
        cat_th: int, optional
                Class threshold for numeric but categorical variables
        car_th: int, optinal
                Class threshold for categorical but cardinal variables
    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        cat_but_car: list
                Categorical view cardinal variable list
    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))
    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.
        The sum of 3 lists with return is equal to the total number of variables: cat_cols + num_cols + cat_but_car = number of variables
    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df = load()
df.head()
df.columns = [col.upper() for col in df.columns]

df.shape
df.describe

########################################################
##############  1. FEATURE ENGINEERING  ################
########################################################

# if the cabin value is not null, NEW_CABIN_BOOL is 1.
# if the cabin value is null, NEW_CABIN_BOOL is 0.
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')

df.groupby("NEW_CABIN_BOOL").agg({"SURVIVED": "mean"})
# There is a marked difference between the mean of the "SURVIVED" variable when the "NEW_CABIN_BOOL" variable is 0 ++
# ++ and the mean of the "SURVIVED" variable when the "NEW_CABIN_BOOL" variable is 1.

# Let's look at if this difference is significant or not
## H0: ...there is no significant difference.
## H1: ...there is a significant difference.

from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "SURVIVED"].sum(),
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "SURVIVED"].sum()],

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "SURVIVED"].shape[0],
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "SURVIVED"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Because of the p-value < 0.05, H0 is rejected. There is a significant difference between the boolean values of "NEW_CABIN_BOOL" variable.
# "NEW_CABIN_BOOL" can be significant feature.

# -----------------------------------------------------------------------

df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"SURVIVED": "mean"})
# There is a marked difference between the mean of the "SURVIVED" variable when the "NEW_IS_ALONE" variable is "YES" ++
# ++ and the mean of the "SURVIVED" variable when the "NEW_IS_ALONE" variable is "NO".

# Let's look at if difference is significant or not
## H0: ...there is no significant difference.
## H1: ...there is a significant difference.

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "SURVIVED"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "SURVIVED"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "SURVIVED"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "SURVIVED"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Because of the p-value < 0.05, H0 is rejected. There is a significant difference between the values of "NEW_IS_ALONE" variable.
# "NEW_IS_ALONE" can be significant feature.

# -----------------------------------------------------------------------

# Name count
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
# name word count
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
# name dr
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# name title
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
# family size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
# age_pclass
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
# is alone
df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
# age level
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
# sex x age
df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & ((df['AGE'] > 21) & (df['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & ((df['AGE'] > 21) & (df['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# removing "PASSENGERID" from numerical columns, because of this is not exactly numerical variable.
num_cols = [col for col in num_cols if "PASSENGERID" not in col]

########################################################
###################  2. OUTLIERS  ######################
########################################################

# the method checks if there are outliers in the numeric columns of the dataset.
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# finding numerical columns in the dataframe
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# checking if there is outlier in the numerical columns
for col in num_cols:
    print(col, check_outlier(df, col))

# It can be seen with also boxplot if there is outlier or not
sns.boxplot(x=df["AGE"])
plt.show()

sns.boxplot(x=df["FARE"])
plt.show()

################ 1.1. SOLVING THE OUTLIER PROBLEM (RE-ASSIGNMENT WITH THRESHOLDS)  ################

# The method changes the outliers in the dataframe with threshold values
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# changing the outliers in the numerical columns of the dataframe with thresholds
for col in num_cols:
    replace_with_thresholds(df, col)

# checking again if there is outlier after re-assignment
for col in num_cols:
    print(col, check_outlier(df,col))

########################################################
################  3. MISSING VALUES  ###################
########################################################

# checking if there are missing values in the dataframe
df.isnull().values.any()

# checking how many missing values are
df.isnull().sum()

# the method returns missing values's ratios in the dataframe and how many missing values are
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df, True)

df.drop("CABIN", inplace=True, axis=1)

remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)
df.head()

# looking the affect of the "NEW_TITLE" variables on the "SURVIVED" and "AGE" variables
df[["NEW_TITLE", "SURVIVED", "AGE"]].groupby(["NEW_TITLE"]).agg({"SURVIVED": "mean", "AGE": ["count", "mean"]})

# filling the missing values of "AGE" variable with "NEW_TITLE" variable's median values
df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

#########  RECREATING FEATURES DERIVED FROM "AGE" VARIABLES  ##########
# Beacuse, the missing values of "AGE" variable filled in the previous row with "NEW_TITLE" variable's median values.

df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & ((df['AGE'] > 21) & (df['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & ((df['AGE'] > 21) & (df['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

# filling the missing values of "EMBARKED" variable with it's mode
df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

########################################################
################  4. LABEL ENCODING  ###################
########################################################

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

# finding variables that have only 2 different values.
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

# applying label encoder on these variables that have only 2 different values
for col in binary_cols:
    df = label_encoder(df, col)

########################################################
#################  5. RARE ENCODING  ###################
########################################################

# finding numerical columns in the dataframe
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Finding the ratio, the frequency of all categorical variables in the dataset and it's status in terms of target.
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

# determining which values of variables is collected as "rare"
def rare_encoder(dataframe, rare_perc, cat_cols):
    # if the ratio is smaller than rare_per and the count is bigger than 1, it is collected as "rare"
    # if the ratio is smaller than rare_per and the count is equal to 1, it is NOT done anyting.
    rare_columns = [col for col in cat_cols if (dataframe[col].value_counts() / len(dataframe) < 0.01).sum() > 1]

    for col in rare_columns:
        tmp = dataframe[col].value_counts() / len(dataframe)
        rare_labels = tmp[tmp < rare_perc].index
        dataframe[col] = np.where(dataframe[col].isin(rare_labels), 'Rare', dataframe[col])

    return dataframe

# Let's control which ones can collect as "rare".
rare_analyser(df, "SURVIVED", cat_cols)

# Collecting the some values of categorical variables as "rare"
df = rare_encoder(df, 0.01, cat_cols)

# Let's control AGAIN which ones were collected as "rare".
rare_analyser(df, "SURVIVED", cat_cols)

########################################################
################  6. ONE-HOT ENCODING  #################
########################################################

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

# finding categorical columns with unique values greater than 2 and less than 10. They are ohe_cols
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols) ## değişkenler üretildi
df.head()
df.shape

# --------------------------------------------------

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# removing "PASSENGERID" from numerical columns
num_cols = [col for col in num_cols if "PASSENGERID" not in col]

# Finding if all variables derived with one-hot encoding is significant or not
rare_analyser(df, "SURVIVED", cat_cols)

# Finding "useless columns" which has only unique 2 values and the ratio in terms of 'target' is smaller than 0.01
useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

# removing useless columns from dataset
df.drop(useless_cols, axis=1, inplace=True)

#######################################################
#################  7. ROBUST SCALER  ##################
#######################################################

# Applying "Scale" process to evaluate all variables at the same conditions

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])
df.describe().T

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#######################################################
#####################  8. MODEL  ######################
#######################################################

y = df["SURVIVED"]  # the dependent variable
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)  # the independent variables

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test) ## THE ACCURACY = % 81

############# COMMENT ##############
# Their survivability is predicted with 81% accuracy by using the characteristics of those who have boarded this ship,
####################################

# drawing a graph showing the importance of the produced variables.
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_train, num=30)

#########################################
#############  IN ADDITION  #############
#########################################

# What will be the score if the model is applied directly to the raw data frame without any of the above operations?

dff = load()
dff.dropna(inplace=True)
dff = pd.get_dummies(dff, columns=["Sex", "Embarked"], drop_first=True)
y = dff["Survived"]
X = dff.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)  # THE ACCURACY =  % 70.9

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_train, num=10)
