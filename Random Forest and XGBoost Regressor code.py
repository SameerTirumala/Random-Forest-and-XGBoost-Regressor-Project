# --- DATA PREPROCESSING ---


# import pandas to load in the data set that I have downloaded locally and preview it
# import train_test_split so that we can split the data ASAP and avoid leakage
import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv("C:/Users/tirum/Downloads/train.csv", index_col = 'Id')
data.head()


# get a lay of the land in terms of features, null values, etc.
data.info()


# Deem features that have a significant amount of null values (25% or more) unreliable and drop them
too_many_nulls_cols = [col for col in data.columns if data[col].isnull().sum()/data[col].count() > .25]
data.drop(too_many_nulls_cols, axis = 1, inplace = True)

data.info()


# set target variable (SalePrice of a house) and remove it from predictive features
# split the data into training and testing subsets
data_y = data.SalePrice
data_X = data.drop('SalePrice', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, train_size=0.8, test_size=0.2,
                                                    random_state=0)


# generate a list of the numeric features and a list of the categorical features from the training data
# Because I plan to encode the categorical features, it is best to only choose categorical columns where the test data 
# does not take on values other than what its training data counterpart does
numeric_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
object_cols = [col for col in X_train.select_dtypes(include = 'object').columns if set(X_test[col]).issubset(X_train[col])]

# only these features are relevant, so filter out other columns from both train and test data
X_train = X_train[numeric_cols + object_cols]
X_test = X_test[numeric_cols + object_cols]

# output descriptive statistics of the numeric columns
X_train[numeric_cols].describe()


# Distributions of numeric features can vary
# for columns whose standard deviations are very high (let's say 50% or more of the mean), imputing the mean doesn't seem ideal
# and since the 1st + 3rd quartiles of some columns seem evenly distributed around the median, imputing the median could be useful there
low_stdev_cols = [col for col in numeric_cols if X_train[col].std()/X_train[col].mean()<.5]
high_stdev_cols = list(set(numeric_cols) - set(low_stdev_cols))


# To decide how we should encode categorical variables, we should investigate cardinality. If the features are high cardinality, it will 
# be impractical to use one hot encoding (ie. dummy variables), and ordinal encoding (ie. labels) would be better
X_train.nunique().sort_values(ascending = False)


# Now we have grouped the features based on what preprocessing is relevant for them
# We can import the relevant preprocessing libraries (including pipelines and transformers, to make our lives easier)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

# and transform the data based on what type of data (numeric or categorical) and its measures of central tendency (mean or median)
numerical_low_stdev_transformer = SimpleImputer()
numerical_high_stdev_transformer = SimpleImputer(strategy = 'median')

# our investigation of cardinality suggests we are better off encoding labels than dummy variables
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('labels', OrdinalEncoder())
])

Impute_and_encode = ColumnTransformer(
    transformers=[
        ('num_mean', numerical_low_stdev_transformer, low_stdev_cols),
        ('num_median', numerical_high_stdev_transformer, high_stdev_cols),
        ('categ', categorical_transformer, object_cols)
    ])

# For the feature engineering techniques we plan to do like Principal Component Analysis, features that are 
# measured with larger units (such as LotArea) may receive bias in variance-based techniques. 
# To work around this, we can standardize them into their z-scores
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# this final processing pipeline will impute missing values (and for categorical values, encode them as numeric labels),
# then scale all the now-numeric data
preprocessor = Pipeline(steps=[('Cleaning', Impute_and_encode),
                      ('Scaling', scaler)
                     ])

# Create a list of column names because our processing pipeline returns an array, which we will convert back to a dataframe
# Get a lay of the land to confirm that there are no more missing values and that all columns are numeric and standardized
# We're now working with 71 features
col_list = X_train.columns
X_train = pd.DataFrame(preprocessor.fit_transform(X_train),  columns = col_list)
X_train.info()
X_train.describe()


# Now we're good to transform the test data
X_test = pd.DataFrame(preprocessor.transform(X_test),  columns = col_list)


# And create a basic random forest before we try to reduce dimensionality
from sklearn.ensemble import RandomForestRegressor
basic_model = RandomForestRegressor(random_state = 0)

basic_model.fit(X_train, y_train)

# The Kaggle course I got this data set from evaluates data using mean absolute error, ie. the absolute value of the average residual
# This seems to be the most practical metric because the absolute average residual is how off we are in our predictions of sale price.
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, basic_model.predict(X_test))

# This basic random forest will serve as our benchmark: an average residual ranging from 17000 to 17500.


# --- DIMENSIONALITY REDUCTION ---


# Now we get into our feature engineering. The two broad types of techniques I want to explore are after the model is fit (Feature Importances)
# And before the model is fit (Principal Component Analysis and Recursive Feature Elimination)
# The default form of feature importances uses mean decrease in impurity, so the only extra libraries we need are for plotting the importances
import matplotlib.pyplot as plt
#%matplotlib inline

# I chose to convert them into a series that allows for an index of feature names
# This is so that rather than plotting the MDI value for a given feature index, it easily plots the MDI value for a given feature name
MDI_importances = pd.Series(basic_model.feature_importances_, index = X_train.columns).sort_values()

fig, ax = plt.subplots()
ax.barh(MDI_importances.index, MDI_importances)
ax.set_title("Feature importances using MDI")
ax.set_xlabel("Mean decrease in impurity")
ax.set_ylabel("Feature")
fig.set_figheight(15)



# However, sklearn documentation warns that impurity-based techniques bias toward high cardinality features. We saw earlier
# that this data definitely contains such columns, so testing the permutation importance technique:
from sklearn.inspection import permutation_importance

permutation_result = permutation_importance(basic_model, X_test, y_test, n_repeats=10, random_state= 0, n_jobs=-1)

permutation_importances = pd.Series(permutation_result.importances_mean, index=X_train.columns).sort_values()

fig, ax = plt.subplots()
ax.barh(permutation_importances.index, permutation_importances)
ax.set_title("Feature importances using feature permutation")
ax.set_ylabel("Mean accuracy decrease")
ax.set_xlabel("Feature")
fig.set_figheight(15)


# As you can see, the results vary from technique to technique, likely because of the influence of high cardinality features.
# Reminding ourselves of the cardinality issue, we can see that many of the features considered relatively important by MDI are indeed
# high in cardinality, so it was worth exploring the permutation method.
X_train.nunique().sort_values(ascending = False)


# Going with the permutation importance method, we see that there is a steep drop-off between the most important feature and second most,
# second most and third most, and then a steady taper from there. Though it seems that removing any feature after the third does little harm,
# to accuracy, checking the mean absolute error using the next block of code showed me that they still do affect this metric
# I also have been experiencing a problem where if I restart my kernel, I receive different values for all mean absolute error calculations
# and feature importance methods. While I try to work around this, I will use the .tail(method) to get the last (ie. the most important features,
# since they were sorted ascending) 10 features. Though my outputs have been unstable, the bars on the plot generally shrink to 0.01 or less 
# after 9 to 12 features. For a nice even number, I will select the 10 most important using our permutation importance method.

most_important_permutation_features = permutation_importances.tail(10).sort_values(ascending = False).index.tolist()

# We call a print method to reality check so we can reality check whether the most important features would logically affect sale price.
print(most_important_permutation_features)

# Honestly speaking, there are times when I run the code and permutation_importance outputs logical features and other times when it 
# provides features that do not seem like they make sense in real life -- probably an overfitting problem.
permutation_importance_X_train = X_train[most_important_permutation_features]
permutation_importance_X_test = X_test[most_important_permutation_features]



# Now that we have selected only these 10 features from the overall training data, we will fit a random forest model with this subset.
permutation_importance_model = RandomForestRegressor(random_state= 0)
permutation_importance_model.fit(permutation_importance_X_train, y_train)
mean_absolute_error(y_test, permutation_importance_model.predict(permutation_importance_X_test))

# The mean absolute error this generates varied from 17500 to 19500, slightly unstable and worse than the basic model of all 70+ features.
# However, we are still getting a promising model after eliminating over 60 features.


# Now we move on to our second technique, Recursive Feature Elimination with Cross Validation (RFECV). The basic Sklearn Recursive Feature 
# Elimination (RFE) class requires the user to define a number of features to select. Its cross validation variant is appealing to maximize
# the potential of a small data set (we have roughly 1000 rows of training data), and because the data set is small, the computational
# cost is less of a concern. 
from sklearn.feature_selection import RFECV
rfecv = RFECV(
    estimator=RandomForestRegressor(random_state=0),
    min_features_to_select=5,
    n_jobs=-1,
    scoring="neg_mean_absolute_error")

rfecv.fit(X_train, y_train)
# Be warned, though, this cell still has a run time of around 3 minutes.


# Now that the rfecv object has been fit to the training data, we can index the training data column list with the object's support_ attribute 
# to return a list of the features rfecv finds to be relevant. We will print the list's length to see how useful this tool was.
print(X_train.columns[rfecv.support_])
print(len(X_train.columns[rfecv.support_]))

# In the times I have run the code, between 31 and 52 features have been found relevant. Even on the low end, not all of the 31 features 
# seem practically relevant at face value. Still, it is interesting that we were able to reduce 20 to 40 features of the training data.


# Moving on, we will select from the training and testing data the subset of features rfecv returned.
rfecv_X_train = X_train[X_train.columns[rfecv.support_]]
rfecv_X_test = X_test[X_train.columns[rfecv.support_]]


# And fit a random forest model with that subset of features.
rfecv_model = RandomForestRegressor(random_state= 0)
rfecv_model.fit(rfecv_X_train, y_train)
mean_absolute_error(y_test, rfecv_model.predict(rfecv_X_test))

# The absolute average residual from rfecv ranges from 17200 to 17900, with 17200 occurring when 50+ features are kept.
# Considering that we are removing a solid amount of features, the rfecv library definitely improved upon the basic model. However, I highly doubt,
# even considering the minimum 31 features, that all of them meaningfully contribute to the model-- we probably have not eliminated all the noise.


# Principal component analysis is the other dimensionality reduction technique we will attempt. Importing the relevant library, instantiating 
# a pca object, and fitting it to the data provides us with how much of the data's variance we explain with n components:
from sklearn.decomposition import PCA
pca = PCA(random_state=0)
pca.fit(X_train)
exp_variance = pca.explained_variance_ratio_
# We will plot the explained variance vs number of components included on a scree plot:
fig, ax = plt.subplots()
ax.bar(range(1, pca.n_components_ + 1),exp_variance)
ax.set_xlabel('Number of Principal Components')
fig.set_figwidth(10)

# We see a drastic drop off after the first component, which in some sense validates the MDI & permutation feature importances we checked earlier.


# However, we will not be able to build a strong model with one feature, so let us examine a cumulative explained variance plot.
# For this plot, we also draw a horizontal line representing the minimum variance we want our model to explain. In my reading, I have found 85% 
# to be the convention. Where the cumulative explained variance intersects the 85% threshold is the number of components we should include.
fig, ax = plt.subplots()
ax.plot(pca.explained_variance_ratio_.cumsum())
ax.axhline(y=0.85, linestyle='--')

# It looks like 38 or 39 is the optimal number. This is relatively consistent with our rfecv method, so I am skeptical for the same reasons.


# We will instantiate a pca object to select the 38 most important components, fit to and transform the training data, and transform the test data.
pca = PCA(n_components=38, random_state=0)
pca_X_train = pca.fit_transform(X_train)
pca_X_test = pca.transform(X_test)

#We are ready to instantiate a random forest model, fit it to our pca-reduced training data, and predict the sale prices of our testing data.
pca_model = RandomForestRegressor(random_state= 0)
pca_model.fit(pca_X_train, y_train)
mean_absolute_error(y_test, pca_model.predict(pca_X_test))

# The mean absolute error here ranged from 18500 to 19000, not bad for reducing 30+ features.


# Out of curiosity, I tested 31 features, because that was the lowest number rfecv recommended. 
pca = PCA(n_components=31, random_state=0)
pca_X_train = pca.fit_transform(X_train)
pca_X_test = pca.transform(X_test)

pca_model.fit(pca_X_train, y_train)
pca_predictions = pca_model.predict(pca_X_test)
mean_absolute_error(y_test, pca_predictions)

# In some of my trials running this code, I actually got a lower error from 31 components than 38.
# This again suggests that we have not fully understood the random noise of the data.


# --- PARAMETER TUNING ---


# We now have three models that reduced our data's dimensionality and still appear viable according to our error metric.
# The next step is to improve the model itself by tuning its parameters. In my reading, and experimentation with this data set specifically,
# the max_depth and max_leaf_nodes arguments did not affect the performance of these models much. Since we are only playing around with
# the n_estimators argument, there is no need to use GridSearchCV and we can simply define a function that instantiates a random forest
# with the user-entered number of estimators, fit it to the permutation importances, rfecv, and pca reduced training data, and measure the error 
# of their predictions.

def get_mae(n_estimators, X_train, X_test, y_train = y_train, y_test = y_test):
    rf_model = RandomForestRegressor(n_estimators, random_state = 0)
    rf_model.fit(X_train, y_train)
    return mean_absolute_error(y_test, rf_model.predict(X_test))

# We will iterate through a set of n_estimators options (ie. the number of decision trees in our random forest model) and compare error values:

for i in [25, 50, 75, 100, 250, 500, 1000]:
    print("\n Using %d trees: Permutation Importance Model's MAE = " % i, get_mae(i, permutation_importance_X_train, 
            permutation_importance_X_test), ", RFECV Model's MAE = ", get_mae(i, rfecv_X_train, rfecv_X_test), 
        ", and PCA Model's MAE = ", get_mae(i, pca_X_train, pca_X_test))

# The PCA model performed the worst at all n_estimator values while the rfecv model performed the best at all values.
# 1000 seems to be the optimal number of n_estimators, and it may be worth re-testing for even higher values. I chose not to because the
# computational cost increases significantly and any more than 500 trees generally takes a while to run.

# The final result is that with 1000 trees, the rfecv reduced data yields a mean absolute error from 17000 to 17100, much more consistent
# than other models run in this project.


# The final technique I plan to test is extreme gradient boosting (partially because I want to practice a flashy new model!).
# XGBoosting begins by fitting and predicting with a single, simple model-- the first of the ensemble. 
# It then performs gradient descent on the loss function of the previous model to fit a new model, which is added to the ensemble.
# This process is repeated, and each model added to the ensemble after the first reduces the overall ensemble's loss.
 
# We can improve xgboost model performance with parameters like:
# 1. n_estimators (in this case, the number of models ie. iterations through the process) 
# 2. learning rate (I think of this like adjusted R^2 for linear regressions, imposing a penalty on overfitting by multiplying each
# iteration added to the ensemble by a constant so that the next estimator helps you less than the one before it)
# 3. early stopping rounds (this is what ties it all together-- the model will choose the optimal number of estimators, stopping before 1000
# specified here if an evaluation metric does not improve for a user-defined number of rounds)

# A high n_estimators along with low learning rate work well with early stopping rounds. It will stop early if that is what the data calls for,
# but also will go up to a high number of models in the ensemble if needed-- more thorough than the get_mae function I defined before.

from xgboost import XGBRegressor

boosted_model = XGBRegressor(n_estimators = 1000, learning_rate = .05, early_stopping_rounds = 5, random_state = 0)
boosted_model.fit(rfecv_X_train, y_train, eval_set = [(rfecv_X_test, y_test)], verbose = False)
mean_absolute_error(y_test, boosted_model.predict(rfecv_X_test))

# The result for this xgboost model of rfecv features is promising, with mean absolute error results as low as the 16900s.
# However, low error values often occur when rfecv selects more features (in the 50s rather than the 30s)
# Also, when rerunning the code, I have seen error values up to the 17800s.


# Fitting the model to the features selected by permutation importances instead:
boosted_model.fit(permutation_importance_X_train, y_train, eval_set = [(permutation_importance_X_test, y_test)], verbose = False)
mean_absolute_error(y_test, boosted_model.predict(permutation_importance_X_test))

# This model also changes noticeably. When permutation importance determined a robust set of 9 to 10 features, I received a mean absolute error
# of about 16500-- the lowest of any model I have built on this data set, even outside of this code. However, I have also gotten 17500.


# --- DISCUSSION ---


# This project was instructive on the functionalities I chose, and it revealed their limitations as well.

# The biggest flaw I find in my code is that some aspect of it changes outputs (all error metrics and feature importances calculated throughout).
# I have set consistent random_state parameters when applicable and experimented with creating new dataframes for each preprocessing step that
# may have distorted the X_train or X_test dataframes, but none of these have solved the issue.

# As for unexplored topics, perhaps so many features were chosen or very few seemed important because collinearity existed. 
# Additionally, I standardized the data but maybe it would be worth controlling for outliers before standardizing.

# Finally, we must consider that rather than a null value being unreliable data where no value was entered, 
# it might simply be a zero value. Imputing 0 rather than the variable's mean or median values would then be the correct way to deal with nulls.


