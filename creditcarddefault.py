# Importing the needed libraries
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
from statstests.tests import overdisp
from statstests.process import stepwise
import statsmodels.api as sm
import statsmodels.formula.api as smf
%autoindent OFF  # Line exclusively for IDE use (neovim), needs to be taken off if used another IDE

# Load the database
full = pd.read_csv('csv/creditcards.csv')
#
# Lowercase for all columns, for the ease of use
full.columns = full.columns.str.lower()
#
# Changing 'sex' column (two variables) to dummy 'ismale'
mask = full['sex'] == 2
full.loc[mask, 'sex'] = 0
full = full.rename(columns={'sex': 'ismale'})
#
# Dealing with education, turning it to categories and adjusting data,
# grouping categories 0, 5 and 6 to 'unknown'
mask = (full['education'] == 5) | (full['education'] == 6) | (full['education'] == 0)
full.loc[mask, 'education'] = 'unknown'
mask = full['education'] == 4
full.loc[mask, 'education'] = 'others'
mask = full['education'] == 3
full.loc[mask, 'education'] = 'high_school'
mask = full['education'] == 2
full.loc[mask, 'education'] = 'university'
mask = full['education'] == 1
full.loc[mask, 'education'] = 'graduate_school'
full['education'] = full['education'].astype(str).astype('category')
#
# 'marriage' feature to categorical: (1=married, 2=single, 3=others)
# As with education, '0' will be unknown
mask = full['marriage'] == 0
full.loc[mask, 'marriage'] = 'unknown'
mask = full['marriage'] == 3
full.loc[mask, 'marriage'] = 'others'
mask = full['marriage'] == 2
full.loc[mask, 'marriage'] = 'single'
mask = full['marriage'] == 1
full.loc[mask, 'marriage'] = 'married'
full['marriage'] = full['marriage'].astype(str).astype('category')
#
# Rename all columns so it's more comprehensive:
full = full.rename(columns={'pay_0': 'pay_sep',
                            'pay_2': 'pay_aug',
                            'pay_3': 'pay_jul',
                            'pay_4': 'pay_jun',
                            'pay_5': 'pay_may',
                            'pay_6': 'pay_apr',
                            'bill_amt1': 'bill_sep',
                            'bill_amt2': 'bill_aug',
                            'bill_amt3': 'bill_jul',
                            'bill_amt4': 'bill_jun',
                            'bill_amt5': 'bill_may',
                            'bill_amt6': 'bill_apr',
                            'pay_amt1': 'paid_sep',
                            'pay_amt2': 'paid_aug',
                            'pay_amt3': 'paid_jul',
                            'pay_amt4': 'paid_jun',
                            'pay_amt5': 'paid_may',
                            'pay_amt6': 'paid_apr',
                            'default.payment.next.month': 'default_next',
                            })
#
# Outliers found in the monetary values, it's easier now to get hold
# of all the features to normalize later
mask = full.dtypes == 'float64'
names_to_norm = full.columns[mask]
#
# Transforming 'education' and 'marriage' into dummies
dummies = pd.get_dummies(full[['education', 'marriage']], drop_first=True, dtype=int)
#
# Create features: % of limit use and % total paid
mask = full.columns.str.find('bill') == 0
names_bill = full.columns[mask]
percent_bill = pd.DataFrame()
for col in names_bill:
    percent_bill[f'bill_pct_{col[-3:]}'] = full[col]/full['limit_bal']
#
percent_paid = pd.DataFrame()
percent_paid['paid_pct_apr'] = full['paid_apr']/full['bill_apr']
percent_paid['paid_pct_may'] = full['paid_may']/full['bill_may']
percent_paid['paid_pct_jun'] = full['paid_jun']/full['bill_jun']
percent_paid['paid_pct_jul'] = full['paid_jul']/full['bill_jul']
percent_paid['paid_pct_aug'] = full['paid_aug']/full['bill_aug']
percent_paid['paid_pct_sep'] = full['paid_sep']/full['bill_sep']
percent_paid = percent_paid.fillna(0)
# Treating also the inf values
percent_paid.replace(np.inf, np.nan, inplace=True)  # replace inf with nans
percent_paid = percent_paid.fillna(1)  # then fill those nans with 1
#
# Concatenating all datasets untill now
#full = pd.concat([full, percent_bill], axis=1)
#full = pd.concat([full, percent_paid], axis=1)
full = pd.concat([full, dummies], axis=1)
#
# Evaluating the customer default history
full['default_hist'] = (full[['pay_apr', 'pay_may', 'pay_jun', 'pay_jul', 'pay_aug', 'pay_sep']] > 0).sum(axis=1)
#
# Checking outliers in 'limit_bal'
#sns.boxplot(y=full['limit_bal'])
#plt.show()
#
# Now, yes, dealing with the normalization
temp = pd.DataFrame(normalize(full[names_to_norm], axis=1))
temp.columns = names_to_norm
#
# Creates a new dataframe with the normalized values
normalized = full.copy()
for col in temp.columns:
    normalized[col] = temp[col]
#
# Getting rid of the original features that were transformed in dummies
# and also the id feature
normalized = normalized.drop(['education', 'marriage', 'id'], axis=1)

# All data wrangling was done, now it's time to build up the classification model!

###
# Before the first model
###
# Let's check the mean and variation of our target variable
target = 'default_next'
normalized[target].mean()  # 0.2212
normalized[target].var()  # 0.1722

# They are not greatly away from one another, so we cannot infer at this point
# that we have a diagnostic for overdispersion. 

###
# First GLM test: Poisson Model
###

target = 'default_next'
columns_list = (normalized.drop(target, axis=1)).columns
model_formula = ' + '.join(columns_list)
model_formula = target + " ~ " + model_formula
print("This is the funcional form of our model:\n\n", model_formula)

model_poisson = smf.glm(formula=model_formula, data=normalized, family=sm.families.Poisson()).fit()
model_poisson.summary()

# Stepwise process to filter out irrelevant features
stepwise(model_poisson, pvalue_limit=0.05)

# So, filtering the irrelevant features:
irrelevant = ['bill_aug', 'bill_apr', 'age', 'bill_jun', 'bill_jul',
              'marriage_others', 'pay_apr', 'paid_jul']
columns_list = (normalized.drop(target, axis=1).drop(irrelevant, axis=1)).columns
model_formula2 = ' + '.join(columns_list)
model_formula2 = target + " ~ " + model_formula2
print("This is the funcional form of our model:\n\n", model_formula)

model_poisson2 = smf.glm(formula=model_formula2, data=normalized, family=sm.families.Poisson()).fit()
model_poisson2.summary()

# With a valid model, now let's test it for overdispersion (Cameron & Trivedi)
# Adding the estimated values to original dataframe
normalized['lambda_poisson'] = model_poisson2.fittedvalues

# Let's make a new feature y_hat
normalized['y_hat'] = (((normalized[target] - normalized['lambda_poisson'])**2) - normalized[target])/normalized['lambda_poisson']

# OLS estimation without intercept, just with the new feature created above
model_poisson2_aux = sm.OLS.from_formula('y_hat ~ 0 + lambda_poisson', normalized).fit()

# Summary of the auxiliary model, checking 'lambda_poisson' feature p-value.
# If p-value > 0.05, indicates equidispersion of the data, and we're good.
# If p-value <= 0.05, indicates overdispersion of the data, then we'll move to a negative binomial model.
model_poisson2_aux.summary()  # p-value 0.000 -- this means overdispersion in data, let's make a second test to be sure

# Second test for overdispersion
overdisp(model_poisson2, normalized)  # also indicates overdispersion, so let's make a Poisson-Gamma model!


###
# Second GLM test: Negative Binomial Model (Poisson-Gamma)
###

target = 'default_next'
columns_list = (normalized.drop(target, axis=1)).columns
model_formula = ' + '.join(columns_list)
model_formula = target + " ~ " + model_formula
print("This is the funcional form of our model:\n\n", model_formula)

# Estimating the best value for phi ('alpha'):
n_samples = 5000
alphas = np.linspace(0, 10, n_samples)
llf = np.full(n_samples, fill_value=np.nan)
#
for i, alpha in tqdm(enumerate(alphas), total=n_samples, desc='Estimating'):
    try:
        model = smf.glm(formula=model_formula,
                        data=normalized,
                        family=sm.families.NegativeBinomial(alpha=alpha)).fit()
    except:
        continue
    llf[i] = model.llf
#
alpha_value = alphas[np.nanargmax(llf)].round(4)

model_nb = smf.glm(formula=model_formula, data=normalized, family=sm.families.NegativeBinomial(alpha=alpha_value)).fit()

# Let's check for the irrelevant features:
model_nb.pvalues[model_nb.pvalues >= 0.05]

# And discard them, then running the model again
discard = ['age', 'pay_apr', 'bill_aug', 'bill_jul', 'bill_jun', 'bill_apr', 'paid_jul', 'marriage_others']
target = 'default_next'
columns_list = (normalized.drop(target, axis=1).drop(discard, axis=1)).columns
model_formula = ' + '.join(columns_list)
model_formula = target + " ~ " + model_formula
print("This is the funcional form of our new model:\n\n", model_formula)

model_nb = smf.glm(formula=model_formula, data=normalized, family=sm.families.NegativeBinomial(alpha=alpha_value)).fit()

# sm.NegativeBinomial.from_formula(model_formula, data=normalized).fit()  # returns error


###
# Third GLM test: Zero-Inflated Poisson
###

y = normalized[target]
x1 = normalized.drop(target, axis=1)
X1 = sm.add_constant(x1)

x2 = normalized.drop(target, axis=1)  # como consegue esse? força bruta? maior correlação com target?
X2 = sm.add_constant(x2)

#model_zip = sm.ZeroInflatedPoisson(y, X1, exog_infl=X2, inflation='logit').fit()  # returns NaN values in summary
model_zip = sm.ZeroInflatedPoisson(y, X1, inflation='logit').fit()

# Filtering pvalues > 0.05
discard = list((model_zip.pvalues[model_zip.pvalues > 0.05]).index)

y = normalized[target]
x1 = normalized.drop(target, axis=1)
x1 = x1.drop(discard, axis=1)
X1 = sm.add_constant(x1)

x2 = normalized.drop(target, axis=1)
X2 = sm.add_constant(x2)

model_zip2 = sm.ZeroInflatedPoisson(y, X1, inflation='logit').fit()

# Filtering pvalues again
discard2 = list((model_zip2.pvalues[model_zip2.pvalues > 0.05]).index)

y = normalized[target]
x1 = normalized.drop(target, axis=1)
x1 = x1.drop(discard, axis=1)
x1 = x1.drop(discard2, axis=1)
X1 = sm.add_constant(x1)

model_zip3 = sm.ZeroInflatedPoisson(y, X1, inflation='logit').fit()


###
# Fourth and last GLM test: Zero-Inflated Poisson-Gamma
###

y = normalized[target]
x1 = normalized.drop(target, axis=1)
X1 = sm.add_constant(x1)

x2 = normalized.drop(target, axis=1)
X2 = sm.add_constant(x2)

model_zinb = sm.ZeroInflatedNegativeBinomialP(y, X1, inflation='logit').fit(maxiter=1000)

model_zinb.params
model_zinb.pvalues

model_zinb.summary()




# Let's save the scores in a new dataframe 'scores_llf'
scores_llf = pd.DataFrame({'Modelo':['Poisson','Negative Binomial', 'Zero Inflated Poisson'],
                           'Score':[model_poisson2.llf, model_nb.llf, model_zip3]})

