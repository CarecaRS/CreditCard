# Importing the needed libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score
%autoindent OFF  # Line exclusively for IDE use (neovim), needs to be taken off if used another IDE

# Load the database
full = pd.read_csv('csv/creditcards.csv')

# Lowercase for all columns, for the ease of use
full.columns = full.columns.str.lower()

# Changing 'sex' column (two variables) to dummy 'ismale'
mask = full['sex'] == 2
full.loc[mask, 'sex'] = 0
full = full.rename(columns={'sex': 'ismale'})

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


# Outliers found in the monetary values, it's easier now to get hold
# of all the features to normalize later
mask = full.dtypes == 'float64'
names_to_norm = full.columns[mask]

# Transforming 'education' and 'marriage' into dummies
dummies = pd.get_dummies(full[['education', 'marriage']], drop_first=True, dtype=int)

# Create features: % of limit use and % total paid
mask = full.columns.str.find('bill') == 0
names_bill = full.columns[mask]
percent_bill = pd.DataFrame()
for col in names_bill:
    percent_bill[f'bill_pct_{col[-3:]}'] = full[col]/full['limit_bal']

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

# Concatenating all datasets untill now
full = pd.concat([full, percent_bill], axis=1)
full = pd.concat([full, percent_paid], axis=1)
full = pd.concat([full, dummies], axis=1)

# Evaluating the customer default history
full['default_hist'] = (full[['pay_apr', 'pay_may', 'pay_jun', 'pay_jul', 'pay_aug', 'pay_sep']] > 0).sum(axis=1)


# Checking outliers in 'limit_bal'
#sns.boxplot(y=full['limit_bal'])
#plt.show()

# Now, yes, dealing with the normalization
temp = pd.DataFrame(normalize(full[names_to_norm], axis=1))
temp.columns = names_to_norm

# Creates a new dataframe with the normalized values
normalized = full.copy()
for col in temp.columns:
    normalized[col] = temp[col]

# Getting rid of the original features that were transformed in dummies
# and also the id feature
normalized = normalized.drop(['education', 'marriage', 'id'], axis=1)

# All data wrangling was done, now it's time to build up the classification model!

# In this project I'm gonna be using 15% of the whole dataset for validation,
# approx. 20% for testing and approx. 65% for training
valid_part = 0.15
test_part = 0.235  # 19.98% of the total
train_part = 0.765  # 65,02% of the total
seed = 42
target = 'default_next'

# Segregating a validation set
norm_model, valid = train_test_split(normalized, test_size=valid_part, random_state=seed)

# Splitting the rest into train/test sets
train, test = train_test_split(norm_model, test_size=test_part, random_state=seed)


##########################################
#    DEFINING AND TRAINING THE MODELS    #
##########################################


test


percent_paid
