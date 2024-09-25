# TO-DO
# 'sex' to 'male' = 1
# check (boxplot) outliers in limit_bal, and change it to int
# change 'education' to categorical: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
# 'marriage' to categorical: (1=married, 2=single, 3=others)
# check for sazonality?
# create feature % of limit use
# binning age?
#
# dummy (será?) 'education', 'marriage'
# estratificação cruzada/aleatória (70/30/10 ou 60/20/20 ou 70/15/15)
# alguma feature de histórico de inadimplência, seja por média de pagamento de atrasos
# ou se pagou atrasado 1+ meses então maior a chance de seguir pagando atrasado
#
# reordenar as features, para melhor entendimento


# Importing the needed libraries
import pandas as pd
%autoindent OFF  # Line exclusively for IDE use (neovim), needs to be taken off if used another IDE

# Load the database
full = pd.read_csv('csv/creditcards.csv')

# Lowercase for all columns, for the ease of use
full.columns = full.columns.str.lower()

full.info()
full.marriage.value_counts()

full.pay_0.value_counts()

full.loc[0]

full['default.payment.next.month'].value_counts()

# Rename all these columns so it's more comprehensive:
'''
PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above) -- posso considerar os números como 'pagamento realizado com X meses de atraso', se for negativo considerado como pagamento adiantado

PAY_2: Repayment status in August, 2005 (scale same as above)

PAY_3: Repayment status in July, 2005 (scale same as above)

PAY_4: Repayment status in June, 2005 (scale same as above)

PAY_5: Repayment status in May, 2005 (scale same as above)

PAY_6: Repayment status in April, 2005 (scale same as above)

BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)

BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)

BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)

BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)

BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)

BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)

PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)

PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)

PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)

PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)

PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)

PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
'''
