import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data Loading
data = pd.read_csv("train.csv")

## Look around -- shape (rows and columns), head()/tail(), info() --
print(data.info())

# Print top 5 rows of dataset
print(data.head(5))

# column names
print(data.columns)

# How to convert object to numericl data?
# We use encoding techniques for that
# Label Encoding


# Before that we see the unique(outlier) values of columns in dataset columns --like the 4+ years in Stay_in_current_city
print(data['Stay_In_Current_City_Years'].unique())

# We correct such values
data['Stay_In_Current_City_Years'] = data['Stay_In_Current_City_Years'].replace('4+', '4')

# To check null values count of each column
print(data.isnull().sum())

# Either remove null values or replace them --mean/median/mode
# Here we remove and convert the column to int datatype
data.dropna(inplace=True)  # inplace true save the data to the dataset after dropping
data['Stay_In_Current_City_Years'] = data['Stay_In_Current_City_Years'].astype(int)

print(data.info())
print(data.isnull().sum())
print(data.describe())

# Since we have dropped rows the indexes of those rows is lost too and so we have missing indexes
data = data.reset_index(drop=True)

# Total count of each unique value in column
print(data['Gender'].value_counts())
# OR by filtering column and finding values by a particular column
print(data[data["Gender"] == 'M']['User_ID'].count())
print(data[data["Gender"] == 'F']['User_ID'].count())

# visualizations to understand the distribution of the various features
sns.countplot(data=data, x="Gender")
plt.show()
sns.countplot(data=data, x="Age", hue='Gender')
plt.show()  # Analysis-- 26-35 is the prominent category
sns.countplot(data=data, x="Marital_Status", hue='Gender')
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(data['Product_Category_1'])  # its frequency of unique values
plt.show()
### The product category 1 sees a great rise of product category 1 and lowest in 12th category

plt.figure(figsize=(10, 5))
sns.countplot(data['Product_Category_2'])  # its frequency of unique values
plt.show()
# The product category 2 sees a considerate balance among categories.
# Cat 2 tops the charts, and other considerable categories are 8,4,5,6,14,15, etc.

plt.figure(figsize=(10, 5))
sns.countplot(data['Product_Category_3'])  # its frequency of unique values
plt.show()

# cat 16 tops the charts, other considerable categories are 15,14,5,8,9,17 etc.
plt.figure(figsize=(10, 5))
sns.countplot(data['Occupation'])  # its frequency of unique values
plt.show()

# occupation sees a constant balance with 4 topping the chart,
# other categories in the considerable amount are 0,1,2,7,12,17,20

plt.figure(figsize=(10, 5))
sns.countplot(data['Stay_In_Current_City_Years'])  # its frequency of unique values
plt.show()

# Majority of the people stay in the current city for 1 year only
sns.histplot(data['Purchase'])
plt.show()
# Kind of like a frequency distribution function where all values ranges add to total purchase values
# It shows the distribution with continuous purchase values in x-axis and the percentage of that particular range of purchase values from the total population in y axis as density(%total purchase)


## Label encoding to convert categorical to numerical columns

# All the colums with dtype as object will be converted to numerical values
# Note:- Even Age is an object here as it takes on values like 55+ or 45-55 which are not numerical values

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

col_list = ['User_ID', 'Product_ID', 'Gender', 'Age', 'City_Category']
for col in col_list:
    data[col] = le.fit_transform(data[col])
    print(col, le.classes_)

# All converted to numbers
print(data.head())

# Correlations
print(data.corr())
# HeatMap
plt.figure(figsize=(15, 5))
sns.heatmap(data.corr(), annot=True)
plt.show()

##-------------------STATISTICAL ANALYSIS-----------------------------------------------------------
# It was observed that the average purchase made by the Men of the age 18-25 was 10000. Is is still the same?

# NULL hypothesis - The mean is 10000
# Alternate hypothesis - The mean is not 10000

# One sample Z-test = is used to test whether or not the mean of a population is equal to some value.
# is used to compare the means of two groups.
#population set
new_data = data.loc[(data['Age'] == 1) & data['Gender'] == 1]  # Male and 18-25 new data corresponding to label encoding
print(new_data['Purchase'].mean())

print(new_data.shape)

#SAMPLING
sample_size = 1000
sample = new_data.sample(sample_size, random_state=44) #Taking random 1000 samples --state gives the same random rows every time
print(sample)

#Finding z score/Can do a T test too
import scipy.stats
from scipy.stats import ttest_1samp
p_mean =10000
sample_mean = sample['Purchase'].mean()
print(sample_mean)

t_stat, p_value = ttest_1samp(sample['Purchase'], 10000) #Here t_stat is the test statistic and p-value is the critical value of alpha probability range(like 1.96 in 0.05 probability)
#This is a two tailed test and we take alpha=0.05 so its p value range is -1.96 to +1.96
print(t_stat, p_value)
# t_stat is the t score value calculated from the formula
# p_value is calculated from t-score and is the probability that the null hypothesis will hold
# if p value is less than the significance level alpha(default 0.05) we reject the null hypothesis
# The actual population mean is 11852.84 -- put that and the p value would be 0.7 which would fail to reject the null hypothesis

# RESULTS -- Since p value is much less than alpha value we reject the null hypothesis

# HYPOTHESIS 2 -----------------------------------------------------------------------------------
# It was observed that the percentage of women of the age that spend more than 10000 was 35%. Is it still the same?
# NULL Hypothesis - proportion is 35%
#Alternate Hypothesis - proportion is not 35%
data_new = data.loc[(data['Purchase']>10000)]
print(data_new.shape)

#no of women in sample
count = data_new['Gender'].value_counts()[0] # 0 for females
#no of obs
nobs = len(data_new["Gender"]) # Total people spending more than 10000
#hypothesize value
p0 = 0.35

print(data_new['Gender'].value_counts()/nobs)
# No of males and females spending more than 10000 in terms of proportion

#Ztest - used to determine whether two population means are different when the variances are known.
from statsmodels.stats.proportion import proportions_ztest
z_stat, p_value = proportions_ztest(count=count, nobs=nobs, value=p0)
print(z_stat, p_value)

#p-value is less than 0.05, reject the null hypothesis i.e.., proportion is not 35%


#-HYPOTHESIS 3-------------------------------------------------------------------
# Is the average purchase made by men and women of the age 18-25 same?
# null hypothesis - average purchase is equal
# alternate hypothesis - average purchase made is not equal

data_men = data[(data['Gender'])==1 & (data['Age']==1)]
data_women = data[(data['Gender'])==0 & (data['Age']==1)]

#Creating Samples
data_men_samples = data_men.sample(500, random_state=0)
data_women_samples = data_women.sample(500, random_state=0)

#checking variance of two samples
print(data_men_samples.Purchase.var())
print(data_women_samples.Purcase.var())

#sample means
print(data_men_samples.Purchase.mean())
print(data_women_samples.Purchase.mean())

#compute f statistic
from scipy.stats import f #f-test is used to compare the variances
F = data_men_samples.Purchase.mean()/data_women_samples.Purchase.mean()

#calculating the degrees of freedom

#Degrees of freedom is the number of independent pieces of
#information used to calculate a statistic
df1 = len(data_men_samples) -1
df2 = len(data_women_samples) -1
print(df1, df2)

#p-value

#cdf - The cumulative distribution function is used
#to describe the probability distribution of random variables

import scipy
scipy.stats.f.cdf(F, df1, df2)

#the p-value is greater than 0.05, do not reject the null hypothesis

#-HYPOTHESIS 4---------------------------------------------------------------
# Is the average purchases made by men and women of the age 18-25 same?

