# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 10:37:10 2020

@author: david
"""

# Import libraries

import pandas as pd

# ---------------------------- READ THE DATA ---------------------------------------

data = pd.read_csv('DataAnalyst.csv')

# ------------------------- UNDERSTANDING THE DATA ----------------------------------

print('Información de la data')
print(data.info())

# Description of the data
print('Data Description')
print(data.describe())

# Missing Values
print('Missing Values')
print(data.isnull().sum())

# -------------------------- DATA CLEANING ------------------------------------------

# Remove the Unnamed column
data.drop(['Unnamed: 0'], axis=1, inplace=True)

# Remove the rows with salary estimate -1
data = data[data['Salary Estimate'] != '-1']

# Salary cleaning

# Remove the text in the salary column
salary = data['Salary Estimate'].apply(lambda x: x.split(' ')[0])

# Remove the K and the $
n_salary =  salary.apply(lambda x: x.replace('K','').replace('$',''))

# Split the salary in the max and min values in a new column
data['min_salary'] = n_salary.apply(lambda x: int(x.split('-')[0]))
data['max_salary'] = n_salary.apply(lambda x: int(x.split('-')[1]))
data['avg_salary'] = (data.max_salary + data.min_salary)/2

# Company name only text
data['Company Text'] = data.apply(lambda x: x['Company Name'] if x['Rating'] <0 else x['Company Name'][:-3], axis=1)

# Location only First letters
data['loc'] = data['Location'].apply(lambda x: x.split(',')[1])
data['loc'] = data['loc'].apply(lambda x: ' CO' if 'arapahoe' in x.lower() else x)

# Headquarters
data['head'] = data['Headquarters'].apply(lambda x: x if x =='-1' else x.split(',')[1])

# Age of the company
data['age'] = data['Founded'].apply(lambda x: x if x<1 else (2020-x))

# Parsing of job description (python, excel, etc)

# Python
data['Python'] = data['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)

# Excel
data['Excel'] = data['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)

# Matlab
data['Matlab'] = data['Job Description'].apply(lambda x: 1 if 'matlab' in x.lower() else 0)

# Spark
data['Spark'] = data['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)

# aws
data['aws'] = data['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)

# Job Title

# Senior
data['senior'] = data['Job Description'].apply(lambda x: 1 if 'senior' in x.lower() else 0)

# Junior
data['junior'] = data['Job Description'].apply(lambda x: 1 if ('junior' or 'jr.' or 'jr') in x.lower() else 0)

# Master
data['master'] = data['Job Description'].apply(lambda x: 1 if 'master' in x.lower() else 0)

# Save the processed data
data.to_csv('Cleaned_Data.csv',index=False)