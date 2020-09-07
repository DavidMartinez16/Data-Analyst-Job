# Data Science Project: Data Analyst Salary Predictor

_This is a data science project that uses the Data Analyst Job dataset, the objective is to predict the average salary by analyzing come characteristics of the company such as sector, industry, number of employees, state, location, size of the company, age of the company,among others._

* [Dataset](https://www.kaggle.com/andrewmvd/data-analyst-jobs) - The used dataset was downloaded from the kaggle repository. The data contains more than 2000 job listing for data analyst position in United States.
* Created a tool that estimates data analyst salaries with a MSE of $14 K in USA by analyzing different features.
* Some Engineered features from the Job Description were extracted, such as amount of Python, Excel and aws that companies put on to see how valuable they were. In addition, applying a word clouds the most common words were extracted in the job description field.
* The Grid Search function was used to optimized the models, such as Linear Regression, Random Forest, K-Nearest Neighbors and Bagging Regression.

## Resources Used 🛠️
* **Python Version:** 3.7
* **Packages:** Pandas, Numpy, Matplotlib, Seaborn and Sklearn
* **Scrapper Github:** https://github.com/PlayingNumbers/ds_salary_proj
* **Programs:** Spyder and Jupyter Notebook

## Dataset Features 📦
_The following are the features contained in the Data Analyst Job Dataset :_

* Job Title
* Salary Estimate
* Job Description
* Company Rating
* Company Name
* Location
* Headquarters
* Size
* Year of Founded
* Type of Ownership¨
* Industry
* Sector
* Revenue
* Competitors
* Easy Apply

# Project Overview

## Data Cleaning
_In order to clean the data, I did the following steps_

* Remove the Unnamed column
* Remove the rows with salary estimate -1
* Remove the text and the $ and K symbols in the salary column
* Create a new column to store only the text of the company name
* Create the Age column, to store the years of the company has been working since the foundation year
* Parsing the job description to find requisites such as Python, Excel, Aws, etc.

## Exploratory Data Analysis (EDA)

Bar Plot to show the top 10 most common Professions related with Data Analyst.
![number_jobs](https://user-images.githubusercontent.com/63115543/92315036-c4e1dc00-efa5-11ea-928d-142842e4302d.jpg)

Words Cloud with the most common words in the Job Description
![Anotación 2020-09-05 183135](https://user-images.githubusercontent.com/63115543/92315045-0bcfd180-efa6-11ea-9a6b-9eeec8a2a6ec.jpg)

Number of companys in different Age Groups
![age](https://user-images.githubusercontent.com/63115543/92315061-48033200-efa6-11ea-81dd-c69dea8abb99.jpg)

## Model Building and Tunning

In this section, I split the data in 20 % for testing and 80 % for training, and evaluated models performance as Linear Regression, Random Forest, K Nearest Neighbors and Bagging Regression using the Negative Mean Absolute Error with Cross Validation. In this case, the model with the lowest NMAE was the Random Forest Regressor.

To tune the model I used the Grid Search CV function in order to get the best parameters. And finally, I tested the selected model with the testing set and I got 14.76 in the Mean Absolute Error.


