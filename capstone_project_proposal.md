# Machine Learning Engineer Nanodegree
# Capstone Proposal

Khalil Zouhar, June 12, 2023

Updated: July 05, 2023
## Proposal

The project is a newly added Kaggle competition: Predict House sales prices.

[Competition details available Here](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)

### Domain Background

House Price Prediction:

An accurate prediction on the house price is important to prospective homeowners,
developers, investors, appraisers, tax assessors and other real estate market stakeholders, such
as, mortgage lenders and insurers. 
There is no standard certified way to assess the price of house. Therefore, the availability of a house price prediction model using machine learning can help fill up an
important information gap and improve the efficiency of the real estate market 

While the house price data at hand refers the region of Ames, Iowa, the techniques used in price housing prediction like data preparation, feature engineering could be similar using data from different 
similar initiatives around the world. This is true since we are trying to solve similar problem in different part of the world.
I attend to leverage feature engineering techniques mentioned in different research articles worldwide.
House classification description may slightly change from one area but the human behavior to prefer 1 house over another one will be similar (ie: house size or neighbourhood will influence the price in similar ways worldwide )

It is worthy to highlight that the scope  of this project is to develop a prediction tool to be used in the region of Ames, Iowa.
The years it covers are from 2010 to 2016. Outside the house features, other attributes may affect the price of house like the economical condition, inflation..etc. 
Those attributes may change year over year, building a model that take into account those attributes is out of scope for this project.
Using the same model to predict the house price in other areas or recent years (ie: 2023) may not lead to a good prediction without training the model with up-to-date data.

To keep the model up to date a pipeline could be build that continuously retrain the model with current data for better prediction.
I also can forecast that with proper data and proper transformation (may be using spark) and feature engineering the same piepline can digest housing data from different parts of the world and
expose and api that can predict a House price worldwide. This is of course out of scope for this project.

I intend to use the following articles and leverage proven techniques of feature engineering for better quality of the models.

#### References:

- [Housing Price Prediction via Improved Machine Learning Techniques] (https://www.sciencedirect.com/science/article/pii/S1877050920316318)
- [House Price Prediction using a Machine Learning Model: A Survey of Literature] (https://www.researchgate.net/publication/347584803_House_Price_Prediction_using_a_Machine_Learning_Model_A_Survey_of_Literature)
- [Prediction and Analysis of Housing Price Based on the Generalized Linear Regression Model] (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9536958/)

### Problem Statement

#### Stakeholders: 
prospective homeowners,
developers, investors, appraisers, tax assessors and other real estate market stakeholders, such
as, mortgage lenders and insurers would like to predict house price.

#### Domain:
Real State Business, Banking Business, Insurance Business

#### Problem:
How to efficiently predict a house price?. There is an information gap when it came to house prediction. The information gap can be filled with using 
Machine learning techniques, thus driving real eastate efficiency.

### Datasets and Inputs

The dataset has records of 1460 house sold between 2010 and 2016 in Ames, Iowa. Each data record captures a list of 79 house features (ie: size, location) and the correspinding sale price.
House features are described in a separate file: data_description.txt

The dataset were compiled by Dean De Cok and collect the house sales in the area of Ames, Iowa.

The Data is available at: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

### Solution Statement

The goal here is to build a system that can predict a price of a given house based on house features. 
The data we have is a tabular data and the first solution that came ot mind is to use Decision trees.

Also, In this scenario, we are being asked to predict a numerical outcome. As result this is regression task.

Therefor this problem can be approached with tree machine learning models: Decision tree, random forest and gradient boosting machines.

I plan to use the decision tree as my baseline model then built on this experience to tune my candidate models.

Decision trees are easy to train and can give an insight on how to tune the hyperparameters.

For execution, I plan to use Jupyter Notebook in AWS environment.

Also, the following libraries will be used to
Pandas — For handling structured data
Scikit Learn — For machine learning
NumPy — For linear algebra and mathematics
Seaborn — For data visualization

In addition to the libraries, I plan also to use aws autogluon to help tune the hyperparameters.

### Benchmark Model

To evaluate the performance of my model I intend to compare the prediction provided by the model to the house pricing information for the Ames area available at : https://www.redfin.com/city/477/IA/Ames/housing-market. 
I will use the Kaggle leaderboard, in which the solutions are evaluated upon test data not made available to the competitors. In the leaderboard we can benchmark my results against solutions from other competitors.

### Evaluation Metrics

The evaluation metric is already defined by the competition to be Root-Mean-Squared-Error (RMSE).

The root-mean-square deviation (RMSD) or root-mean-square error (RMSE) is a frequently used measure of the differences between values (sample or population values) predicted by a model or an estimator and the values observed. The RMSD represents the square root of the second sample moment of the differences between predicted values and observed values or the quadratic mean of these differences.

https://en.wikipedia.org/wiki/Root-mean-square_deviation

### Project Design

#### Intended workflow:
![alt text](images/training_lifec_cycle.png "Training Life Cylce")

#### Data cleanup

- Remove duplicate record. Checking for missing values.
- Transform N/A to a category to a avoid errors.
- Converting  date values to string value.
- Transform number encoded categories to string so the emphasis is not on the magnitude of the number but rather the category.

The following highlight  all features in the training data and their types:

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1460 entries, 0 to 1459
Data columns (total 80 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   MSSubClass     1460 non-null   int64  
 1   MSZoning       1460 non-null   object 
 2   LotFrontage    1201 non-null   float64
 3   LotArea        1460 non-null   int64  
 4   Street         1460 non-null   object 
 5   Alley          91 non-null     object 
 6   LotShape       1460 non-null   object 
 7   LandContour    1460 non-null   object 
 8   Utilities      1460 non-null   object 
 9   LotConfig      1460 non-null   object 
 10  LandSlope      1460 non-null   object 
 11  Neighborhood   1460 non-null   object 
 12  Condition1     1460 non-null   object 
 13  Condition2     1460 non-null   object 
 14  BldgType       1460 non-null   object 
 15  HouseStyle     1460 non-null   object 
 16  OverallQual    1460 non-null   int64  
 17  OverallCond    1460 non-null   int64  
 18  YearBuilt      1460 non-null   int64  
 19  YearRemodAdd   1460 non-null   int64  
 20  RoofStyle      1460 non-null   object 
 21  RoofMatl       1460 non-null   object 
 22  Exterior1st    1460 non-null   object 
 23  Exterior2nd    1460 non-null   object 
 24  MasVnrType     588 non-null    object 
 25  MasVnrArea     1452 non-null   float64
 26  ExterQual      1460 non-null   object 
 27  ExterCond      1460 non-null   object 
 28  Foundation     1460 non-null   object 
 29  BsmtQual       1423 non-null   object 
 30  BsmtCond       1423 non-null   object 
 31  BsmtExposure   1422 non-null   object 
 32  BsmtFinType1   1423 non-null   object 
 33  BsmtFinSF1     1460 non-null   int64  
 34  BsmtFinType2   1422 non-null   object 
 35  BsmtFinSF2     1460 non-null   int64  
 36  BsmtUnfSF      1460 non-null   int64  
 37  TotalBsmtSF    1460 non-null   int64  
 38  Heating        1460 non-null   object 
 39  HeatingQC      1460 non-null   object 
 40  CentralAir     1460 non-null   object 
 41  Electrical     1459 non-null   object 
 42  1stFlrSF       1460 non-null   int64  
 43  2ndFlrSF       1460 non-null   int64  
 44  LowQualFinSF   1460 non-null   int64  
 45  GrLivArea      1460 non-null   int64  
 46  BsmtFullBath   1460 non-null   int64  
 47  BsmtHalfBath   1460 non-null   int64  
 48  FullBath       1460 non-null   int64  
 49  HalfBath       1460 non-null   int64  
 50  BedroomAbvGr   1460 non-null   int64  
 51  KitchenAbvGr   1460 non-null   int64  
 52  KitchenQual    1460 non-null   object 
 53  TotRmsAbvGrd   1460 non-null   int64  
 54  Functional     1460 non-null   object 
 55  Fireplaces     1460 non-null   int64  
 56  FireplaceQu    770 non-null    object 
 57  GarageType     1379 non-null   object 
 58  GarageYrBlt    1379 non-null   float64
 59  GarageFinish   1379 non-null   object 
 60  GarageCars     1460 non-null   int64  
 61  GarageArea     1460 non-null   int64  
 62  GarageQual     1379 non-null   object 
 63  GarageCond     1379 non-null   object 
 64  PavedDrive     1460 non-null   object 
 65  WoodDeckSF     1460 non-null   int64  
 66  OpenPorchSF    1460 non-null   int64  
 67  EnclosedPorch  1460 non-null   int64  
 68  3SsnPorch      1460 non-null   int64  
 69  ScreenPorch    1460 non-null   int64  
 70  PoolArea       1460 non-null   int64  
 71  PoolQC         7 non-null      object 
 72  Fence          281 non-null    object 
 73  MiscFeature    54 non-null     object 
 74  MiscVal        1460 non-null   int64  
 75  MoSold         1460 non-null   int64  
 76  YrSold         1460 non-null   int64  
 77  SaleType       1460 non-null   object 
 78  SaleCondition  1460 non-null   object 
 79  SalePrice      1460 non-null   int64  
dtypes: float64(3), int64(34), object(43)
memory usage: 912.6+ KB

```

The missing data can negatively impact significantly the accuracy of our model. 
The following highlight the missing values in the training data:

```
Rows     :  1460
Columns  :  81

Features : 
 ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'SalePrice']

Missing values :   7829

Unique values :  
 Id               1460
MSSubClass         15
MSZoning            5
LotFrontage       110
LotArea          1073
Street              2
Alley               2
LotShape            4
LandContour         4
Utilities           2
LotConfig           5
LandSlope           3
Neighborhood       25
Condition1          9
Condition2          8
BldgType            5
HouseStyle          8
OverallQual        10
OverallCond         9
YearBuilt         112
YearRemodAdd       61
RoofStyle           6
RoofMatl            8
Exterior1st        15
Exterior2nd        16
MasVnrType          3
MasVnrArea        327
ExterQual           4
ExterCond           5
Foundation          6
                 ... 
BedroomAbvGr        8
KitchenAbvGr        4
KitchenQual         4
TotRmsAbvGrd       12
Functional          7
Fireplaces          4
FireplaceQu         5
GarageType          6
GarageYrBlt        97
GarageFinish        3
GarageCars          5
GarageArea        441
GarageQual          5
GarageCond          5
PavedDrive          3
WoodDeckSF        274
OpenPorchSF       202
EnclosedPorch     120
3SsnPorch          20
ScreenPorch        76
PoolArea            8
PoolQC              3
Fence               4
MiscFeature         4
MiscVal            21
MoSold             12
YrSold              5
SaleType            9
SaleCondition       6
SalePrice         663
Length: 81, dtype: int64
Test (1459, 80) Train (1460, 81)

```

The following represent features with missing values: 

```
PoolQC          1453
MiscFeature     1406
Alley           1369
Fence           1179
MasVnrType       872
FireplaceQu      690
LotFrontage      259
GarageYrBlt       81
GarageCond        81
GarageType        81
GarageFinish      81
GarageQual        81
BsmtFinType2      38
BsmtExposure      38
BsmtQual          37
BsmtCond          37
BsmtFinType1      37
MasVnrArea         8
Electrical         1
```

## Categorical features marked as NA

Some categorical features are marked as NA when the feature of the house is non-existant. 
All those should be identified and replaced with something more descriptive.
Numerical categorical features that are marked as NA will be replaced with 0.

The remaining ones will be replaced as follows:

```
    "PoolQC": "No Pool",
    "MiscFeature": "No Feature",
    "Alley": "No Alley",
    "Fence": "No Fence",
    "FireplaceQu": "No Fireplace",
    "GarageCond": "No Garage",
    "GarageType": "No Garage",
    "GarageYrBlt": "No Garage",
    "GarageFinish": "No Garage",
    "GarageQual": "No Garage",
    "BsmtExposure": "No Basement",
    "BsmtFinType2": "No Basement",
    "BsmtFinType1": "No Basement",
    "BsmtCond": "No Basement",
    "BsmtQual": "No Basement",
    "MasVnrType": "No Veneer",
```

## Date features

The following features are categorized as date feature by looking at the data and the description:
YearBuilt, YrSold, GarageYrBlt and YearRemodAdd. However, their type is represented as numerical type (int64).

Since the features above doesn't reflect a magnitude it will be converted to string type so they don't negatively impact the prediction capabilities of our model.

#### Categorical features marked as NA

The following categories are represented with numerical values type (int64). 

The machine learning algorithm could interpret the magnitude of the number to be relevant instead of considering the number as another category.
The solution is to convert the number to the description  defined in data_description.txt.

```
MSSubClass: Identifies the type of dwelling involved in the sale.	

        20	1-STORY 1946 & NEWER ALL STYLES
        30	1-STORY 1945 & OLDER
        40	1-STORY W/FINISHED ATTIC ALL AGES
        45	1-1/2 STORY - UNFINISHED ALL AGES
        50	1-1/2 STORY FINISHED ALL AGES
        60	2-STORY 1946 & NEWER
        70	2-STORY 1945 & OLDER
        75	2-1/2 STORY ALL AGES
        80	SPLIT OR MULTI-LEVEL
        85	SPLIT FOYER
        90	DUPLEX - ALL STYLES AND AGES
       120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
       150	1-1/2 STORY PUD - ALL AGES
       160	2-STORY PUD - 1946 & NEWER
       180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
       190	2 FAMILY CONVERSION - ALL STYLES AND AGES
       
 OverallQual: Rates the overall material and finish of the house

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average
       5	Average
       4	Below Average
       3	Fair
       2	Poor
       1	Very Poor
	
OverallCond: Rates the overall condition of the house

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average	
       5	Average
       4	Below Average	
       3	Fair
       2	Poor
       1	Very Poor      
```

#### Labels

Our goal is to predict the price of a given house in the region of Ales. 
The label will be the Sales Price. I plotted the Sales Price on histogram:

![alt text](images/sales_price.png)

The distribution is right skewed which is expected as few houses will be expensive.

#### Correlations

To analyze the relations between features, the correlation can higlight any abnormal pattern in data.

The following represents the correlation matrix for the house data features after excluding non-numeric features. 
Lighter color indicates stronger correlation. Based on the graph below it seems features that are related to space correlates well
with Sales Price which is expected since house with more space tends to be more expensive.

![alt text](images/correlation.png)


#### Features Transformation 
- Convert variables into features. 
- Standardize/normalize features, 
- Apply numerical transformations, 
- perform one-hot encoding.

#### Features Creation 
Analyse the possibility of deriving new features from the existing ones

#### Features Selection or Extraction
Not all 79 features influence the price. So we need extract the relevant features.

#### Machine Learning Model Selection (see the solution section for the motive behind choosing the algorithms below):

   - Decision Tree:  A tree algorithm used in machine learning to find patterns in data by learning decision rules.
   - Random Forest — A type of bagging method that plays on ‘the wisdom of crowds’ effect. It uses multiple independent decision trees in parallel to learn from data and aggregates their predictions for an outcome.
   - Gradient Boosting Machines — A type of boosting method that uses a combination of decision tree in series. Each tree is used to predict and correct the errors by the preceding tree additively.
   
   Random forests and gradient boosting can significantly improve the quality of weak decision trees. 
   They’re great algorithms to use if we  have small training data sets like in this case.

#### Training:
   
   At this stage we will teach our model using examples from the dataset. In the training stage we will tune the model hyperparameter. 
   Our goal at this stage if to find the optimal hyperparmater value that lower the model bias and model variance.

- Model bias:
Refers to models that under-fit the training data leading to poor predictive capacity on unseen data. Generally, the simpler the model the higher the bias.

- Model variance:
  Refers to Models that over-fit the training data leading to poor predictive capacity on unseen data. Generally, the more complexity in the model the higher the variance.
   
I plan to tune the following hyperparamaters:

   - max_depth — The maximum number of nodes for a given decision tree.
   - max_features — The size of the subset of features to consider for splitting at a node.
   - n_estimators — The number of trees used for boosting or aggregation. This hyperparameter only applies to the random forest and gradient boosting machines.
   - learning_rate — The learning rate acts to reduce the contribution of each tree. This only applies for gradient boosting machines.

For better hyperparameter tuning (to ovoid over-fitting and model bias) I will use the following techniques:
   
   - Grid search: Choosing the range of your hyperparameters is an iterative process. With more experience you’ll begin to get a feel for what ranges to set. The good news is once you’ve chosen your possible hyperparameter ranges, grid search allows you to test the model at every combination of those ranges. I’ll talk more about this in the next section.
   - Cross validation: Models are trained with a 5-fold cross validation. A technique that takes the entirety of your training data, randomly splits it into train and validation data sets over 5 iterations.

#### Evaluation
   At this stage we evaluate our selected machine learning models. Either we are happy with the results or we need to go trough to another cycle/iteration.  

#### Submit Results
   Finally, after few iterations of improving our select model we reach the optimal model and we are ready to submit our result.