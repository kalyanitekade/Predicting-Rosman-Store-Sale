from sklearn.model_selection import train_test_split
from ML_pipeline.utils import read_dataset
from ML_pipeline.utils import merge_df
from ML_pipeline.utils import year_from_date
from ML_pipeline.utils import month_from_date
from ML_pipeline.utils import remove_outliers
from ML_pipeline.data_cleaning import cleaning
from ML_pipeline.data_transformation import cat_to_num
from ML_pipeline.model_training import train_model
from ML_pipeline.model_evaluation import eval_model
from ML_pipeline.feature_importance import feature_importance
from ML_pipeline.graphs import barplot
from sklearn.impute import SimpleImputer
import pandas as pd 
import numpy as np 
import joblib 

#Read dataset 
store_details = pd.read_csv("./input/store.csv")
train_data = pd.read_csv("./input/train.csv", low_memory=False)

#Combining two dataframes 
combined_data = merge_df(store_details, train_data, 'Store')

# Extract year from date
combined_data = year_from_date(combined_data, 'Date', 'Year')

# Extract month from date
combined_data = month_from_date(combined_data, 'Date', 'Month')

#Exploring data
barplot('Year','Sales', combined_data)
barplot('DayOfWeek','Sales',combined_data)
barplot('Promo', 'Sales', combined_data)

combined_data.loc[combined_data["StateHoliday"] == 0, "StateHoliday"] = "0"
barplot('StateHoliday', 'Sales', combined_data)
barplot('SchoolHoliday', 'Sales', combined_data)
barplot('StoreType', 'Sales', combined_data)
barplot('Assortment', 'Sales', combined_data)

store_details.isnull().sum()
train_data.isnull().sum()

#Data cleaning 

#1: The null values in Column Promo2SinceWeek, Promo2SinceYear, PromoInterval is due to Promo2 is 0 for those stores. So we would fill all the null values in these columns with 0.
#2: Since Competition Distance for 3 stores isn't given so we could fill it with mean of the distance given for all other stores
#3: CompetitionOpenSinceMonth, CompetitionOpenSinceYear can be filled using the most occuring month and year respectively. 

store_details = cleaning(store_details, 'Promo2SinceWeek', method='value')
store_details = cleaning(store_details, 'Promo2SinceYear', method='value')
store_details = cleaning(store_details, 'PromoInterval', method='value')
store_details = cleaning(store_details, 'CompetitionDistance', method='mean')
store_details = cleaning(store_details, 'CompetitionOpenSinceMonth', method='mode')
store_details = cleaning(store_details, 'CompetitionOpenSinceYear', method='mode')
combined_data = merge_df(train_data, store_details, 'Store')

#Outlier Detection
sales_zero = combined_data.loc[combined_data['Sales'] == 0] 
sales_greater_than_30 = combined_data.loc[combined_data['Sales'] > 30000]

print("Length of actual dataset:", len(combined_data))
print("Length of data where sales is 0:", len(sales_zero),
      " which is", len(sales_zero)/len(combined_data)*100, "% of the whole data", )

print("Length of data which is greater than 30:", len(sales_greater_than_30),
      "which is ", len(sales_greater_than_30)/len(combined_data)*100, "% of the whole data")

combined_data = remove_outliers(combined_data, 'Sales',  30000)

# Removing Exceptions
combined_data.drop(combined_data.loc[(combined_data['Sales'] == 0) & (combined_data['Open'] == 1) &
                                     (combined_data['StateHoliday'] == 0) &
                                     (combined_data['SchoolHoliday'] == 0)].index, inplace=True)

# Catagorical to numerical
combined_data = cat_to_num(combined_data, 'Assortment', 'default')
combined_data = cat_to_num(combined_data, 'StoreType', 'default')
impute_dict = {
    "Jan,Apr,Jul,Oct": 1,
    "Feb,May,Aug,Nov": 2,
    "Mar,Jun,Sept,Dec": 3
}
combined_data = cat_to_num(combined_data, 'PromoInterval', 'custom', values=impute_dict)
impute_dict_2 = {
    'a': 1,
    'b': 2,
    'c': 3
}
combined_data = cat_to_num(combined_data, 'StateHoliday', 'custom', values=impute_dict_2)

# Convert to numeric
combined_data['StateHoliday'] = pd.to_numeric(combined_data['StateHoliday'])
combined_data['PromoInterval'] = pd.to_numeric(combined_data['PromoInterval'])

combined_data_subset = combined_data[combined_data['Open'] == 1]
combined_data_subset_closed = combined_data[combined_data['Open'] == 0]

# Drop unnecessary columns without inplace=True
X = combined_data_subset.drop(['Sales', 'Customers', 'Open', 'Date'], axis=1)
y = combined_data_subset['Sales']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Perform train-test split on imputed data
x_train, x_test, y_train, y_test_open = train_test_split(X_imputed, y, test_size=0.20)

# Models to train
model_names = ['LinearReg', 'SGDRegr', 'RFReg', 'DTReg']

# Directory to save models
output_dir = './output/'

# Iterate over each model
for model_name in model_names:
    print(f"Training and evaluating {model_name}")
    
    # Train model
    model_path = f"{output_dir}{model_name}.pkl"
    pred = train_model(x_train, x_test, y_train, y_test_open, model_name, model_path)
    
    # Predictions for closed stores (all zeros)
    prediction_closed = np.zeros(combined_data_subset_closed.shape[0])
    
    # Combine predictions
    prediction = np.append(pred, prediction_closed)
    y_test = np.append(y_test_open, np.zeros(combined_data_subset_closed.shape[0]))
    
    # Evaluate model
    results = eval_model(y_test, prediction)
    print(f"Results for {model_name}: {results}")
    
    # Load the model and get feature importance
    model = joblib.load(model_path)
    fi = feature_importance(X.columns, model)
    if fi:
        feature_importances = feature_importance(X.columns, model)
        # Sort features by importance
        sorted_features = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)
        # Select top 5-6 features
        top_features = [feature for feature, importance in sorted_features[:6]]
        print("Top features:", top_features)