# house-prediction-Lab


****                                                                                   **SAS Viya CODE**
proc contents data=WORK.IMPORT; 
run;
data house_data_cleaned;
    set WORK.IMPORT;
    length Amount_Lac 8;
    /* Fix: Reference the column using name literal */
    Amount_Upper = upcase(strip('Amount(in rupees)'n));
    if index(Amount_Upper, 'LAC') then do;
        Amount_Lac = input(scan(Amount_Upper, 1, ' '), best.);
    end;
    else if index(Amount_Upper, 'CR') then do;
        Amount_Lac = input(scan(Amount_Upper, 1, ' '), best.) * 100;
    end;
    drop 'Amount(in rupees)'n Amount_Upper;
run;


data house_data_cleaned;
    set house_data_cleaned;
    length Carpet_Num 8;
    /* Step 1: Handle variable with space using name literal */
    Carpet_Lower = lowcase(strip('Carpet Area'n));
    /* Step 2: Extract numeric value */
    Area_Value = input(compress(scan(Carpet_Lower, 1, ' '), ','), best.);
    /* Step 3: Convert to sqft */
    if index(Carpet_Lower, 'acre') then Carpet_Num = Area_Value * 43560;
    else if index(Carpet_Lower, 'bigha') then Carpet_Num = Area_Value * 27225;
    else if index(Carpet_Lower, 'cent') then Carpet_Num = Area_Value * 435.6;
    else if index(Carpet_Lower, 'ground') then Carpet_Num = Area_Value * 2400;
    else if index(Carpet_Lower, 'kanal') then Carpet_Num = Area_Value * 5445;
    else if index(Carpet_Lower, 'marla') then Carpet_Num = Area_Value * 272.25;
    else if index(Carpet_Lower, 'sqm') or index(Carpet_Lower, 'sq meter') then Carpet_Num = Area_Value * 10.7639;
    else if index(Carpet_Lower, 'sqyrd') or index(Carpet_Lower, 'sq yard') then Carpet_Num = Area_Value * 9;
    else if index(Carpet_Lower, 'sqft') then Carpet_Num = Area_Value;
    else Carpet_Num = .;
    drop 'Carpet Area'n Carpet_Lower Area_Value;
run;


data house_data_cleaned;
    set house_data_cleaned;
    length Floor_Num 8;
    /* Use lowercase version for easier checks */
    Floor_Lower = lowcase(strip(Floor));
    /* Extract numeric floor or convert label */
    if index(Floor_Lower, 'ground') then Floor_Num = 0;
    else if index(Floor_Lower, 'lower basement') then Floor_Num = -1;
    else if index(Floor_Lower, 'upper basement') then Floor_Num = -2;
    else Floor_Num = input(scan(Floor_Lower, 1, ' '), best.);
    drop Floor Floor_Lower;
run;


data house_data_cleaned;
    set house_data_cleaned;
    length Title_Lower Place_From_Title $100;
    /* Convert to lowercase for case-insensitive search */
    Title_Lower = lowcase(Title);
    /* Look for "sale in" first */
    pos_in = index(Title_Lower, "sale in");
    /* If "sale in" is found, extract text after it */
    if pos_in > 0 then Place_From_Title = substr(Title, pos_in + 8);
    /* Else look for just "sale" */
    else do;
        pos_sale = index(Title_Lower, "sale");
        if pos_sale > 0 then Place_From_Title = substr(Title, pos_sale + 5);
        else Place_From_Title = "";
    end;
    drop Title_Lower pos_in pos_sale;
run;


data house_data_cleaned;
    set house_data_cleaned;
    /* Fill Society ONLY if it's missing AND Place_From_Title is available */
    if missing(Society) then Society = Place_From_Title;
run;
 


data house_data_cleaned;
    set house_data_cleaned;
    length Super_Area_Num 8;
    /* Step 1: Standardize case and remove extra space */
    Super_Lower = lowcase(strip('Super Area'n));
    /* Step 2: Extract numeric value from string */
    Area_Value = input(compress(scan(Super_Lower, 1, ' '), ','), best.);
    /* Step 3: Apply conversion to sqft based on unit */
    if index(Super_Lower, 'acre') then Super_Area_Num = Area_Value * 43560;
    else if index(Super_Lower, 'bigha') then Super_Area_Num = Area_Value * 27225;
    else if index(Super_Lower, 'cent') then Super_Area_Num = Area_Value * 435.6;
    else if index(Super_Lower, 'ground') then Super_Area_Num = Area_Value * 2400;
    else if index(Super_Lower, 'kanal') then Super_Area_Num = Area_Value * 5445;
    else if index(Super_Lower, 'marla') then Super_Area_Num = Area_Value * 272.25;
    else if index(Super_Lower, 'sqm') or index(Super_Lower, 'sq meter') then Super_Area_Num = Area_Value * 10.7639;
    else if index(Super_Lower, 'sqyrd') or index(Super_Lower, 'sq yard') then Super_Area_Num = Area_Value * 9;
    else if index(Super_Lower, 'sqft') then Super_Area_Num = Area_Value;
    else Super_Area_Num = .;
    drop 'Super Area'n Super_Lower Area_Value;
run;


data house_data_cleaned;
    set house_data_cleaned;
    length Carpet_Num 8;
    if missing(Carpet_Num) and not missing(Super_Area_Num) then
        Carpet_Num = 0.75 * Super_Area_Num;
run;

proc means data=house_data_cleaned median;
    var Bathroom;
run;
data house_data_cleaned;
    set house_data_cleaned;
    if missing(Bathroom) then Bathroom = 2;
run;
proc means data=house_data_cleaned median;
    var Floor_Num;
run;
data house_data_cleaned;
    set house_data_cleaned;
    if missing(Floor_Num) then Floor_Num = 2;
run;


data house_data_cleaned;

    set house_data_cleaned;
    length BHK 8 Property_Type $20 BHK_Combined $30;
    /* Step 1: Extract BHK number from Title */

    if prxmatch("/\d+\s*BHK/i", Title) then

        BHK = input(scan(Title, 1, ' '), best.);
    /* Step 2: Extract Property Type from Title */

    if index(upcase(Title), 'FLAT') then Property_Type = 'Flat';

    else if index(upcase(Title), 'VILLA') then Property_Type = 'Villa';

    else if index(upcase(Title), 'APARTMENT') then Property_Type = 'Apartment';

    else Property_Type = 'Other';
    /* Step 3: Merge into BHK_Combined */

    if not missing(BHK) and not missing(Property_Type) then

        BHK_Combined = cats(put(BHK, 1.), ' BHK ', Property_Type);

    else if not missing(Property_Type) then

        BHK_Combined = Property_Type;

    else if not missing(BHK) then

        BHK_Combined = cats(put(BHK, 1.), ' BHK');

    else BHK_Combined = 'Unknown';

run;

data house_data_cleaned;
    set house_data_cleaned;
    if missing(Amount_Lac) then delete;
run;
 
data house_data_cleaned;
    set house_data_cleaned;

    /* Handle missing character fields */
    if missing(Transaction) then Transaction = "Unknown";
    if missing(Furnishing) then Furnishing = "Unknown";
    if missing(Ownership) then Ownership = "Unknown";
    if missing(Facing) then Facing = "Unknown";
    if missing(Overlooking) then Overlooking = "Unknown";

run;
 
data house_data_cleaned;
    set house_data_cleaned;
 
    length Price_Per_Sqft_Cat $15;
 
    if missing('Price (in rupees)'n) then 
        Price_Per_Sqft_Cat = "Unknown";
    else 
        Price_Per_Sqft_Cat = strip(put('Price (in rupees)'n, 8.));
run;
 
data house_data_cleaned;
    set house_data_cleaned;
 
    length Balcony_Cat $10;
 
    if missing(Balcony) then 
        Balcony_Cat = "Unknown";
    else 
        Balcony_Cat = strip(put(Balcony, 8.));
run;
 
data house_data_cleaned;
    set house_data_cleaned;
 
    length Car_Parking_Cat $10;
 
    if missing('Car Parking'n) then 
        Car_Parking_Cat = "Unknown";
    else 
        Car_Parking_Cat = strip(put('Car Parking'n, 8.));
run;
 
 
data house_data_cleaned;
    set house_data_cleaned;
 
    length Carpet_Area_Cat $15;
 
    if missing(Carpet_Num) then 
        Carpet_Area_Cat = "Unknown";
    else 
        Carpet_Area_Cat = strip(put(Carpet_Num, 8.));
run;

                                                       **Regression_model_python_code**




# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load cleaned dataset
df = pd.read_csv('HOUSE_DATA_CLEANED_v10_Flat_dataset.csv')

# Convert relevant columns to numeric
df['Carpet_Area_Cat'] = pd.to_numeric(df['Carpet_Area_Cat'], errors='coerce')
df['Balcony_Cat'] = pd.to_numeric(df['Balcony_Cat'], errors='coerce')

# Drop rows with missing Carpet or Balcony info for simplicity
df = df.dropna(subset=['Carpet_Area_Cat', 'Balcony_Cat'])

# Define target and features
target = 'Amount_Lac'
features = ['Transaction', 'Furnishing', 'Ownership', 'BHK_Combined',
            'Floor_Num', 'Carpet_Area_Cat', 'Balcony_Cat']

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for categorical and numeric features
cat_cols = ['Transaction', 'Furnishing', 'Ownership', 'BHK_Combined']
num_cols = ['Floor_Num', 'Carpet_Area_Cat', 'Balcony_Cat']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols)
    ]
)

# -------------------------
# Linear Regression Model
# -------------------------
linear_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

linear_pipeline.fit(X_train, y_train)
linear_preds = linear_pipeline.predict(X_test)

linear_mse = mean_squared_error(y_test, linear_preds)
linear_rmse = np.sqrt(linear_mse)

# -------------------------
# Polynomial Regression (degree 2) on log(Amount_Lac)
# -------------------------
X_poly_train = preprocessor.fit_transform(X_train)
X_poly_test = preprocessor.transform(X_test)

poly_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)

y_train_log = np.log1p(y_train)
poly_model.fit(X_poly_train, y_train_log)
poly_preds_log = poly_model.predict(X_poly_test)
poly_preds = np.expm1(poly_preds_log)

poly_mse = mean_squared_error(y_test, poly_preds)
poly_rmse = np.sqrt(poly_mse)

# -------------------------
# Visualization of predictions
# -------------------------
plt.figure(figsize=(10, 5))

# Linear Regression Plot
plt.subplot(1, 2, 1)
plt.scatter(np.log1p(y_test), np.log1p(linear_preds), alpha=0.3)
plt.plot([np.log1p(y_test).min(), np.log1p(y_test).max()],
         [np.log1p(y_test).min(), np.log1p(y_test).max()], color='red')
plt.xlabel('Actual log(Amount_Lac)')
plt.ylabel('Predicted log(Amount_Lac)')
plt.title('Linear Regression: log(Amount_Lac)')

# Polynomial Regression Plot
plt.subplot(1, 2, 2)
plt.scatter(np.log1p(y_test), poly_preds_log, alpha=0.3)
plt.plot([np.log1p(y_test).min(), np.log1p(y_test).max()],
         [np.log1p(y_test).min(), np.log1p(y_test).max()], color='red')
plt.xlabel('Actual log(Amount_Lac)')
plt.ylabel('Predicted log(Amount_Lac)')
plt.title('Polynomial Regression: log(Amount_Lac)')

plt.tight_layout()
plt.savefig('Actual_vs_Predicted_Log_Amount_Lac.png')
plt.show()

# Print evaluation results
print(f'Linear Regression RMSE: {linear_rmse:.2f}')
print(f'Polynomial Regression RMSE: {poly_rmse:.2f}')
