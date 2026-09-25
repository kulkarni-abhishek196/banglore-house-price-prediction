# Bengaluru House Price Prediction

Predicting residential property prices in Bengaluru using Machine Learning.
Built an end-to-end pipeline handling 13,000+ listings with extensive 
data cleaning, feature engineering, and model comparison.

**Best model:** Random Forest | R² = 65.78% | MAPE = 24.51%

## Data Cleaning Decisions

The following data-cleaning decisions have been made based on the dataset structure, missing-value analysis, outlier analysis, and business context:

1. **Created a standardized BHK column**

   * Added a new column as a substitute for the original `size` column.
   * Extracted and standardized the number of **BHKs** from the `size` column, which originally contained both *bedroom* and *BHK* representations.
   * This provides a consistent representation of the property size.

2. **Imputed missing values in `bath` and `balcony`**

   * Missing (`NaN`) values in the `bath` and `balcony` columns were imputed using the **median value corresponding to the property's BHK/size category**.
   * Using the median with respect to size helps preserve the relationship between property size and these features while reducing the influence of extreme values.

3. **Removed the `society` column**

   * The `society` column was removed entirely because it contained **more than 5,000 missing values**, representing approximately **41% of the dataset**.
   * Additionally, much of the information captured by `society` is already represented through the `location` feature, making it less valuable for the model.

4. **Removed extreme outliers from the BHK feature**

   * Outlier analysis was performed using a **boxplot** on the newly created BHK/size feature.
   * Properties with **more than 5 BHKs** accounted for only around **500 rows out of approximately 13,000 records (~3.78%)**.
   * Since the business objective is focused on the general housing market rather than the **luxury housing segment**, these extreme values were treated as outliers and removed.

5. **Standardized the `availability` column**

   * The `availability` column contained both categorical values and specific dates.
   * All rows where `availability` contained a **date** were converted to **`Not Ready To Move`**.
   * This simplifies the feature into a consistent categorical representation focused on whether the property is currently ready for possession.

6. **Removed inconsistent `total_sqft` records**

   * The `total_sqft` column contains values represented as **ranges** (e.g., `1200 - 1500`) as well as values expressed in different units such as **Square Yards** and **Square Meters**.
   * Instead of attempting to transform or standardize these different representations, the affected **234 rows were removed from the dataset**.
   * These records represent only **1.83% of the total dataset**, so removing them has a minimal impact on the overall dataset size.
   * This approach also avoids introducing assumptions through midpoint conversion or unit conversion and keeps the remaining `total_sqft` values consistent for further analysis and model training.
   * After removing the range values and mismatched units from `total_sqft`, some unrealistic outliers still remained, such as **1 sqft, 60 sqft**, and similar values. Considering that the minimum standard housing size in Bengaluru should be approximately **350–400 sqft**, all records with `total_sqft` **below 300 sqft** were removed from the dataset.

7. **Removed Duplicate Records**

    * Identified and removed **937 duplicate rows**, representing approximately **7.5% of the dataset**.
    * The duplicate records had **identical values across all columns, including the decimal values in the `price` column**.
    * These duplicates are likely to represent the **same property being listed multiple times by different agents** or may have resulted from a **data-scraping artifact**.
    * Since there was no distinguishing information between the duplicate records, retaining them would unnecessarily give those properties more weight during model training.
    * **The first occurrence of each duplicate record was retained**, while subsequent identical occurrences were removed.


## Feature Engineering & Preprocessing Decisions

### Created `price_per_sqft` and Removed `area_type`

* A new feature, **`price_per_sqft`**, was calculated using the existing **price** and **`total_sqft`** values.
* This derived feature provides a more granular representation of the property's price relative to its reported area.
* Since the dataset also contains **`location`**, the calculated `price_per_sqft` helps capture the **location-wise price premium**. Properties in different locations can have significantly different price levels, and the price per square foot provides a way to represent this variation.
* The `area_type` column contains only four broad categories: **Plot Area, Carpet Area, Built-up Area, and Super Built-up Area**. These categories represent different definitions of area and therefore do not have the same meaning in terms of the actual area of an individual house.
* For example, in the case of **Plot Area**, the `total_sqft` value can represent the area of the entire plot/building rather than the actual area of an individual house. Therefore, using `area_type` as an important feature could introduce ambiguity when interpreting the relationship between area and price.
* Instead, the calculated `price_per_sqft` provides a more direct measure of the **price relative to the reported area**, while also capturing the pricing variation associated with different locations.
* Based on this reasoning, the **`area_type` column was removed**, and the newly calculated **`price_per_sqft` feature was retained** for further analysis and model building.
* Note: price_per_sqft was used for EDA and outlier detection 
but excluded from model features to prevent target leakage 
(it's derived from the target variable price).

## Feature Engineering

A derived feature `sqft_per_BHK` was created by dividing `total_sqft` by the number of BHKs. This captures property spaciousness independently of size — two 1,200 sqft properties with 2 BHK and 3 BHK respectively have very different per-room space, which directly impacts price. Existing features alone couldn't distinguish this difference.

## Handling High-Cardinality Location Column

The `location` column had 1,227 unique values. `TargetEncoder` was used to encode location as mean price per location — but locations appearing fewer than 10 times produced unstable encodings, causing the model to extrapolate wildly (one property was predicted at ₹4.85 billion). These rare locations (990 out of 1,227) were bucketed into a single `other_rare_location` category before encoding, which stabilised predictions significantly.

## Log Transformation

Both the target (`price`) and `total_sqft` feature were log-transformed before modelling. Raw house prices and sqft values are heavily right-skewed — log transformation compresses the scale, making the relationship between features and target more linear and improving model performance. Predictions are inverse-transformed using `np.expm1()` to return values in lakhs.

## Performance Metrics

| Model | R2_Score | MAPE |
| :--- | :---: | ---: |
| Linear Regression | 53.87 | 27.81% |
| Random Forest | 65.78 | 24.51% |
| XGBoost Regressor | 59.73 | 24.72% |

## Observations

**Primary metric:** MAPE = 24.51%

Model predictions are typically within **24.51%** of actual price.

**RMSE (80 lakhs)** is skewed by luxury properties (>500 lakhs),  
which represent **<2% of listings** but dominate squared error metrics.

For typical residential properties (**50–200 lakh range**),  
model performs significantly better.

Linear Regression with TargetEncoder produced extreme predictions 
(up to ₹4.85 billion) due to extrapolation beyond training range. 
Tree-based models naturally cap predictions within seen value ranges, 
making them more suitable for this dataset.

## Future Work
- Incorporate proximity features (distance to metro, tech parks, schools)
- Add builder reputation and project quality data
- Scope model to sub-500 lakh properties for higher accuracy
- Experiment with geospatial features using lat/long coordinates