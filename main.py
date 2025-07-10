import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from prophet import Prophet
import lightgbm as lgb
import japanize_matplotlib # Ensure Japanese characters are displayed correctly

# Load the stock price data
df = pd.read_csv("/content/drive/MyDrive/Athena/stock_price.csv")

# Convert '日付け' column to datetime objects
df['日付け'] = pd.to_datetime(df['日付け'], format='%Y-%m-%d')

# Function to parse volume data (e.g., 'K', 'M', 'B' suffixes)
def parse_volume(text):
    multipliers = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000}
    if isinstance(text, str) and text[-1] in multipliers:
        return float(text[:-1]) * multipliers[text[-1]]
    return float(text)

# Apply parsing to '出来高' and '変化率 %' columns
df["出来高"] = [parse_volume(a) for a in df["出来高"]]
df["変化率 %"] = [float(a.replace("%", "")) for a in df["変化率 %"]]

# Reverse the DataFrame to have the oldest dates first (ascending order)
df = df.iloc[::-1].reset_index(drop=True)

# Split the data into training and testing sets (80% train, 20% test)
train_size = int(len(df) * 0.8)
train_df = df.iloc[0:train_size].copy()
test_df = df.iloc[train_size:len(df)].copy()

print(f"元のデータ数: {len(df)}")
print(f"訓練データ数: {len(train_df)}")
print(f"テストデータ数: {len(test_df)}")

# Set rolling window parameters
train_window_years = 2
predict_window_weeks = 2

# Calculate the start date for the 2-year evaluation period within the training data
two_years_before_test_start = test_df['日付け'].iloc[0] - pd.DateOffset(years=train_window_years)

# Ensure the start date is within the training data range
evaluation_start_date = max(train_df['日付け'].iloc[0], two_years_before_test_start)

# Prepare the evaluation dataset (last 2 years of train_df + test_df)
evaluation_df = df[(df['日付け'] >= evaluation_start_date)].copy()

print(f"\n評価期間の開始日: {evaluation_start_date.strftime('%Y-%m-%d')}")
print(f"評価データ数: {len(evaluation_df)}")

# 1. Create an empty list to store Prophet predictions
prophet_rolling_predictions = []

# 2. Set up a loop to iterate through the evaluation period
# Start the loop from the first date in the evaluation_df
current_prediction_start_date = evaluation_df['日付け'].iloc[0]

# The loop continues until the prediction window extends beyond the end of the evaluation data
# We predict 2 weeks ahead, so the last possible start date is 2 weeks before the end of evaluation_df
evaluation_end_date = evaluation_df['日付け'].iloc[-1]
predict_window_days = predict_window_weeks * 7

# Define the end date of the last possible prediction window
last_prediction_end_date = evaluation_end_date

print("\n--- Prophetモデルによるローリングウィンドウ予測を開始 ---")

while current_prediction_start_date <= last_prediction_end_date:

    # 3. Define training data and prediction period for the current window
    # Training data is the last 'train_window_years' up to the start of the prediction period
    train_end_date = current_prediction_start_date - timedelta(days=1)
    train_start_date = train_end_date - pd.DateOffset(years=train_window_years)

    # Ensure train_start_date is not before the beginning of the entire dataset
    train_start_date = max(train_start_date, df['日付け'].iloc[0])

    current_train_df = df[(df['日付け'] >= train_start_date) & (df['日付け'] <= train_end_date)].copy()

    # Prediction period is 'predict_window_weeks' starting from current_prediction_start_date
    current_prediction_end_date = current_prediction_start_date + timedelta(days=predict_window_days - 1)

    # Ensure prediction end date does not go beyond the overall data end date
    current_prediction_end_date = min(current_prediction_end_date, df['日付け'].iloc[-1])

    # Filter the dates within the current prediction window that are actually in the original dataframe
    # This handles potential gaps in the data (e.g., weekends, holidays)
    future_dates_df = df[(df['日付け'] >= current_prediction_start_date) & (df['日付け'] <= current_prediction_end_date)][['日付け']].copy()

    if current_train_df.empty or future_dates_df.empty:
        print(f"Skipping window starting {current_prediction_start_date.strftime('%Y-%m-%d')} due to insufficient data.")
        # Move to the next prediction window start date
        current_prediction_start_date += timedelta(days=predict_window_days)
        continue

    # 4. Fit a new Prophet model instance
    # Convert training data to Prophet format
    current_train_prophet_df = current_train_df.rename(columns={'日付け': 'ds', '終値': 'y'})

    model_prophet_rolling = Prophet()
    model_prophet_rolling.fit(current_train_prophet_df)

    # 5. Make predictions for the defined prediction period
    # Convert future dates to Prophet format
    future_prophet_df = future_dates_df.rename(columns={'日付け': 'ds'})
    forecast = model_prophet_rolling.predict(future_prophet_df)

    # 6. Extract predictions and add to the list
    # Only keep 'ds' and 'yhat' from the forecast and append
    prophet_rolling_predictions.extend(forecast[['ds', 'yhat']].to_dict('records'))

    # Move to the next prediction window start date
    current_prediction_start_date += timedelta(days=predict_window_days)

print("--- Prophetモデルによるローリングウィンドウ予測が完了 ---")

# 7. Convert the collected predictions to a Pandas DataFrame
prophet_rolling_predictions_df = pd.DataFrame(prophet_rolling_predictions)

# 8. Sort the DataFrame by date and set the index to match the evaluation_df dates
prophet_rolling_predictions_df['ds'] = pd.to_datetime(prophet_rolling_predictions_df['ds'])
prophet_rolling_predictions_df.sort_values(by='ds', inplace=True)

# It's better to merge or reindex based on evaluation_df dates to ensure alignment
# First, filter predictions to only include dates present in evaluation_df
prophet_rolling_predictions_df = prophet_rolling_predictions_df[
    prophet_rolling_predictions_df['ds'].isin(evaluation_df['日付け'])
].copy()


# Now set the index based on the 'ds' column which contains the date
prophet_rolling_predictions_df.set_index('ds', inplace=True)

# Reindex the predictions DataFrame to match the index of evaluation_df to ensure proper alignment for later comparison
# This will introduce NaNs for dates in evaluation_df that were not in the prediction windows
prophet_rolling_predictions_df = prophet_rolling_predictions_df.reindex(evaluation_df['日付け'])

# The target variable from evaluation_df for comparison
prophet_actual_values = evaluation_df.set_index('日付け')['終値'].copy()

# Ensure the index names match if necessary
prophet_rolling_predictions_df.index.name = '日付け'
prophet_actual_values.index.name = '日付け'

# Combine actual and predicted for easy viewing and RMSE calculation
prophet_evaluation_comparison = pd.DataFrame({
    '実績値': prophet_actual_values,
    'Prophet 予測値': prophet_rolling_predictions_df['yhat']
}).dropna(subset=['実績値']) # Drop rows where actual values are missing

# 9. Store the final Prophet rolling window prediction results
# The prophet_rolling_predictions_df and prophet_evaluation_comparison DataFrames now hold the results

# 1. Create an empty list to store LightGBM predictions
lgbm_rolling_predictions = []

# Set up the rolling window loop parameters again, based on evaluation_df
current_prediction_start_date = evaluation_df['日付け'].iloc[0]
evaluation_end_date = evaluation_df['日付け'].iloc[-1]
predict_window_days = predict_window_weeks * 7 # Should be 14

print("\n--- LightGBMモデルによるローリングウィンドウ予測を開始 ---")

while current_prediction_start_date <= evaluation_end_date:

    # 3. Define training data and prediction period for the current window
    # Training data is the last 'train_window_years' up to the start of the prediction period
    train_end_date = current_prediction_start_date - timedelta(days=1)
    train_start_date = train_end_date - pd.DateOffset(years=train_window_years)

    # Ensure train_start_date is not before the beginning of the entire dataset
    train_start_date = max(train_start_date, df['日付け'].iloc[0])

    current_train_df = df[(df['日付け'] >= train_start_date) & (df['日付け'] <= train_end_date)].copy()

    # Prediction period is 'predict_window_weeks' starting from current_prediction_start_date
    current_prediction_end_date = current_prediction_start_date + timedelta(days=predict_window_days - 1)

    # Ensure prediction end date does not go beyond the overall data end date
    current_prediction_end_date = min(current_prediction_end_date, df['日付け'].iloc[-1])

    # Filter the dates within the current prediction window that are actually in the original dataframe
    # This handles potential gaps in the data (e.g., weekends, holidays)
    future_dates_df = df[(df['日付け'] >= current_prediction_start_date) & (df['日付け'] <= current_prediction_end_date)][['日付け']].copy()

    if current_train_df.empty or future_dates_df.empty:
        print(f"Skipping window starting {current_prediction_start_date.strftime('%Y-%m-%d')} due to insufficient data.")
        # Move to the next prediction window start date
        current_prediction_start_date += timedelta(days=predict_window_days)
        continue

    # 4. Implement feature engineering for the current training data and future dates
    # Combine current train data with a placeholder for future dates for feature engineering consistency
    # This is important for lag and rolling window features
    current_data_for_features = pd.concat([current_train_df[['日付け', '終値']], future_dates_df.rename(columns={'日付け': 'ds'}).rename(columns={'ds': '日付け'}).assign(終値=np.nan)], ignore_index=True)

    # Sort by date
    current_data_for_features.sort_values(by='日付け', inplace=True)

    # Create lag features ('終値' - Closing Price)
    for i in [1, 7, 30]:
        current_data_for_features[f'終値_lag_{i}'] = current_data_for_features['終値'].shift(i)

    # Create moving average features ('終値' - Closing Price)
    for i in [7, 30]:
        current_data_for_features[f'終値_rolling_mean_{i}'] = current_data_for_features['終値'].rolling(window=i).mean()

    # Create date-related features
    current_data_for_features['year'] = current_data_for_features['日付け'].dt.year
    current_data_for_features['month'] = current_data_for_features['日付け'].dt.month
    current_data_for_features['day'] = current_data_for_features['日付け'].dt.day
    current_data_for_features['dayofweek'] = current_data_for_features['日付け'].dt.dayofweek

    # Split back into training and future dataframes after feature engineering
    # Training data is where '終値' is not NaN
    current_train_featurized = current_data_for_features.dropna(subset=['終値']).copy()

    # Future data is where '終値' is NaN (these are the dates we want to predict)
    current_future_featurized = current_data_for_features[current_data_for_features['終値'].isnull()].copy()

    # Drop rows with NaN values created by feature engineering in the training data
    current_train_featurized.dropna(inplace=True)

    # Ensure future featurized data also has all required feature columns, fill NaNs where appropriate for prediction
    # Use the mean of the training data for filling NaNs in future features that weren't created by lag/rolling
    # Example: if lag features at the very beginning of the future_dates_df are NaN
    for col in current_train_featurized.columns:
        if col not in ['日付け', '終値'] and current_future_featurized[col].isnull().any():
            mean_val = current_train_featurized[col].mean()
            current_future_featurized[col].fillna(mean_val, inplace=True)


    # 5. Split featurized training data into features (X) and target (y)
    features = [col for col in current_train_featurized.columns if col not in ['日付け', '終値', '月']] # Exclude date, target and '月'
    target = '終値'

    X_train_lgbm = current_train_featurized[features]
    y_train_lgbm = current_train_featurized[target]
    X_future_lgbm = current_future_featurized[features] # Features for prediction

    if X_train_lgbm.empty or X_future_lgbm.empty:
         print(f"Skipping window starting {current_prediction_start_date.strftime('%Y-%m-%d')} due to empty feature sets after engineering.")
         # Move to the next prediction window start date
         current_prediction_start_date += timedelta(days=predict_window_days)
         continue


    # 6. Initialize LightGBM model (already imported lgb)
    lgbm_model_rolling = lgb.LGBMRegressor(random_state=42)

    # 7. Train the LightGBM model
    lgbm_model_rolling.fit(X_train_lgbm, y_train_lgbm)

    # 8. Make predictions on the future dates
    lgbm_window_predictions = lgbm_model_rolling.predict(X_future_lgbm)

    # 9. Extract predicted values and their corresponding dates
    # Combine dates from current_future_featurized with predictions
    window_predictions_df = pd.DataFrame({
        'ds': current_future_featurized['日付け'],
        'yhat': lgbm_window_predictions
    })

    # 10. Append predictions to the list
    lgbm_rolling_predictions.extend(window_predictions_df.to_dict('records'))


    # Move to the next prediction window start date
    current_prediction_start_date += timedelta(days=predict_window_days)

print("--- LightGBMモデルによるローリングウィンドウ予測が完了 ---")

# 11. Convert the collected predictions to a Pandas DataFrame
lgbm_rolling_predictions_df = pd.DataFrame(lgbm_rolling_predictions)

# 12. Sort the DataFrame by date
lgbm_rolling_predictions_df['ds'] = pd.to_datetime(lgbm_rolling_predictions_df['ds'])
lgbm_rolling_predictions_df.sort_values(by='ds', inplace=True)

# 13. Filter the LightGBM predictions DataFrame to include only dates present in the evaluation_df
lgbm_rolling_predictions_df = lgbm_rolling_predictions_df[
    lgbm_rolling_predictions_df['ds'].isin(evaluation_df['日付け'])
].copy()


# 14. Set the date column ('ds') as the index of the LightGBM predictions DataFrame.
lgbm_rolling_predictions_df.set_index('ds', inplace=True)

# 15. Reindex the LightGBM predictions DataFrame to match the index of evaluation_df
lgbm_rolling_predictions_df = lgbm_rolling_predictions_df.reindex(evaluation_df['日付け'])

# 16. Store the final LightGBM rolling window prediction results
# lgbm_rolling_predictions_df now holds the results.

# 1. Calculate RMSE for Prophet and LightGBM over the evaluation period

# Ensure evaluation_df has '日付け' as index for easy alignment with predictions
evaluation_df_indexed = evaluation_df.set_index('日付け')

# Calculate RMSE for Prophet
# Align Prophet predictions with the actual values in evaluation_df_indexed
prophet_evaluation_aligned = evaluation_df_indexed.join(prophet_rolling_predictions_df, how='left')
prophet_evaluation_aligned.rename(columns={'終値': '実績値', 'yhat': 'Prophet 予測値'}, inplace=True)

# Drop rows with missing values in either actual or predicted for RMSE calculation
prophet_evaluation_cleaned = prophet_evaluation_aligned.dropna(subset=['実績値', 'Prophet 予測値'])

if not prophet_evaluation_cleaned.empty:
    rmse_prophet_rolling = np.sqrt(mean_squared_error(prophet_evaluation_cleaned['実績値'], prophet_evaluation_cleaned['Prophet 予測値']))
    print(f"評価期間におけるProphetモデルのRMSE: {rmse_prophet_rolling:.3f}")
else:
    rmse_prophet_rolling = np.nan
    print("Prophetモデルの評価データが不足しているため、RMSEは計算できません。")


# Calculate RMSE for LightGBM
# Align LightGBM predictions with the actual values in evaluation_df_indexed
lgbm_evaluation_aligned = evaluation_df_indexed.join(lgbm_rolling_predictions_df, how='left')
lgbm_evaluation_aligned.rename(columns={'終値': '実績値', 'yhat': 'LightGBM 予測値'}, inplace=True)

# Drop rows with missing values in either actual or predicted for RMSE calculation
lgbm_evaluation_cleaned = lgbm_evaluation_aligned.dropna(subset=['実績値', 'LightGBM 予測値'])

if not lgbm_evaluation_cleaned.empty:
    rmse_lgbm_rolling = np.sqrt(mean_squared_error(lgbm_evaluation_cleaned['実績値'], lgbm_evaluation_cleaned['LightGBM 予測値']))
    print(f"評価期間におけるLightGBMモデルのRMSE: {rmse_lgbm_rolling:.3f}")
else:
    rmse_lgbm_rolling = np.nan
    print("LightGBMモデルの評価データが不足しているため、RMSEは計算できません。")


# 2. Create a combined DataFrame for plotting actual vs. predicted values on the evaluation period
# Use evaluation_df_indexed as the base to ensure all evaluation dates are included
comparison_evaluation_df = evaluation_df_indexed[['終値']].copy()
comparison_evaluation_df.rename(columns={'終値': '実績値'}, inplace=True)

# Join the predictions from both models
comparison_evaluation_df = comparison_evaluation_df.join(prophet_rolling_predictions_df.rename(columns={'yhat': 'Prophet 予測値'}), how='left')
comparison_evaluation_df = comparison_evaluation_df.join(lgbm_rolling_predictions_df.rename(columns={'yhat': 'LightGBM 予測値'}), how='left')

# Ensure index is datetime for plotting
comparison_evaluation_df.index = pd.to_datetime(comparison_evaluation_df.index)
comparison_evaluation_df.sort_index(inplace=True) # Sort by date


# Plot actual vs. predicted values for the evaluation period
plt.figure(figsize=(15, 8))
plt.plot(comparison_evaluation_df.index, comparison_evaluation_df['実績値'], label='実績値', color='green', linewidth=2)
if 'Prophet 予測値' in comparison_evaluation_df.columns and not comparison_evaluation_df['Prophet 予測値'].isnull().all():
    plt.plot(comparison_evaluation_df.index, comparison_evaluation_df['Prophet 予測値'], label='Prophet 予測値', color='purple', linestyle=':')
if 'LightGBM 予測値' in comparison_evaluation_df.columns and not comparison_evaluation_df['LightGBM 予測値'].isnull().all():
     plt.plot(comparison_evaluation_df.index, comparison_evaluation_df['LightGBM 予測値'], label='LightGBM 予測値', color='orange', linestyle='-.')

plt.title('株価予測結果比較 (評価期間): 実績値 vs. 各モデルによる予測値')
plt.xlabel('日付け')
plt.ylabel('終値')
plt.legend()
plt.grid(True)
plt.show()


# 3. Create a bar chart to visualize the RMSE values
rmse_values_rolling = {'Prophet': rmse_prophet_rolling,
                       'LightGBM': rmse_lgbm_rolling}

rmse_df_rolling = pd.DataFrame(list(rmse_values_rolling.items()), columns=['Model', 'RMSE'])
rmse_df_rolling.set_index('Model', inplace=True)
rmse_df_rolling.dropna(inplace=True) # Drop models with missing RMSE values

if not rmse_df_rolling.empty:
    plt.figure(figsize=(8, 5))
    rmse_df_rolling['RMSE'].plot(kind='bar', color=['purple', 'orange'])
    plt.title('モデル別RMSE比較 (評価期間)')
    plt.xlabel('モデル')
    plt.ylabel('RMSE')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.show()
else:
    print("RMSE DataFrame is empty, cannot generate the RMSE comparison bar chart.")