#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
株価予測プロジェクト - ProphetとLightGBMを使用した株価予測システム
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Prophet and LightGBM imports
from prophet import Prophet
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import re

def parse_volume(volume_str):
    """出来高の文字列を数値に変換"""
    if pd.isna(volume_str):
        return 0

    volume_str = str(volume_str).strip()

    # 数値のみの場合
    if volume_str.replace('.', '').replace(',', '').isdigit():
        return float(volume_str.replace(',', ''))

    # M, K, B などの単位を処理
    multipliers = {'K': 1000, 'M': 1000000, 'B': 1000000000}

    for suffix, multiplier in multipliers.items():
        if volume_str.upper().endswith(suffix):
            number = volume_str[:-1]
            try:
                return float(number) * multiplier
            except ValueError:
                return 0

    # その他の場合は0を返す
    return 0

def parse_percentage(pct_str):
    """変化率のパーセンテージ文字列を数値に変換"""
    if pd.isna(pct_str):
        return 0

    pct_str = str(pct_str).strip()

    # %記号を削除して数値に変換
    if pct_str.endswith('%'):
        try:
            return float(pct_str[:-1])
        except ValueError:
            return 0

    try:
        return float(pct_str)
    except ValueError:
        return 0

def load_and_preprocess_data(file_path):
    """データの読み込みと前処理"""
    print("データを読み込み中...")

    # CSVファイルを読み込み
    df = pd.read_csv(file_path)

    print(f"読み込んだデータ: {df.shape[0]}行, {df.shape[1]}列")

    # 日付列を datetime に変換
    df['日付け'] = pd.to_datetime(df['日付け'])

    # 出来高を数値に変換
    df['出来高_数値'] = df['出来高'].apply(parse_volume)

    # 変化率を数値に変換
    df['変化率_数値'] = df['変化率 %'].apply(parse_percentage)

    # データを日付順にソート（古い日付から新しい日付へ）
    df = df.sort_values('日付け').reset_index(drop=True)

    print(f"データ期間: {df['日付け'].min()} から {df['日付け'].max()}")
    print(f"データ点数: {len(df)}日分")

    return df

def create_features_for_lightgbm(df):
    """LightGBM用の特徴量を作成"""
    df_features = df.copy()

    # 移動平均
    df_features['ma_5'] = df_features['終値'].rolling(window=5).mean()
    df_features['ma_10'] = df_features['終値'].rolling(window=10).mean()
    df_features['ma_20'] = df_features['終値'].rolling(window=20).mean()

    # ボラティリティ
    df_features['volatility_5'] = df_features['終値'].rolling(window=5).std()
    df_features['volatility_10'] = df_features['終値'].rolling(window=10).std()

    # 価格レンジ
    df_features['price_range'] = df_features['高値'] - df_features['安値']
    df_features['price_range_pct'] = df_features['price_range'] / df_features['終値'] * 100

    # ラグ特徴量
    for lag in [1, 2, 3, 5, 10]:
        df_features[f'close_lag_{lag}'] = df_features['終値'].shift(lag)
        df_features[f'volume_lag_{lag}'] = df_features['出来高_数値'].shift(lag)

    # 時系列特徴量
    df_features['day_of_week'] = df_features['日付け'].dt.dayofweek
    df_features['month'] = df_features['日付け'].dt.month
    df_features['quarter'] = df_features['日付け'].dt.quarter

    return df_features

def train_prophet_model(train_data):
    """Prophetモデルの訓練"""
    print("Prophetモデルを訓練中...")

    # Prophet用のデータ形式に変換
    prophet_data = pd.DataFrame({
        'ds': train_data['日付け'],
        'y': train_data['終値']
    })

    # Prophetモデルの作成と訓練
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )

    model.fit(prophet_data)

    return model

def train_lightgbm_model(train_data):
    """LightGBMモデルの訓練"""
    print("LightGBMモデルを訓練中...")

    # 特徴量を作成
    train_features = create_features_for_lightgbm(train_data)

    # 特徴量の選択
    feature_columns = [
        '始値', '高値', '安値', '出来高_数値', '変化率_数値',
        'ma_5', 'ma_10', 'ma_20', 'volatility_5', 'volatility_10',
        'price_range', 'price_range_pct', 'day_of_week', 'month', 'quarter'
    ]

    # ラグ特徴量を追加
    for lag in [1, 2, 3, 5, 10]:
        feature_columns.extend([f'close_lag_{lag}', f'volume_lag_{lag}'])

    # NaNを削除
    train_features = train_features.dropna()

    if len(train_features) == 0:
        raise ValueError("訓練データが不足しています")

    X_train = train_features[feature_columns]
    y_train = train_features['終値']

    # LightGBMモデルの訓練
    train_data_lgb = lgb.Dataset(X_train, label=y_train)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }

    model = lgb.train(
        params,
        train_data_lgb,
        num_boost_round=100,
        valid_sets=[train_data_lgb],
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
    )

    return model, feature_columns

def rolling_prediction(df, train_window_years=2, predict_window_weeks=4):
    """ローリング予測の実行"""
    print(f"ローリング予測を実行中... (訓練期間: {train_window_years}年, 予測期間: {predict_window_weeks}週)")

    train_window_days = train_window_years * 365
    predict_window_days = predict_window_weeks * 7

    # 結果を保存するリスト
    results = []

    # 最小訓練データ数を確保
    min_train_size = max(train_window_days, 500)  # 最低500日分のデータ

    # ローリング予測のループ
    start_idx = min_train_size

    while start_idx + predict_window_days < len(df):
        # 訓練データとテストデータの分割
        train_end_idx = start_idx
        test_start_idx = start_idx
        test_end_idx = min(start_idx + predict_window_days, len(df))

        train_data = df.iloc[max(0, train_end_idx - train_window_days):train_end_idx].copy()
        test_data = df.iloc[test_start_idx:test_end_idx].copy()

        if len(train_data) < 100 or len(test_data) == 0:
            start_idx += predict_window_days
            continue

        try:
            # Prophetモデルの訓練と予測
            prophet_model = train_prophet_model(train_data)

            future_dates = pd.DataFrame({
                'ds': test_data['日付け']
            })

            prophet_forecast = prophet_model.predict(future_dates)
            prophet_predictions = prophet_forecast['yhat'].values

            # LightGBMモデルの訓練と予測
            lgb_model, feature_columns = train_lightgbm_model(train_data)

            # テストデータの特徴量作成
            test_features = create_features_for_lightgbm(
                pd.concat([train_data.tail(50), test_data], ignore_index=True)
            ).tail(len(test_data))

            # 特徴量が揃っているデータのみ予測
            test_features_clean = test_features.dropna()

            if len(test_features_clean) > 0:
                X_test = test_features_clean[feature_columns]
                lgb_predictions = lgb_model.predict(X_test)

                # 結果を保存
                for i, (idx, row) in enumerate(test_features_clean.iterrows()):
                    if i < len(prophet_predictions) and i < len(lgb_predictions):
                        results.append({
                            'date': row['日付け'],
                            'actual': row['終値'],
                            'prophet_pred': prophet_predictions[i] if i < len(prophet_predictions) else np.nan,
                            'lightgbm_pred': lgb_predictions[i]
                        })

        except Exception as e:
            print(f"予測エラー (インデックス {start_idx}): {e}")

        # 次のウィンドウに移動
        start_idx += predict_window_days

    return pd.DataFrame(results)

def evaluate_models(results_df):
    """モデルの評価"""
    print("\nモデル評価結果:")

    # 有効な予測値のみを使用
    valid_results = results_df.dropna()

    if len(valid_results) == 0:
        print("有効な予測結果がありません")
        return {}

    # RMSE計算
    prophet_rmse = np.sqrt(mean_squared_error(valid_results['actual'], valid_results['prophet_pred']))
    lightgbm_rmse = np.sqrt(mean_squared_error(valid_results['actual'], valid_results['lightgbm_pred']))

    print(f"Prophet RMSE: {prophet_rmse:.4f}")
    print(f"LightGBM RMSE: {lightgbm_rmse:.4f}")

    return {
        'prophet_rmse': prophet_rmse,
        'lightgbm_rmse': lightgbm_rmse,
        'data_points': len(valid_results)
    }

def create_visualizations(results_df, evaluation_results):
    """結果の可視化"""
    print("グラフを作成中...")

    # 有効な結果のみを使用
    valid_results = results_df.dropna()

    if len(valid_results) == 0:
        print("可視化するデータがありません")
        return

    # 図のサイズを設定
    plt.figure(figsize=(15, 10))

    # 1. 予測結果の比較
    plt.subplot(2, 1, 1)
    plt.plot(valid_results['date'], valid_results['actual'], 
             label='実績値', color='black', linewidth=2)
    plt.plot(valid_results['date'], valid_results['prophet_pred'], 
             label='Prophet予測', color='blue', alpha=0.7)
    plt.plot(valid_results['date'], valid_results['lightgbm_pred'], 
             label='LightGBM予測', color='red', alpha=0.7)

    plt.title('株価予測結果比較 (評価期間)', fontsize=16, fontweight='bold')
    plt.xlabel('日付', fontsize=12)
    plt.ylabel('株価', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # 2. RMSE比較
    plt.subplot(2, 1, 2)
    models = ['Prophet', 'LightGBM']
    rmse_values = [evaluation_results['prophet_rmse'], evaluation_results['lightgbm_rmse']]
    colors = ['blue', 'red']

    bars = plt.bar(models, rmse_values, color=colors, alpha=0.7)
    plt.title('RMSE値比較', fontsize=16, fontweight='bold')
    plt.ylabel('RMSE', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')

    # バーの上に値を表示
    for bar, value in zip(bars, rmse_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('/home/user/output/stock_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("グラフを保存しました: /home/user/output/stock_prediction_results.png")

def main():
    """メイン実行関数"""
    print("=== 株価予測プロジェクト開始 ===")

    try:
        # データの読み込みと前処理
        df = load_and_preprocess_data("stock_price.csv")

        # ローリング予測の実行
        results_df = rolling_prediction(df, train_window_years=2, predict_window_weeks=4)

        if len(results_df) == 0:
            print("予測結果が得られませんでした")
            return

        # モデルの評価
        evaluation_results = evaluate_models(results_df)

        # 結果の可視化
        create_visualizations(results_df, evaluation_results)

        # 結果をCSVファイルに保存
        results_df.to_csv('/home/user/output/prediction_results.csv', index=False, encoding='utf-8')
        print("予測結果を保存しました: /home/user/output/prediction_results.csv")

        print("\n=== 株価予測プロジェクト完了 ===")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
