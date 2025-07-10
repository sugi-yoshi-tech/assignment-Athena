import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Noto Sans JP を設定
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False # マイナス記号の文字化け防止

print("matplotlibで日本語が表示できるように設定しました。")

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')


# --- 基本設定 ---
# スタイルの設定
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Hiragino Sans' # Mac用の日本語フォント
# Windowsの場合は 'Yu Gothic' や 'MS Gothic' などを試してください
# plt.rcParams['font.family'] = 'Yu Gothic'
plt.rcParams['axes.unicode_minus'] = False # マイナス記号の文字化け防止

# --- データの読み込みと前処理 ---
try:
    # Corrected file path based on user input
    df = pd.read_csv('/content/drive/MyDrive/Athena/ett.csv', parse_dates=['date'], index_col='date')
except FileNotFoundError:
    print("エラー: ett.csvファイルが見つかりません。Google Driveをマウントし、パスを確認してください。")
    # exit() # Remove exit() to allow subsequent code to run if df is defined elsewhere (though not the case here)
    # If the file is not found, df will not be defined and subsequent code will fail.
    # It's better to let the NameError occur naturally if the file is not found after mounting.
    pass # Keep the try...except for informative error message but don't exit

# Check if df was loaded
if 'df' not in locals():
    print("エラー: ett.csvファイルが見つ込まれず、データフレーム'df'が作成できませんでした。処理を中断します。")
else:
    # 欠損値の確認と補間 (前方補間)
    df.ffill(inplace=True)
    print("--- データ読み込み完了 ---")
    print(f"データ期間: {df.index.min()} から {df.index.max()} まで")
    print(f"データポイント数: {len(df)}")
    print(f"補間後の欠損値の数: {df.isnull().sum().sum()}")


    # ==============================================================================
    # モデル1: 線形回帰モデル (ラグ特徴量) - この部分は変更しません
    # ==============================================================================
    print("\n" + "="*60)
    print("モデル1: 線形回帰モデルの処理を開始")
    print("="*60)

    # --- 特徴量エンジニアリング (線形回帰用) ---
    def create_linear_features(dataframe, lag_steps):
        df_featured = dataframe.copy()
        for lag in range(1, lag_steps + 1):
            df_featured[f'OT_lag_{lag}'] = df_featured['OT'].shift(lag)
        df_featured['hour'] = df_featured.index.hour
        df_featured['dayofweek'] = df_featured.index.dayofweek
        df_featured['month'] = df_featured.index.month
        return df_featured

    lag_steps_lr = 24
    df_lr = create_linear_features(df, lag_steps_lr)
    df_lr.dropna(inplace=True)

    # --- モデルのトレーニング (線形回帰) ---
    features_lr = [col for col in df_lr.columns if col != 'OT']
    X_lr = df_lr[features_lr]
    y_lr = df_lr['OT']

    split_point_lr = int(len(X_lr) * 0.8)
    X_train_lr, X_test_lr = X_lr[:split_point_lr], X_lr[split_point_lr:]
    y_train_lr, y_test_lr = y_lr[:split_point_lr], y_lr[split_point_lr:]

    model_lr = LinearRegression()
    model_lr.fit(X_train_lr, y_train_lr)

    # --- モデルの評価 (線形回帰) ---
    y_pred_lr = model_lr.predict(X_test_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test_lr, y_pred_lr))
    print(f"\n線形回帰モデルの評価 (RMSE): {rmse_lr:.4f}")

    # --- 結果の可視化 (線形回帰) ---
    plt.figure(figsize=(15, 7))
    plt.plot(y_test_lr.index, y_test_lr, label='実際の値 (Actual)', color='blue', alpha=0.8)
    plt.plot(y_test_lr.index, y_pred_lr, label='予測値 (Linear Regression)', color='orange', linestyle='--', alpha=0.8)
    plt.title('線形回帰モデルによるオイル温度の予測結果', fontsize=16)
    plt.xlabel('日付', fontsize=12)
    plt.ylabel('オイル温度 (OT)', fontsize=12)
    plt.legend()
    plt.grid(True)
    # plt.show() # NOTE: 最後にまとめて表示するためコメントアウト


    # ==============================================================================
    # モデル2: LSTMモデル (Sliding Window 学習・最後の2週間で評価)
    # ==============================================================================
    print("\n" + "="*60)
    print("モデル2: LSTMモデルの処理を開始 (Sliding Window 学習・最後の2週間で評価)")
    print("="*60)

    # --- データの前処理 (LSTM用) ---
    data_lstm = df[['OT']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_lstm)

    # 予測したい期間（2週間 = 336時間）
    forecast_period_hours = 14 * 24
    # 学習に使う過去の期間（1年間 = 365日 = 8760時間）
    look_back_period_hours = 365 * 24

    # テストデータの開始インデックス (データの最後から forecast_period_hours 分前)
    test_start_index = len(scaled_data) - forecast_period_hours

    # Sliding Window で学習データセットを作成する関数
    def create_sliding_window_dataset(dataset, look_back, forecast_horizon):
        dataX, dataY = [], []
        # 学習データはテストデータの開始位置より前で sliding window を適用
        # dataset[:test_start_index] に対して処理を行う
        for i in range(len(dataset[:test_start_index]) - look_back - forecast_horizon + 1):
            x = dataset[i:(i + look_back), 0]
            y = dataset[i + look_back : i + look_back + forecast_horizon, 0] # forecast_horizon 期間を予測
            dataX.append(x)
            dataY.append(y)
        return np.array(dataX), np.array(dataY)

    # テストデータセットを作成する関数
    def create_test_dataset(dataset, look_back, forecast_horizon, start_index):
        dataX, dataY = [], []
        # テストデータは start_index から look_back + forecast_horizon 期間を使用
        # ループは start_index から開始し、予測期間の終端までデータがある範囲で終了する
        for i in range(start_index, len(dataset) - forecast_horizon + 1):
             x = dataset[i - look_back:i, 0]
             y = dataset[i:i + forecast_horizon, 0]
             dataX.append(x)
             dataY.append(y)
        return np.array(dataX), np.array(dataY)


    # 学習データセットの作成
    # 学習データはテストデータの開始位置より前で sliding window を適用
    X_train, y_train = create_sliding_window_dataset(scaled_data, look_back_period_hours, forecast_period_hours)

    # テストデータセットの作成
    # テストデータは最後の look_back_period_hours + forecast_period_hours 期間を使用
    # test_start_index は forecast_period_hours 分後ろから数えている
    # テストデータセットの開始は test_start_index から
    X_test, y_test = create_test_dataset(scaled_data, look_back_period_hours, forecast_period_hours, test_start_index)


    # LSTMモデルへの入力形状にリシェイプ
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    # y_train は forecast_period_hours の長さを持つシーケンスなので、形状は (samples, forecast_horizon) のまま
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    # y_test も (samples, forecast_horizon) のまま

    print(f"LSTMトレーニングデータ形状 (X): {X_train.shape}")
    print(f"LSTMトレーニングデータ形状 (y): {y_train.shape}")
    print(f"LSTMテストデータ形状 (X): {X_test.shape}")
    print(f"LSTMテストデータ形状 (y): {y_test.shape}")


    # --- モデルの構築とトレーニング (LSTM) ---
    # 出力層のユニット数を forecast_period_hours に変更
    model_lstm = Sequential([
        LSTM(50, return_sequences=True, input_shape=(look_back_period_hours, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(forecast_period_hours) # 予測期間と同じユニット数
    ])
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("\nLSTMモデルのトレーニングを開始します...")
    history = model_lstm.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50, batch_size=64, verbose=1,
        callbacks=[early_stopping]
    )

    # --- モデルの評価 (LSTM) ---
    test_predict = model_lstm.predict(X_test)

    # スケーリングを元に戻す
    # test_predict と y_test は (samples, forecast_horizon) の形状
    # scaler.inverse_transform は (samples, features) または (features,) の形状
    # ここでは各予測時間ステップに対して独立にスケーリングを元に戻す
    test_predict_orig = scaler.inverse_transform(test_predict)
    y_test_orig = scaler.inverse_transform(y_test)

    # 評価指標の計算（各時間ステップの平均誤差など）
    # RMSEとMAEを計算するために、y_test_origとtest_predict_origをフラット化
    rmse_lstm = np.sqrt(mean_squared_error(y_test_orig.flatten(), test_predict_orig.flatten()))
    mae_lstm = mean_absolute_error(y_test_orig.flatten(), test_predict_orig.flatten())
    print(f"\nLSTMモデル(評価用)の評価 (テストデータ RMSE): {rmse_lstm:.4f}")
    print(f"LSTMモデル(評価用)の評価 (テストデータ MAE): {mae_lstm:.4f}")

    # --- 結果の可視化 (LSTM) ---
    # 可視化のために、最後のテストデータサンプルを使用
    # X_test の最後のサンプルに対応する期間の実際の値と予測値を取得
    last_test_sample_actual = y_test_orig[-1]
    last_test_sample_predict = test_predict_orig[-1]

    # テスト期間のインデックスを作成
    # テストデータは scaled_data の test_start_index から始まっている
    # 予測期間は test_start_index から test_start_index + forecast_period_hours まで
    test_dates = df.index[test_start_index : test_start_index + forecast_period_hours]


    plt.figure(figsize=(15, 7))
    # 過去のデータ (テスト期間の直前の look_back_period_hours 分)
    # test_start_index は forecast_period_hours 分後ろから数えているので、
    # 過去のデータの開始位置は test_start_index - look_back_period_hours
    historical_dates = df.index[test_start_index - look_back_period_hours : test_start_index]
    plt.plot(historical_dates, df['OT'][historical_dates], label=f'過去の実測値 ({look_back_period_hours}時間)', color='blue')

    # テスト期間の実際の値
    plt.plot(test_dates, last_test_sample_actual, label='テスト期間の実測値 (Actual Test)', color='green', alpha=0.8)
    # テスト期間の予測値
    plt.plot(test_dates, last_test_sample_predict, label='テスト期間の予測値 (LSTM Prediction)', color='red', linestyle='--', alpha=0.8)

    plt.title('LSTMモデルによるオイル温度の予測結果 (最後の2週間)', fontsize=16)
    plt.xlabel('日付', fontsize=12)
    plt.ylabel('オイル温度 (OT)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()