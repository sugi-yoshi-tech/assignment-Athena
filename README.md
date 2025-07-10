### 時系列予測プロジェクト
このプロジェクトは、ett.csv ファイルに含まれる時系列データを使用して、線形回帰モデルとLSTM (Long Short-Term Memory) モデルによる予測を行うPythonスクリプトです。予測結果はグラフで可視化され、モデルの評価指標（RMSE、MAE）も表示されます。

### 特徴
データ読み込みと前処理: CSVファイルからデータを読み込み、日付時刻インデックスへの変換、欠損値処理、特徴量エンジニアリング、データのスケーリングを行います。

線形回帰モデル: 過去のデータポイントに基づいて単純な線形回帰予測を行います。

LSTMモデル: 時系列予測に特化したディープラーニングモデルであるLSTMを構築し、訓練、評価、予測を行います。

結果の可視化: 実際の値と予測値をグラフで比較し、予測の精度を視覚的に確認できます。

モデル評価: RMSE (Root Mean Squared Error) と MAE (Mean Absolute Error) を用いてモデルの性能を評価します。

Google Drive連携: Google Colab環境での使用を想定し、Google Driveからのデータ読み込みに対応しています。

セットアップ
必要なライブラリ
このプロジェクトを実行するには、以下のPythonライブラリが必要です。

pip install pandas numpy scikit-learn matplotlib seaborn tensorflow


### データの準備
プロジェクトのルートディレクトリに ett.csv という名前のCSVファイルを配置してください。このファイルは、日付時刻の列と、予測対象となる時系列データ（例: OT 列）を含む必要があります。

CSVファイルの例:

date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
2016-07-01 00:00:00,5.827,2.009,1.598,0.462,4.203,1.340,30.531
2016-07-01 01:00:00,5.692,2.075,1.491,0.425,4.142,1.371,27.787
...


Google Colabでの使用
Google Colabで実行する場合、以下のコードでGoogle Driveをマウントし、ett.csv ファイルへのパスを適切に設定してください。

from google.colab import drive
drive.mount('/content/drive')
# df = pd.read_csv('/content/drive/MyDrive/Athena/ett.csv', parse_dates=['date'], index_col='date')
```main.py` ファイル内の`df = pd.read_csv('/content/drive/MyDrive/Athena/ett.csv', parse_dates=['date'], index_col='date')`の行が、Google Drive上のファイルのパスと一致していることを確認してください。

## 実行方法

`main.py` スクリプトを直接実行します。

```bash
python main.py


スクリプトは以下の処理を実行します。

ett.csv からデータを読み込みます。

線形回帰モデルを訓練し、予測と評価を行います。

LSTMモデル用にデータを準備し、モデルを訓練し、予測と評価を行います。

線形回帰とLSTMモデルの予測結果をグラフで表示します。

各モデルのRMSEとMAEをコンソールに出力します。

設定
main.py ファイル内の以下の変数を変更することで、モデルの動作を調整できます。

look_back_period_hours: 予測に使用する過去のデータポイントの数（時間）。(デフォルト: 24)

forecast_period_hours: 予測する未来のデータポイントの数（時間）。

train_test_split_ratio: 訓練データとテストデータの分割比率。

lstm_epochs: LSTMモデルの訓練エポック数。

lstm_batch_size: LSTMモデルの訓練バッチサイズ。

結果
スクリプトの実行後、線形回帰とLSTMモデルによる予測結果を示すプロットが生成されます。また、コンソールには各モデルのRMSEとMAEが出力されます。

例: プロットの説明
線形回帰モデルの予測: 実際のデータと線形回帰モデルによる予測が青とオレンジの線で表示されます。

LSTMモデルの予測: 実際のデータ、過去のデータ、およびLSTMモデルによる未来の予測が異なる色で表示されます。

これらのプロットと評価指標を通じて、各モデルの性能と、時系列予測における線形モデルとディープラーニングモデルの違いを理解することができます。