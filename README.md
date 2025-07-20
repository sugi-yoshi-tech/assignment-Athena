# 📈 株価予測プロジェクト

ProphetとLightGBMを使用した機械学習による株価予測システム

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-completed-success.svg)

## 🎯 プロジェクト概要

このプロジェクトは、37年間（1987年2月〜2024年8月）の株価データを使用して、**Prophet**と**LightGBM**の2つの機械学習モデルによる株価予測システムを構築・比較したものです。

### 主な特徴
- 📊 **長期データ活用**: 9,202日分の株価データを使用
- 🤖 **2つのアプローチ**: 時系列専用モデル（Prophet）と勾配ブースティング（LightGBM）
- 🔄 **ローリング予測**: 2年間の訓練データで4週間先を予測する実用的な評価手法
- 📈 **包括的な評価**: RMSE指標による定量的な性能比較

## 🏆 主要な結果

| モデル | RMSE | 性能 |
|--------|------|------|
| **LightGBM** | **3.3706** | 🥇 最優秀 |
| Prophet | 9.7431 | 🥈 |

**LightGBMモデルが約65%優れた予測精度を達成！**

## 🛠️ 技術スタック

- **Python 3.8+**
- **Prophet**: Facebook開発の時系列予測ライブラリ
- **LightGBM**: Microsoft開発の勾配ブースティングフレームワーク
- **pandas**: データ操作・分析
- **matplotlib**: データ可視化
- **scikit-learn**: 機械学習ユーティリティ
- **japanize-matplotlib**: 日本語フォント対応

## 📦 インストール

### 1. リポジトリのクローン
```bash
git clone https://github.com/your-username/stock-prediction-project.git
cd stock-prediction-project
```

### 2. 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 3. データファイルの配置
`stock_price.csv` ファイルをプロジェクトルートに配置してください。

## 🚀 使用方法

### 基本的な実行
```bash
python main.py
```

## 📁 プロジェクト構造

```
stock-prediction-project/
│
├── main.py                    # メインの予測スクリプト
├── presentation.pptx          # プレゼンテーション資料
├── requirements.txt           # 依存関係
├── README.md                  # プロジェクト説明
├── stock_price.csv           # 株価データ（要配置）
│
└── output/                   # 出力ファイル
    ├── prediction_results.csv
    └── stock_prediction_results.png
```

## 📊 データ仕様

### 入力データ形式
CSVファイルに以下の列が必要です：

| 列名 | 説明 | 例 |
|------|------|-----|
| 日付け | 取引日 | 2024-08-01 |
| 終値 | 終値 | 156.3 |
| 始値 | 始値 | 159.3 |
| 高値 | 高値 | 159.4 |
| 安値 | 安値 | 156.1 |
| 出来高 | 出来高 | 79.15M |
| 変化率 % | 変化率 | -2.56% |

## 🔧 技術的詳細

### Prophet モデル
- **季節性**: 日次・週次・年次の自動検出
- **トレンド**: 変化点の自動識別
- **堅牢性**: 欠損値や外れ値に対する耐性

### LightGBM モデル
- **特徴量エンジニアリング**:
  - 移動平均（5日、10日、20日）
  - ボラティリティ指標（5日、10日）
  - ラグ特徴量（1-10日）
  - 時系列特徴量（曜日、月、四半期）
- **最適化**: 早期停止とハイパーパラメータ調整

### ローリング予測手法
```
訓練期間: 2年間（730日）
予測期間: 4週間（28日）
評価方法: 時系列を順次移動しながら予測精度を評価
```

## 📈 結果の解釈

### 予測精度比較
- **LightGBM**: RMSE 3.3706 - 多様な特徴量を効果的に活用
- **Prophet**: RMSE 9.7431 - 季節性の捉え方は優秀だが短期変動への対応が限定的

### 可視化出力
1. **予測結果比較グラフ**: 実績値vs予測値の時系列プロット
2. **RMSE比較棒グラフ**: モデル間の性能比較

## 🔄 カスタマイズ

### パラメータ調整
`main.py`内の以下の変数を変更可能：

```python
# ローリング予測の設定
train_window_years = 2    # 訓練期間（年）
predict_window_weeks = 4  # 予測期間（週）
```

### 特徴量の追加
`create_features_for_lightgbm()`関数内で新しい特徴量を追加できます。

## 🚀 今後の改善案

### モデル改善
- [ ] アンサンブル手法の導入
- [ ] 深層学習モデル（LSTM、Transformer）の実装
- [ ] ハイパーパラメータの自動最適化
- [ ] クロスバリデーション戦略の改善

### データ拡張
- [ ] マクロ経済指標の追加
- [ ] ニュース感情分析の組み込み
- [ ] 他の金融商品との相関分析
- [ ] リアルタイムデータ取得機能

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルをご覧ください。

## 🤝 コントリビューション

プルリクエストや課題報告を歓迎します！

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/AmazingFeature`)
3. 変更をコミット (`git commit -m 'Add some AmazingFeature'`)
4. ブランチにプッシュ (`git push origin feature/AmazingFeature`)
5. プルリクエストを作成

## 📞 お問い合わせ

プロジェクトに関する質問や提案がございましたら、Issueを作成してください。

## 🙏 謝辞

- **Facebook Prophet Team**: 優秀な時系列予測ライブラリの提供
- **Microsoft LightGBM Team**: 高性能な勾配ブースティングフレームワークの開発
- **オープンソースコミュニティ**: 素晴らしいPythonライブラリエコシステムの構築

---
