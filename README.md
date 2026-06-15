# 経路選択モデル演習 — Recursive Logit Model

松山市の歩行者 GPS トラッキングデータ（マップマッチング済み）を使い，
**Recursive Logit (RL) モデル**（Fosgerau et al., 2013）を推定する演習リポジトリです．

## 演習の概要

| パート | 内容 |
|--------|------|
| Part 1 | ネットワーク・経路データの可視化（folium） |
| Part 2 | リンク説明変数の基礎集計（相関行列・分布） |
| Part 3 | デフォルト設定での RL 推定・サマリー表示 |
| Part 4 | SAM3 航空写真セグメンテーション → 説明変数追加 → 再推定 |

## セットアップ

### 1. データの配置

別途配布された `data/` フォルダをリポジトリ直下に配置してください．

```
traffic-class-rcm-2026/
└── data/
    ├── network/
    │   ├── link.csv
    │   └── node.csv
    ├── dlink_features.parquet
    ├── routes.parquet
    ├── aerial_photo/
    │   ├── matsuyama_2010_z18.png
    │   └── matsuyama_2010_z18_geo.json
    └── land_use/
        └── matsuyama/
```

### 2. パッケージのインストール

**Google Colab の場合**（ノートブック先頭のセルを実行）:

```python
!pip install uv -q
!uv pip install -e . --system -q
```

**ローカル環境の場合**:

```bash
pip install uv
uv pip install -e .
```

### 3. ノートブックを開く

`exercise.ipynb` を Google Colab または Jupyter Notebook で開いてセルを順番に実行してください．

## 設定ファイル

`config.yaml` でデータパス・モデルパラメータ・SAM3 キーワードを設定します．

```yaml
model:
  max_iter: 500      # L-BFGS-B 最大反復数
  conv_eps: 1.0e-6   # 収束判定閾値

sam3:
  words: []          # ← ここにキーワードを追加（Part 4）
  # 例: ["road", "sidewalk", "tree", "building"]
```

## Part 4: SAM3 の使い方

1. `config.yaml` の `sam3.words` にセグメントしたいキーワードをリストで追加する
2. ノートブックの Part 4 先頭セルで SAM3 をインストールする:
   ```python
   !uv pip install ".[sam3]" --system -q
   ```
3. `config.yaml` を再読み込み後，`get_sam3_features()` を実行する

キーワード 1 件につきリンクごとのセグメント面積割合が 1 列追加され，
RL モデルの追加説明変数として使用されます．
セグメンテーション結果は `cache/sam3/` にキャッシュされるため，
2 回目以降は推論が実行されません．

## データの説明

| ファイル | 内容 |
|----------|------|
| `data/network/link.csv` | 道路リンク（リンクID・起終点ノード・歩行者可否・車線数・歩道幅員など） |
| `data/network/node.csv` | 道路ノード（ノードID・経緯度） |
| `data/dlink_features.parquet` | リンク属性（土地利用 10 カテゴリ・車線数・歩道幅・速度） |
| `data/routes.parquet` | マップマッチング済み歩行者経路（155 経路） |
| `data/aerial_photo/` | 松山市 2010 年航空写真（zoom 18）と地理参照情報 |
| `data/land_use/` | 国土数値情報 土地利用細分メッシュデータ（平成 20 年） |

## モデルの概要

Recursive Logit モデルは，Bellman 方程式を対数空間で解くことで，
経路集合を明示的に列挙せずに経路選択確率を定式化します．

$$V(a) = \log \sum_{j \in \mathcal{A}(a)} \exp\!\bigl(\beta^\top x_j + \gamma \, V(j)\bigr)$$

- $\beta$: リンク効用パラメータ（推定対象）
- $\gamma \in (0, 1)$: 将来の効用に対する割引率（推定対象）
- $x_j$: リンク $j$ の説明変数ベクトル
- $V(a)$: リンク $a$ を出発したときの期待最大効用

パラメータは条件付き対数尤度を L-BFGS-B で最大化することで推定します．

## 参考文献

Fosgerau, M., Frejinger, E., & Karlstrom, A. (2013).
A link based network route choice model with unrestricted choice set.
*Transportation Research Part B*, 56, 70–80.
