# CoLingo: Cooperation, Communication and Consensus Language Emergence

## TODO
- 部品を差し替えられるAgent
  - encoders, shared model, decoders
  - MLPEncoder, MLPDecoder, MLPSharedNet (Mixed)
  - RNNEncoder, RNNDecoder
  - TransformerEncoder, TransformerDecoder
  - Seq2SeqEncoder, Seq2SeqDecoder
  - VAEncoder, VADecoder

- Agentのcommandに対する挙動は外部から差し替えられるように
  - command: strをキー，挙動関数を値とする辞書を渡す

- Evaluator -> Logger の間に WandBProcessor を挟む
- Agent数の2乗に比例して計算量が増えるEvaluatorの対処
  - verbose: int = 0 | 1 | 2 で，0は何もしない，1は1エージェントのみ，2は全エージェント
- 指定されたメトリクスを監視し，閾値を超えたり停滞したら発火するイベントをどう作る？
  - Logger を継承した Metric Observerが発火する?
  - 初めて閾値を超えたとき
  - 移動分散がしきい値を下回ったとき

- 自分の発言は自分にも聞こえる
- Task
  - Speech
  - Telephon
  - Conversion
  - OpenAI Gym
- documentation
  - torchtyping, typeguard, pytest
  - configファイルを1回通してチェックする
- Actor-Critic
- Metrics
  - Stability
  - (Async)
  - テストのためにハンドメイド言語を用意する
- 混合精度
- Faction
- Image input
- channel noise
  - メッセージの記号をランダムに書き換える
  - 特定の記号ペアは入れ替わりやすい
  - メッセージを前か後にシフトする（話し始めを聞きそびれる的な）
## メモ

Numbaを使ったscipyのspearmanrの高速化は3倍ほど遅くなる結果に終わった．
これはspearmanrの内部ではfor文を使わずにnumpyの関数のみを使っていたことが理由だろう．
一方，scipyのpdistはfor文を使っているため，Numbaで1.5倍ほど高速化できた．


## 考察

無言期
はじめは喋るとペナルティかかるから何もしゃべらない

語彙爆発期
そこからなにか喋ると報酬が与えらることに気づく．Receiverは即時に対応できる．
SenderがConceptに対してMessageを作るとき，このMessageがほかのConceptに使うかは気にしない．喋って報酬が与えられなかったとき，単に別のメッセージを作ろうとするが，使われていないMessageを作らないと正答率は上がらない．

停滞期
学習が進んだあと，0.02%ぐらいから進まない頃，Senderが異なるConceptを同じMessageに割り当てちゃう．ReceiverがカバーできないからSenderが強化学習でなんとかしないといけないけど時間がかかる．

### 停滞期の解消考察

- 人真似
    言語習得者のMessageはある程度バラけているため，これを真似すれば即時に停滞期までスキップできる．Messageのダブリング解決に時間を多くかけられる．
    色んな人のMessageを真似することでいいとこ取りできる？
- 語彙辞書を参照する．レキシコン
    新しく言葉を作るときに，既存の言葉を参照する．これにより，ダブリングを防げる．


## 目的

CoLingoは、複数のエージェントが関与する環境での言語創発実験を促進するためのオープンソースライブラリです。このライブラリの主な目的は、言語創発研究の新しい領域を探求するのを支援することです。

## 背景

既存の言語創発研究のほとんどは、2人のエージェント間のシグナリングゲームに基づいていますが、実際の世界のシナリオでは、2人以上のエージェント間での協力がしばしば発生します。したがって、そのような状況を模倣できる言語創発実験への需要が高まっています。CoLingoは、三人以上のエージェントが関与する環境で、より現実的な言語創発研究の状況を提供することを目指して設計されています。既存の言語創発研究やライブラリに比べて、CoLingoは柔軟性と拡張性のある設計を提供します。

## 概要

このライブラリを使用して行われる実験では、まず、環境内のエージェント、実験で使用するデータセット、各イテレーションで実行されるタスクが生成されます。その後、これらのタスクが繰り返し実行されます。

## 要素

- **エージェント**: エージェントは、タスクからの入力を処理し、出力を返すモデルを持っています。彼らがタスクから入力を受け取るとき、そのタスクでの彼らの役割も受け取り、それに応じてモデルの振る舞いを切り替えます。

- **タスク**: タスクは、実験全体を通じて繰り返し実行されるプロセスです。これは、エージェントの学習、メトリクスの計算、モデルの保存など、何でも可能です。

- **ネットワーク**: ネットワークはエージェント間の通信チャネルを定義します。これはグラフ構造で、ノードはエージェントを、エッジはエージェント間の通信パスを表します。一部のタスクではネットワークが必要になる場合があります。



## Installing

```
git clone https://github.com/gizmrkv/CoLingo.git
cd CoLingo
poetry lock
poetry install
```