# CoLingo: Cooperation, Communication and Consensus Language Emergence

## TODO

- ハンドメイド言語
  - 整数と記号が1対1
  - 常に同じ1文字
- Metrics
  - Stability
  - (Async)
  - テストのためにハンドメイド言語を用意する
- 混合精度
- Task Scheduler（複数のTaskの実行開始判定，終了判定，順序を計画する.）
  - ランダムに選んで1個実行するScheduler
  - 順番に1回ずつ実行するScheduler
- Evaluator
  - 学習中に定期的に実行する？セーブしたモデルをロードして計算する？
  - Agent一人にdatasetとrole与えてoutput出力
  - Agent複数人にdatasetとrole与えてoutput出力
- Network
  - Complete network
  - Bipartite network
- Periodic model initializer
- Faction
- Image input
- channel noise
- num_workers = 2 (num_workersの数だけGPUが必要かもしれない)
- Task
  - Broadcast
  - Telephon
  - Conversion
  - Parrot
  - 

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