# CoLingo (Cooperative Language Emergence Library)

## TODO
- Network
  - Empty network
  - Complete network
  - Bipartite network
- Metrics
  - Topographic similarity
  - Language syncronization
  - Stability
  - (Async)
- Periodic model initializer
- Faction
- Image input

## 著者
- gizmrkv

## 目的
本ライブラリは、複数のエージェントを含む環境での言語創発実験を支援し、言語創発研究の新たな領域を探求することを目的としています。

## 背景
既存の言語創発研究では、通常、2人のAgent間のシグナリングゲームに基づく実験が行われています。Lewisのシグナリングゲームをもとにした2人の間の言語創発実験のためのオープンソースライブラリ（EGG, 2019）も存在します。しかし、現実世界では3人以上のAgentが協力する状況が一般的であり、このような状況を模倣する言語創発実験の需要が高まっています。

CoLingoは、3人以上のAgentが参加する環境での言語創発実験を容易に行えるよう設計されており、研究者がより現実に近い状況で言語創発を研究できる利点を持っています。また、既存の言語創発研究やライブラリと比較して、より柔軟で拡張性に優れた設計が特徴です。

## 概要
本ライブラリを用いて行われる実験では，まず

1. 環境に存在するAgent，実験に用いるデータセット，各イテレーションで行うTaskを生成する．
2. 生成されたTaskを繰り返し実行する．

## 要素

### Agent
Agentはモデルを持ち，これを使ってTaskからの入力を処理して出力を返します．Taskから入力を受け取る際，そのTaskでのAgentの役割も受け取り，これに従ってモデルの振る舞いを切り替えます．例えばLewisのシグナリングゲームでは，Senderとして振る舞うAgentには"sender"，Receiverとして振る舞うReceiverには"receiver"という役割を与えます．これら以外にも任意の役割とTaskを定義することができます．

### Task
Taskは実験全体の反復で繰り返し実行される処理です．Agentの学習，メトリクスの計算，モデルの保存など，任意の処理をTaskとして実行できます．例えばLewisのシグナリングゲームでは，Agentの中からランダムにSenderとReceiverを選び，学習を行います．

### Network
NetworkはAgent間の通信路を定義します．これはグラフ構造であり，ノードはAgentを表し，エッジはAgent間の通信路を表します．TaskによってはNetworkを必要とします．


