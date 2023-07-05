from typing import Iterable

import src.logger.operator as op
from src.logger import ConsoleLogger

logger = op.pipe(
    op.filter(pred=lambda x: len(x) > 1),
    op.select(func=lambda x: {"length": len(x)}),
    ConsoleLogger(),
)

logger.log({"a": 1, "b": 2, "x": 9})
logger.on_update(0)
logger.log({"c": 1, "d": 2})
logger.on_update(0)
logger.log({"e": 1})
logger.on_update(0)

"""
1. ゲームのメトリクス:distを計算する．
2. 全部ロギングする
3. ここで分岐
    a. メトリクスから値を１つだけ取り出す．
    b. 値がしきい値を超えたらタイミングで一度だけ通す．
    c. その時の呼び出し回数を名前をつけてロギングする
4. ここで分岐
    a. メトリクスから値を１つだけ取り出す．
    b. 値が一定期間変化しなかった場合に通す．
    c. 指定された無名関数を実行する．
5. ここで分岐
    a. n回に１回だけ通す．
    b. 時間のかかるメトリクスを計算する．
6. ここで分岐
    a. m回に１回だけ通す．
    b. 時間のかかるメトリクスを計算する．

logger = op.publish(
    ConsoleLogger(),
    WandBLogger(),
    FileLogger(),
    TensorBoardLogger(),
    
)
reach_pipe = op.pipe(
    op.once(pred=lambda x: x["acc"]>0.99),
    op.select(lambda x: x["acc"]),
    logger,
)
stable_pipe = op.pipe(
    op.stable(target=lambda x: x["acc"], n=10),
    op.select(lambda x: x["acc"]),
    logger,
)
slow_pipe = op.pipe(
    op.interval(period=1000, offset=0),
    op.select(slow_metric),
    logger,
)
slow_pipe2 = op.pipe(
    op.interval(period=100, offset=0),
    op.select(slow_metric2),
    logger,
)

logger = op.publish(
    logger,
    reach_pipe,
    stable_pipe,
    slow_pipe,
    slow_pipe2,
)

evaluatorがbegin，update，endそれぞれのLoggerを受け取ればおｋ．
"""
