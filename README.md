# convolutional-nn-cifar10
CIFAR10データセットを使用した畳込みニューラルネットワークによる特徴表現の学習の可視化です。

## 実行方法
`python train.py`  
はじめて実行した時はデータセットをダウンロードするため、しばらく時間がかかります。

## 実行結果
畳込み層の重みを可視化したもの  
![](https://github.com/s059ff/convolutional-nn-cifar10/blob/master/sample/kernel.png)  
学習曲線(終端のテスト誤差は約1.67, テスト正解率は約77%)  
![](https://github.com/s059ff/convolutional-nn-cifar10/blob/master/sample/loss.png)  
