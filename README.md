# Abstract
A program for generating teacher data and machine learning to identify t-designs, and a program for estimating the design of a measure by calculating Frame Potential.
It is written in C++ and Python, and the C++ compiler is assumed to be MSVC++ for Windows and GNU for Linux systems. On Python, it is assumed to be Anaconda3 for Windows, and vanilla Python for Linux.
It has confirmed that it works on Windows-anaconda3 and Linux-pip3 in my environment. pip3 on Linux is python3.9 on pyenv.

The programs for data generation and FP calculations rely heavily on [Qulacs](https://github.com/qulacs/qulacs).

# How to run
## Setup
1. Run "setup_linux.sh" or "setup_win.bat" accroding to your environment
2. Prepare your environment if needed("conda_requirements.txt" or "pip_requirements.txt")
3. Install qulacs folloing [this](https://github.com/qulacs/qulacs#use-qualcs-as-c-library)(qulacs is already cloned if you run the setup file)
## Run data generation program
1. Change the directory to "t-design_MPbased_cpp/data_gen/main"
2. Compile "main.cpp" and make exe file
3. Rename the exe file "main" in Linux, "msvc_project.exe" in Windows
4. Run "auto_create.py"
5. Cut&Paste the created folder into "ml/datasets/"
## Run machine learning program
1. Change the directory to "t-design_MPbased_cpp/ml"
2. Choose python code which you want to run
## Run Frame Potential program
1. Change the directory to "t-design_MPbased_cpp/fp"
2. Run python code or complie C++ code and run it


**More detail explanation is [below](https://github.com/mf-22/t-design_MPbased_cpp/edit/master/README.md#%E3%83%97%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%A0%E3%81%AE%E5%AE%9F%E8%A1%8C%E6%96%B9%E6%B3%95) of this page(written in Japanese, so please translate).**

# 概要
t-designの識別のための教師データ生成と機械学習のプログラム、そして特定の量子ビット数・回路の深さのときのLocal Random Circuit, Random Diagonal Circuitが
どれぐらいのdesignになっているか推定するためのFramePotentialの計算プログラムです。  
C++とPythonで書かれており、C++のコンパイラはWindowsならMSVC++、Linux系ならGNUを想定しています。
PythonのほうはWindowsならAnaconda Python、Linux系ではvanilaなpythonを想定しています。

手元のWindows-anaconda3、Linux-pip3の環境で動くことを確認しました。Linuxのpip3はpyenv上のpython3.9です。

データ生成、FPの計算のプログラムは[Qulacs](https://github.com/qulacs/qulacs)にかなり依存しています。

## Requirement
Windows-Anacondaの場合はconda_requirements.txtを、Linux-pip3の場合はpip_requirements.txtを用いると環境構築ができます。  
基本的に[Qulacsのrequirement](https://github.com/qulacs/qulacs#requirement)を満たしていれば大丈夫なはずです。  
細かい部分は以下：  
- ディープラーニングのプログラム（ml/deep.py）
  - 作成したNeural Networkのモデルの図をpdfで保存するときに[tensorflow.keras.utils.plot_model](https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model)を使っています
  （[ココ](https://github.com/mf-22/t-design_MPbased_cpp/blob/master/ml/deep.py#L106)）。
  このとき環境に応じてgraphvizが必要になるかもしれません。ダウンロードして設定するか、面倒だったらコメントアウトしてください。
  一応テキストファイルでモデルを保存するようにしてあります。  
  Windows-Anacondaの場合は
  ```
  conda install pydot graphviz
  ```
  を実行すれば大丈夫で、Linux-pip3の場合は
  ```
  pip3 install pydot
  apt install graphviz
  ```
  等でインストールできると思います。各requirements.txtにLinux-pip3のgraphviz以外は含まれています。
- FramePotentialの計算のプログラム（fp/calc_fp.cpp）
  - cloneしたままのコードだとC++17が必要です。ただ、計算結果を保存するためのディレクトリ作成でfilesystemを使っているだけなので、標準出力でOKなら問題ないです。

## Setup
Windowsの場合はトップにある`setup_win.bat`を、Linuxの場合は`setup_linux.sh`を実行すると必要なディレクトリの作成とQulacsのクローン・ビルドをします。  
ただ、データ生成のプログラムをコンパイルするときに最適化オプション(-O3等)を使うようMakefileに書いてあり、Qulacsでも同等の最適化オプションをつける必要がある場合があるかもしれません。
そのときは[QualcsのCMakeLists.txt](https://github.com/qulacs/qulacs/blob/master/CMakeLists.txt)を編集しもう一度Qulacsをビルドしてください。Qulacsのビルド時にOMPをつけない
ようなときも同様に編集してください。  
その後はOSに応じて次の操作を行ってください：  

### Windows環境の場合  
- データ生成のプログラム（data_gen/auto_crate.py）の設定
  - Visual Studio等を使って`main.cpp`をコンパイルし実行ファイル(\*\*\*.exe)を作成してください。その後、できた実行ファイルを`data_gen/main/`にコピーし、
  ファイル名を`msvc_project.exe`に変更するか`data_gen/main/auto_create.py`の[15行目](https://github.com/mf-22/t-design_MPbased_cpp/blob/master/data_gen/main/auto_create.py#L15)を
  ```
  exe_cpp = "***.exe" (実行ファイル名)
  ```
  に変更してください。
  この[コミット](https://github.com/mf-22/t-design_MPbased_cpp/commit/c4c545a89c46cc6ea024cab2ec4723398c1aba02)からCliffordの実行が、subprocessによるプログラムの実行から
  関数呼び出しになったので修正は不要です。

### Linux環境の場合
基本的にMakeFileのmakeコマンドを使ってビルドすると実行ファイル`main`が生成されます。そのmainファイルを実行すればOKです。もし`data_gen/main/main.cpp`の実行ファイル名が
`main`以外の場合は`data_gen/main/auto_create.py`の[18行目](https://github.com/mf-22/t-design_MPbased_cpp/blob/master/data_gen/main/auto_create.py#L18)を
```
exe_cpp = "./(実行ファイル名)"
```
に変更してください。

## プログラムの実行方法
### 教師データの生成（data_gen/）
データ生成プログラムは`auto_create.py`を実行する方法と、`random_clif.py`/`main.cpp`をそれぞれで実行するの２通りがあります。これらの違いは生成されたデータを特定のディレクトリ
に格納するかの違いです。データ生成をした後は機械学習を行いますが、機械学習のプログラムでは教師データを読み込むときに以下のディレクトリ構造を想定しています：
```
datasets --- dataset1 --- info.txt 
                       |
                       |- train --- haar_***.npy
                       |          ∟ clif_***.npy
                       |
                       |- valid --- haar_***.npy
                       |          ∟ clif_***.npy
                       |
         　             ∟ test  --- test1 --- haar_***.npy
                                 |          ∟ clif_***.npy
                                 |
                                 |- test2 --- haar_***.npy          
                                 |          ∟ clif_***.npy
                                 |
                   　             ∟ test3 --- lrc_***.npy
```
上記の例はHaar測度とClifford系のデータを使ってtraining/validiationを行い、テストデータはtest1,2がHaar測度とClifford系のミックス、test3がLocal Random Circuitのみでテストを行う
データセットです。`info.txt`はデータ生成時のパラメータが保存されたテキストファイルで機械学習時にパラメータを読みに行く必要がある場合はこのファイルを読みに行きます。（なお、
本来はそれぞれの一番下のディレクトリに各教師データ作成時のパラメータが保存されたテキストファイル`info_+++_***.npy`も作成されますが省いています）  
この`dataset1`以下のディレクトリ構造を`auto_create.py`を実行すると自動で作成してくれます。作成場所は`data_gen/main/`の中なので`ml/datasets/`に移動してください。  
`random_clif.py`/`main.cpp`をそれぞれで実行した場合は`data_gen/result/`の中に教師データ`haar_***.npy`とその時のパラメータが保存されたテキストファイル`info_haar_***.txt`が
作成されるので、上記のデータ構造に沿って格納してもらえれば問題ありません。  
#### auto_create.pyの実行例
`python auto_create.py`を実行すると以下のような形で入力を求められます。パラメータの入力がすべて終わるとデータの作成を開始します。
```
$python auto_create.py #引数に"n_proc=x"とするとCliffordのシミュレーションをx並列で行うように指定できる。

*** start creating dataset ***
input folder name : dataset1 #データセットの名前の入力。dataset{番号}を推奨。
input the number of test datasets : 7 #テストデータセットを用意する数。1つのデータセットに最大2種類のデータを含めることができる。

Do you custom? (y/n) #2クラス分類を行うときに、データの種類(Haar,Clif,LRC,RDC)を2種類で固定してしまうかどうか
y #例えばtrainとvalid、testがHaarとClifだけで良いならn。訓練はHaarとClifでテストにLRCを使うとかならy。

input index == 1:haar  2:clifford  3:LRC  4:RDC (0:don`t create) #データの種類の入力
train1 :  1   #訓練データの1種類目
train2 :  2   #    〃　　の2種類目
valid1 :  1   #検証データの1種類目
valid2 :  2   #    〃　　の2種類目
test1-1 :  1  #テストデータセット1番の1種類目
test1-2 :  2  #       〃            の2種類目
test2-1 :  1
test2-2 :  2
test3-1 :  1
test3-2 :  2
test4-1 :  3  #テストデータセット4番の1種類目(LRC)
test4-2 :  0  #テストデータセット4番の2種類目(なし) => テストデータセット4は中身がLRCのみ
test5-1 :  3
test5-2 :  0
test6-1 :  3
test6-2 :  0
test7-1 :  4
test7-2 :  0

input parameters of circuit simulation
train |S| : 1000      #訓練データのデータサイズ(特徴量ベクトルの本数)
valid |S| : 100       #検証データの　　〃
each test |S| : 100   #各テストデータの  〃
Nu : 100              #1つの教師データに含まれるユニタリの数
Ns : 100              #1つのユニタリの測定回数
Nq : 3                #量子ビット数
Do you change the depth according to the data? (y/n) #量子回路の深さをすべて統一するかどうか
y #nなら深さがある場合にすべて統一する。yならそれぞれで変更する
depth(clifford - train) :  0 #クリフォードの回路の深さ。0のときはglobal random cliffordになる
depth(clifford - valid) :  0
depth(clifford - test1) :  0
depth(clifford - test2) :  3 #クリフォードで回路の深さを入力するとlocal random cliffordか、CNOTと1qubitクリフォードのものか選択する
  CNOT_1qC(clifford - test2) (1:=Yes, 0:=No):  0 #0を入力するとlocal random clifford
depth(clifford - test3) :  3
  CNOT_1qC(clifford - test3) (1:=Yes, 0:=No):  1 #1を入力するとCNOTと1qubitクリフォード
depth(LRC - test4) :  3 #Local Random Circuitの回路の深さ
depth(LRC - test5) :  3
depth(LRC - test6) :  3
depth(RDC - test7) :  3 #Random Diagonal Circuitの回路の深さ。ランダム対角ユニタリとHadmardがかかるまでを深さ1と定義している
Do you change the noise config according to the data? (y/n) #Local Random Circuitにノイズを加えるかどうか
y #yの場合はノイズを入れる。nの場合はノイズなしで統一
noise operator(0:=nothing, 1:=Depolarizing, 2:=Measurement)(LRC - test4) :  0 #0の場合はノイズなし
noise operator(0:=nothing, 1:=Depolarizing, 2:=Measurement)(LRC - test5) :  1 #1のときはdepolarrizing noiseをLRCの各層に挿入する
  noise probability(LRC - test5) :  0.1 #ノイズの割合
noise operator(0:=nothing, 1:=Depolarizing, 2:=Measurement)(LRC - test6) :  2 #2のときは測定をLRCの各層に挿入する。ただし最後の2層には入れない
  noise probability(LRC - test6) :  0.1

You will create below data #パラメータの入力が終了し作成するデータの情報を列挙されている
['haar - train' 'clifford - train' 'haar - valid' 'clifford - valid'
 'haar - test1' 'clifford - test1' 'haar - test2' 'clifford - test2'
 'haar - test3' 'clifford - test3' 'LRC - test4' 'LRC - test5'
 'LRC - test6' 'RDC - test7']

If all parameters are OK, press Enterkey. #問題がなければエンターキーを押すとデータの作成が開始される

start creating data...

haar - train
.
.
.

Elapsed Time(all) : 56.28598989999999[s] #すべてのデータを作成するのにかかった時間

  ***    All finished!!!    ***
```

### 機械学習の実行（ml/）
機械学習の実行は`ml/`の中のプログラムで実行したいアルゴリズムが書かれたファイルを実行してください。実行すると最初にデータセット名（上記の例では`dataset1`）の入力が求められ、
その次に計算するビット相関の数`k`、モーメントの次数`k'`を入力するように求められます。その後は自動的に学習・テストが行われ、その結果は
`ml/datasets/{データセット名}/{選択した機械学習アルゴリズム名}/{現在時刻}/`の中に保存されます。

### FramePotentailの計算（fp/）
`fp/calc_fp.cpp`をコンパイルし実行すればOKです。cloneしたままのコードであれば、`fp/result`の中にそれぞれのパラメータ(LRC/RDC, qubit数, 回路の深さ, t-designのtの値)のときの
計算結果が保存されます。
