#
#とりあえず動くところまで作った iris分類.
#
import argparse
import numpy
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.cuda as cuda

#irisを分類するモデル.
class IrisModel(chainer.Chain):
    def __init__(self):
        super(IrisModel, self).__init__(
            l1=L.Linear(4, 100),     #4つの素性を受け取り
            l2=L.Linear(100, 100),
            l3=L.Linear(100, 3),     #3つの分類のどれに近いかを返す.
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)

        return y

#分類した結果が文字列なので [0,1,2] という数字に変換する
def iris_name_to_index(s):
    if s == b'setosa':
        return 0   #正解データは0からスタートしないとダメ
    elif s == b'virginica':
        return 1
    elif s == b'versicolor':
        return 2
    else:
        print("Unknown Data:{}".format(s))
        raise()

#逆に、数字から、setosa/virginica/versicolor に変換する.
def iris_index_to_name(i):
    if i == 0:
        return 'setosa'
    elif i == 1:
        return 'virginica'
    elif i == 2:
        return 'versicolor'
    else:
        print("Unknown Label:{}".format(i))
        raise()


# 引数の処理
parser = argparse.ArgumentParser()
parser.add_argument('--mode',      type=str,   default='all') #train:学習だけ pred:分類だけ all:両方
parser.add_argument('--gpu',       type=int,   default=0)   #GPUを使うかどうか
parser.add_argument('--epoch',     type=int,   default=500) #何回学習を繰り返すか
parser.add_argument('--trainstart',type=int,   default=0)   #全データ150件のうち、何件から学習に使うか
parser.add_argument('--trainsize', type=int,   default=50)  #全データ150件のうち、何件を学習に使うか ディフォルト(150件中 0件から50件までを学習に使う)
parser.add_argument('--trainbatch',type=int,   default=50)  #学習するミニバッチに一度にかけるデータの個数
args = parser.parse_args()

#cudaを使うなら初期化する.
if args.gpu > 0:
    cuda.init()

#csv読込
#最初の4つが学習データ 最後の5番目が正解データ
csv = numpy.loadtxt("iris.csv",
    delimiter=",",                    #csvなので  , で、データは区切られている
    skiprows=1,                       #ヘッダーを読み飛ばす
    converters={4:iris_name_to_index} #4カラム目は分類がテキストでかかれているので 0から始まる数字ラベルに置き換える
    )

#学習データの定義
#学習データは float32の2次元配列
#例: [ [1,2,3,4],[5,6,7,8],[1,2,3,4],[5,6,7,8],[1,2,3,4] ].astype(numpy.float32)
data = csv[:,0:4].astype(numpy.float32)

#正解データの定義
#正解データは 0から始まる int32の1次元配列
#例: [ 0,1,2,1,0 ].astype(numpy.int32)
label= csv[:,4].astype(numpy.int32)

#学習
if args.mode in ['all','train']:
    #分類機の用意
    model = L.Classifier(IrisModel())
    if args.gpu > 0:
        model.to_gpu()

    #最適化アルゴリズムの設定
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    for epoch in range(args.epoch):
        print('epoch', epoch)

        #perm = numpy.random.permutation(args.trainsize)
        #サンプルとかを見ると perm[i:i+args.trainbatch] しているが、これを使う意味がよくわからんので外している.

        #学習を開始する.
        #学習に使うデータの範囲を args.trainbatchサイズに分けて学習器に投げ込む
        for i in range(args.trainstart, args.trainsize, args.trainbatch):
            x = data[i:i + args.trainbatch]
            y = label[i:i + args.trainbatch]
            if args.gpu > 0:
                x = cuda.to_gpu(x)
                y = cuda.to_gpu(y)

            x = chainer.Variable(x)
            y = chainer.Variable(y)

            optimizer.update(model, x, y)

    #学習結果を保存.
    chainer.serializers.save_npz("model.dat", model)

#分類
if args.mode in ['all','pred']:

    #学習結果を読み込む
    model = L.Classifier(IrisModel())
    chainer.serializers.load_npz("model.dat",model)
    if args.gpu > 0:
        model.to_gpu()
    else:
        model.to_cpu()

    #すべてのデータに対して、1件ずつ、分類を答えさせて、正しいか採点する.
    ok_count = 0
    for i in range(len(data)):
        x = data[i:i+1]        #このデータについて調べたい.
        if args.gpu > 0:
            x = cuda.to_gpu(x)

        x = chainer.Variable(x)

        y = model.predictor(x) #分類の結果を取得

        y = y.data             #chainer.Variableからデータの取出し
        if args.gpu > 0:
            y = cuda.to_cpu(y)

        pred = numpy.argmax(y)    #3つの分類のうち、どれの確率が一番高いのかを返す
        if label[i] == pred:
           ok_count = ok_count + 1
           #print("OK i:{} pred:{}({}) data:{}".format(i,iris_index_to_name(pred),iris_index_to_name(label[i]),data[i:i+1] ))
        else:
           print("NG i:{} pred:{}({}) data:{}".format(i,iris_index_to_name(pred),iris_index_to_name(label[i]),data[i:i+1] ))

    print("total:{} OK:{} NG:{} rate:{}".format(len(data),ok_count,len(data)-ok_count,ok_count/len(data)) )

