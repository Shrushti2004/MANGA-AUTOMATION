# NameGen

## 吹き出しDBの構築

### データのダウンロード

NameGenはManga109とManga109Dialogを用いて吹き出し配置の参照とする吹き出しDBを構築する.
これらを事前にダウンロードしておく.

#### URLs
* [Manga109](http://www.manga109.org/ja/download.html)
    * Manga109のダウンロードは事前にフォームを送り許諾を受ける必要がある
* [Manga109Dialog](https://github.com/manga109/public-annotations)

### 手順

ダウンロードしたデータセットを以下のように`dataset`ディレクトリに配置する.

```
- dataset/
  |- Manga109/
     |- annotations/
     |- images/
     |- books.txt
     |- readme.txt
  |- Manga109Dialog/
     |- AisazuNihairarenai.xml
     |- ...
```

その後, `util/curate_dataset.py`を実行すると, `curated_dataset`に吹き出しDBに必要な情報が保存される. 

```
python util/curate_dataset.py
```

ただし, この状態では各作品のアノテーションファイルと画像しか存在しない. 全作品をまとめた一つのアノテーションファイルを以下のコマンドで作成する. 

```
python src/dataprepare.py
```

結果は`./curated_datset/database.json`として保存される.

## StableDiffusionの準備

NameGenでは画像生成モデルとしてStable Diffusionを使用する. まず, このリポジトリとは別のディレクトリに[stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)を保存し, 起動できるようにする. stable-diffusion-webuiの環境構築は, 公式githubリポジトリのREADMEを参照して欲しい.

stable-diffusion-webuiの`webui-user.sh`に以下を追記する.

```
export COMMANDLINE_ARGS="--api --xformers"
```

これにより, NameGen側からAPIとしてStable Diffusionを呼び出せるようになる. stable-diffusion-webuiを起動したのち, URLの後ろに`/docs`をつけてAPIのドキュメントが開くならば正しく起動できている.

アニメ風の画像生成では, [T-anime-v4](https://civitai.com/models/69552/t-anime-v4-pruned)と[sketch anime pose](https://civitai.com/models/106609/sketch-anime-pose)のLoRAを用いた.
これらをダウンロードし, stable-diffusion-webuiの`models/Stable-diffusion`と`models/Lora`にそれぞれ保存する.


## ネームの生成

事前に漫画原稿を`.txt`ファイルに保存しておく. stable-diffusion-webuiを起動しておく.

以下のコマンドを実行する.

```
python src/pipeline.py --script_path path/to/script.txt --output_path path/to/output_dir
```

実行コマンドの例は`scripts/run_pipeline.sh`に保存した.


### 追記

#### Setup.pyの追加

setup.pyを追加し以下を記入
```
from setuptools import setup, find_packages

setup(
    name='bubblealloc',
    version='0.1',
    packages=find_packages(),
    install_requires=[],
    author='Kazuki Kitano',
    description='Project aiming for auto allocation of speech bubbles',
)
```

#### torchのダウンロード

```
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
```


なお, 実行時に`--resume_latest`を指定すると, 最新のディレクトリの途中から実行を開始できる. 画像生成は終わっているが, 吹き出しの配置ができていないなどで有効.
=======
# MANGA-AUTOMATION
>>>>>>> 6466af625c5f8b0ea34c54698b2486edd27607e3
