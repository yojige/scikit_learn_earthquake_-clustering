import os
import glob
import shutil
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import geopandas as gpd


def diff(df, s1, s2):
    d = []
    t = []
    for i in range(1, len(df[s2])-1):
        if df[s2][i] > df[s2][i-1] and df[s2][i] > df[s2][i+1]:
            d.append(df[s2][i])
            t.append(df[s1][i])
    df = pd.DataFrame(list(zip(t, d)), columns = ['period','acc'])
    return(df)


if __name__ == '__main__':

    # 予め計算しておいた各記録波の応答スペクトルを読込む
    files = glob.glob('*.spec.csv')
    feature = np.array([])
    for file in files:
        df = pd.read_csv(file, index_col=None)
        # feature = np.append(feature, df['acc']/df['acc'][0])
        feature = np.append(feature, df['acc'])

    # 1次元numpy配列を2次元に変換する（応答スペクトルの計算ポイント数300）
    feature = feature.reshape(-1, 300)  

    # 5種類のグループにクラスタリングする
    model = KMeans(n_clusters=5, init='random').fit(feature)

    # 学習結果のラベルを取得する
    labels = model.labels_
    
    # 応答スペクトルをクラスタ毎に色分けしてプロットする
    colors = ["red", "green", "blue", "magenta", "cyan"]
    plt.grid()
    plt.xscale('log')
    plt.xlabel('Period(sec)')
    plt.ylabel('Response Acceleration Spectrum')
    for label, v, file in zip(labels, feature, files):
        plt.plot(df['period(sec)'], v, color=colors[label])
        #png_file = file.replace('spec.csv', 'png')
        #os.makedirs(f"./group/{label}", exist_ok=True)
        #shutil.copyfile(png_file, f'./group/{label}/{png_file}')
    #plt.show()
    plt.savefig('spec1.png')


    # 各観測点の情報を読込む
    geo = pd.read_csv('geo.csv', index_col='name')

    # クラスタ毎に色分けして地図上にプロットする
    jpnShp = gpd.read_file('./gm-jpn-all_u_2_2/coastl_jpn.shp')
    ax = jpnShp.plot(figsize=(10, 20), color='black', linewidth=.5)
    for label, file in zip(labels, files):
        plt.scatter(geo.at[file[:6],'Long'], geo.at[file[:6],'Lat'], marker='o', s=10, color=colors[label], alpha=1.0, linewidths = 1)
    #plt.show()
    plt.savefig('map1.png')
