import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def find_optimal_eps(X, k=None):
    save_path = './dbscan_k_distance_plot.png'
    if k is None:
        k = min(5, len(X) - 1)  # 默认k值
    
    neighbors = NearestNeighbors(n_neighbors=k, metric='cosine')
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    
    distances = np.sort(distances[:, k-1], axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel('数据点排序')
    plt.ylabel(f'{k}最近邻距离')
    plt.title('k距离图 - 选择eps（寻找拐点）')
    plt.grid(True)
    
    # 保存图形到文件
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"k距离图已保存到: {save_path}")
    
    return distances

# 读取数据
df = pd.read_csv('/root/sc_code/loghub/HDFS/HDFS_2k.out.csv')

# 使用TF-IDF进行文本向量化
feature_column = 'Content'
text_data = df[feature_column].astype(str)
text_data = text_data.str.replace('/', ' ')
text_data = text_data.str.replace('(', ' ')
text_data = text_data.str.replace(')', ' ')
text_data = text_data.str.replace('.', '')
text_data = text_data.str.replace('-', '')
text_data = text_data.str.replace(':', ' ')

# 创建TF-IDF向量化器
vectorizer = TfidfVectorizer(
    min_df=4,           # 忽略出现次数少于2的词
    max_df=0.8,         # 忽略出现在80%以上文档中的词
    stop_words='english', # 移除英文停用词
    ngram_range=(1, 2)  # 使用单字和双字词组
)

X_tfidf = vectorizer.fit_transform(text_data)

print(f"文本向量化完成，特征数量: {X_tfidf.shape[1]}")
print(f"文档数量: {X_tfidf.shape[0]}")

# # 使用k距离图寻找最佳eps参数
# print("生成k距离图...")
# distances = find_optimal_eps(X_tfidf, k=5)

# 使用DBSCAN聚类
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
clusters = dbscan.fit_predict(X_tfidf)
df['cluster'] = clusters


df.to_csv('/root/sc_code/bgl/loghub_train_data/split_data/dbscan_HDFS.csv', index=False)

n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = sum(clusters == -1)

print(f"聚类数量: {n_clusters}")
print(f"噪声点数量: {n_noise}")
print(f"每个簇的数据点分布:")
cluster_counts = pd.Series(clusters).value_counts().sort_index()
print(cluster_counts)