from sklearn.cluster import KMeans
from node_embedding import node_embedding_gcn, node_embedding_gat, node_embedding_graphsage


#GCN
def node_clustering_gcn(x, edge_index, num_clusters=7):
    model, embeddings = node_embedding_gcn(x, edge_index)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings.numpy())

    return model, embeddings, cluster_labels


#GAT
def node_clustering_gat(x, edge_index, num_clusters=7):
    model, embeddings = node_embedding_gat(x, edge_index)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings.numpy())

    return model, embeddings, cluster_labels


#GraphSAGE
def node_clustering_graphsage(x, edge_index, num_clusters=7):
    model, embeddings = node_embedding_graphsage(x, edge_index)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings.numpy())

    return model, embeddings, cluster_labels