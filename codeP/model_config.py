# Tasks
from link_prediction import link_prediction_gcn, link_prediction_gat, link_prediction_graphsage
from node_classification import node_classification_gcn, node_classification_gat, node_classification_graphsage
from node_regression import node_regression_gcn, node_regression_gat, node_regression_graphsage
from edge_classification import edge_classification_gcn, edge_classification_gat, edge_classification_graphsage
from edge_regression import edge_regression_gcn, edge_regression_gat, edge_regression_graphsage
from node_clustering import node_clustering_gcn, node_clustering_gat, node_clustering_graphsage
from node_embedding import node_embedding_gcn, node_embedding_gat, node_embedding_graphsage
from graph_classification import graph_classification_gcn, graph_classification_gat, graph_classification_graphsage
from graph_regression import graph_regression_gcn, graph_regression_gat, graph_regression_graphsage
from graph_matching import graph_matching_siamese_gcn
from graph_reconstruction import graph_reconstruction_gae
from graph_generation import graph_generation_graphvae


# Models
available_models = {
    "gcn": {
        "name": "GCN",
        "module": {
            "link_prediction": link_prediction_gcn,
            "node_classification": node_classification_gcn,
            "node_regression": node_regression_gcn,
            "edge_classification": edge_classification_gcn,
            "edge_regression": edge_regression_gcn,
            "node_clustering": node_clustering_gcn,
            "node_embedding": node_embedding_gcn,
            "graph_classification": graph_classification_gcn,
            "graph_regression": graph_regression_gcn,
        }
    },
    "gat": {
        "name": "GAT",
        "module": {
            "link_prediction": link_prediction_gat,
            "node_classification": node_classification_gat,
            "node_regression": node_regression_gat,
            "edge_classification": edge_classification_gat,
            "edge_regression": edge_regression_gat,
            "node_clustering": node_clustering_gat,
            "node_embedding": node_embedding_gat,
            "graph_classification": graph_classification_gat,
            "graph_regression": graph_regression_gat,
        }
    },
    "graphsage": {
        "name": "GraphSAGE",
        "module": {
            "link_prediction": link_prediction_graphsage,
            "node_classification": node_classification_graphsage,
            "node_regression": node_regression_graphsage,
            "edge_classification": edge_classification_graphsage,
            "edge_regression": edge_regression_graphsage,
            "node_clustering": node_clustering_graphsage,
            "node_embedding": node_embedding_graphsage,
            "graph_classification": graph_classification_graphsage,
            "graph_regression": graph_regression_graphsage,
        }
    },
    "siamese_gcn": {
        "name": "Siamese GCN",
        "module": {
            "graph_matching": graph_matching_siamese_gcn,
        }
    },
    "gae": {
        "name": "GAE",
        "module": {
            "graph_reconstruction": graph_reconstruction_gae,
        }
    },
    "graphvae": {
        "name": "GraphVAE",
        "module": {
            "graph_generation": graph_generation_graphvae,
        }
    },
}