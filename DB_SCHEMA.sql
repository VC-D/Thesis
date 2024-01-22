BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "df_density" (
	"GRAPH"	TEXT,
	"DENSITY"	INTEGER
);
CREATE TABLE IF NOT EXISTS "df_average_degree" (
	"graph"	TEXT,
	"average_degree"	INTEGER
);
CREATE TABLE IF NOT EXISTS "df_diameter_approximation" (
	"graph"	TEXT,
	"diameter_approximation"	INTEGER
);
CREATE TABLE IF NOT EXISTS "df_average_clustering_coefficient" (
	"graph"	TEXT,
	"average_clustering_coefficient"	INTEGER
);
CREATE TABLE IF NOT EXISTS "df_average_degree_centrality" (
	"graph"	TEXT,
	"average_degree_centrality"	INTEGER
);
CREATE TABLE IF NOT EXISTS "df_average_closeness_centrality" (
	"graph"	TEXT,
	"average_closeness_centrality"	INTEGER
);
CREATE TABLE IF NOT EXISTS "df_average_betweenness_centrality" (
	"graph"	TEXT,
	"average_betweenness_centrality"	INTEGER
);
CREATE TABLE IF NOT EXISTS "df_average_eigenvector_centrality" (
	"graph"	TEXT,
	"average_eigenvector_centrality"	INTEGER
);
CREATE TABLE IF NOT EXISTS "df_metric_means_for_GRAPH_d" (
	"GRAPH_d"	TEXT,
	"means"	REAL
);
CREATE TABLE IF NOT EXISTS "df_transitivity" (
	"graph"	TEXT,
	"transitivity"	INTEGER
);
CREATE TABLE IF NOT EXISTS "df_average_path_length" (
	"graph"	TEXT,
	"number_of_connected_nodes"	INTEGER,
	"average_path_length"	INTEGER
);
CREATE TABLE IF NOT EXISTS "df_betweenness_centrality_GRAPH_a" (
	"idf1"	INTEGER,
	"node"	INTEGER,
	"betweenness_centrality"	REAL,
	FOREIGN KEY("node") REFERENCES "NODES_GRAPH_a"("id") ON UPDATE CASCADE,
	PRIMARY KEY("idf1")
);
CREATE TABLE IF NOT EXISTS "NODES_GRAPH_c" (
	"idnodesc"	INTEGER,
	"customers_that_bought_this_product_id"	INTEGER,
	"also_bought_this_product_id_NUMBER_OF_NODES"	INTEGER,
	PRIMARY KEY("idnodesc")
);
CREATE TABLE IF NOT EXISTS "NODES_GRAPH_d" (
	"idnodesd"	INTEGER,
	"unq_editor_user_id"	INTEGER,
	"unq_edited_talk_page_user_id_NUMBER_OF_NODES"	INTEGER,
	PRIMARY KEY("idnodesd")
);
CREATE TABLE IF NOT EXISTS "Amazon_Product_Co_Purchasing_Network_GRAPH_c" (
	"idGRAPH_c"	INTEGER,
	"from_node_id"	INTEGER,
	"to_node_id"	INTEGER,
	PRIMARY KEY("idGRAPH_c")
);
CREATE TABLE IF NOT EXISTS "Communication_Network_Wikipedia_GRAPH_d" (
	"idGRAPH_d"	INTEGER,
	"editor_user_id"	INTEGER,
	"edited_talk_page_user_id"	INTEGER,
	PRIMARY KEY("idGRAPH_d")
);
CREATE TABLE IF NOT EXISTS "GitHub_Social_Network_GRAPH_a" (
	"idGRAPH_a"	INTEGER,
	"id_1"	INTEGER,
	"id_2"	INTEGER,
	PRIMARY KEY("idGRAPH_a")
);
CREATE TABLE IF NOT EXISTS "Social_Circles_from_Facebook_GRAPH_b" (
	"idGRAPH_b"	INTEGER,
	"profile_id"	INTEGER,
	"has_friend_profile_id"	INTEGER,
	PRIMARY KEY("idGRAPH_b")
);
CREATE TABLE IF NOT EXISTS "NODES_GRAPH_a" (
	"id"	INTEGER,
	"name"	TEXT,
	"ml_target_(0=web_dev_1=ML_dev)"	INTEGER,
	PRIMARY KEY("id")
);
CREATE TABLE IF NOT EXISTS "NODES_GRAPH_b" (
	"unq_profile_id"	INTEGER,
	PRIMARY KEY("unq_profile_id")
);
CREATE TABLE IF NOT EXISTS "df_betweenness_centrality_GRAPH_b" (
	"idf2"	INTEGER,
	"node"	INTEGER,
	"bc"	REAL,
	FOREIGN KEY("node") REFERENCES "NODES_GRAPH_b"("unq_profile_id") ON UPDATE CASCADE,
	PRIMARY KEY("idf2")
);
CREATE TABLE IF NOT EXISTS "df_betweenness_centrality_GRAPH_c" (
	"idf3"	INTEGER,
	"node"	INTEGER,
	"betweenness_centrality"	REAL,
	PRIMARY KEY("idf3"),
	FOREIGN KEY("node") REFERENCES "NODES_GRAPH_c"("also_bought_this_product_id_NUMBER_OF_NODES") ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS "df_betweenness_centrality_GRAPH_d" (
	"idf4"	INTEGER,
	"node"	INTEGER,
	"betweenness_centrality"	REAL,
	PRIMARY KEY("idf4"),
	FOREIGN KEY("node") REFERENCES "NODES_GRAPH_d"("unq_edited_talk_page_user_id_NUMBER_OF_NODES") ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS "df_closeness_centrality_a" (
	"idf5"	INTEGER,
	"node"	INTEGER,
	"closeness_centrality"	REAL,
	PRIMARY KEY("idf5"),
	FOREIGN KEY("node") REFERENCES "NODES_GRAPH_a"("id") ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS "df_closeness_centrality_b" (
	"idf6"	INTEGER,
	"node"	INTEGER,
	"closeness_centrality"	REAL,
	PRIMARY KEY("idf6"),
	FOREIGN KEY("node") REFERENCES "NODES_GRAPH_b"("unq_profile_id") ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS "df_closeness_centrality_c2" (
	"idf7"	INTEGER,
	"random_c_node"	INTEGER,
	"closeness_centrality"	REAL,
	FOREIGN KEY("random_c_node") REFERENCES "NODES_GRAPH_c"("also_bought_this_product_id_NUMBER_OF_NODES") ON UPDATE CASCADE,
	PRIMARY KEY("idf7")
);
CREATE TABLE IF NOT EXISTS "df_clustering_coefficient_a" (
	"idf9"	INTEGER,
	"node"	INTEGER,
	"clustering_coefficient"	REAL,
	PRIMARY KEY("idf9"),
	FOREIGN KEY("node") REFERENCES "NODES_GRAPH_a"("id") ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS "df_clustering_coefficient_b" (
	"idf10"	INTEGER,
	"node"	INTEGER,
	"clustering_coefficient"	REAL,
	PRIMARY KEY("idf10"),
	FOREIGN KEY("node") REFERENCES "NODES_GRAPH_b"("unq_profile_id") ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS "df_clustering_coefficient_c" (
	"idf11"	INTEGER,
	"node"	INTEGER,
	"clustering_coefficient"	REAL,
	PRIMARY KEY("idf11"),
	FOREIGN KEY("node") REFERENCES "NODES_GRAPH_c"("also_bought_this_product_id_NUMBER_OF_NODES") ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS "df_clustering_coefficient_d" (
	"idf12"	INTEGER,
	"node"	INTEGER,
	"clustering_coefficient"	REAL,
	PRIMARY KEY("idf12"),
	FOREIGN KEY("node") REFERENCES "NODES_GRAPH_d"("unq_edited_talk_page_user_id_NUMBER_OF_NODES") ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS "df_degree_a" (
	"idf13"	INTEGER,
	"nodes"	INTEGER,
	"degrees"	INTEGER,
	PRIMARY KEY("idf13"),
	FOREIGN KEY("nodes") REFERENCES "NODES_GRAPH_a"("id") ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS "df_degree_b" (
	"idf14"	INTEGER,
	"nodes"	INTEGER,
	"degrees"	INTEGER,
	PRIMARY KEY("idf14"),
	FOREIGN KEY("nodes") REFERENCES "NODES_GRAPH_b"("unq_profile_id") ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS "df_degree_c" (
	"idf15"	INTEGER,
	"nodes"	INTEGER,
	"degrees"	INTEGER,
	PRIMARY KEY("idf15"),
	FOREIGN KEY("nodes") REFERENCES "NODES_GRAPH_c"("also_bought_this_product_id_NUMBER_OF_NODES") ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS "df_degree_d" (
	"idf16"	INTEGER,
	"nodes"	INTEGER,
	"degrees"	INTEGER,
	FOREIGN KEY("nodes") REFERENCES "NODES_GRAPH_d"("unq_edited_talk_page_user_id_NUMBER_OF_NODES") ON UPDATE CASCADE,
	PRIMARY KEY("idf16")
);
CREATE TABLE IF NOT EXISTS "df_degree_centrality_a" (
	"idf17"	INTEGER,
	"node"	INTEGER,
	"degree_centrality"	REAL,
	FOREIGN KEY("node") REFERENCES "NODES_GRAPH_a"("id") ON UPDATE CASCADE,
	PRIMARY KEY("idf17")
);
CREATE TABLE IF NOT EXISTS "df_degree_centrality_b" (
	"idf18"	INTEGER,
	"node"	INTEGER,
	"degree_centrality"	REAL,
	FOREIGN KEY("node") REFERENCES "NODES_GRAPH_b"("unq_profile_id") ON UPDATE CASCADE,
	PRIMARY KEY("idf18")
);
CREATE TABLE IF NOT EXISTS "df_degree_centrality_c" (
	"idf19"	INTEGER,
	"node"	INTEGER,
	"degree_centrality"	REAL,
	FOREIGN KEY("node") REFERENCES "NODES_GRAPH_c"("also_bought_this_product_id_NUMBER_OF_NODES") ON UPDATE CASCADE,
	PRIMARY KEY("idf19")
);
CREATE TABLE IF NOT EXISTS "df_degree_centrality_d" (
	"idf20"	INTEGER,
	"node"	INTEGER,
	"degree_centrality"	REAL,
	PRIMARY KEY("idf20"),
	FOREIGN KEY("node") REFERENCES "NODES_GRAPH_d"("unq_edited_talk_page_user_id_NUMBER_OF_NODES") ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS "df_eigenvector_centrality_a" (
	"idf21"	INTEGER,
	"node"	INTEGER,
	"eigenvector_centrality"	REAL,
	PRIMARY KEY("idf21"),
	FOREIGN KEY("node") REFERENCES "NODES_GRAPH_a"("id") ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS "df_eigenvector_centrality_b" (
	"idf22"	INTEGER,
	"node"	INTEGER,
	"eigenvector_centrality"	REAL,
	PRIMARY KEY("idf22"),
	FOREIGN KEY("node") REFERENCES "NODES_GRAPH_b"("unq_profile_id") ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS "df_eigenvector_centrality_c" (
	"idf23"	INTEGER,
	"node"	INTEGER,
	"eigenvector_centrality"	REAL,
	PRIMARY KEY("idf23"),
	FOREIGN KEY("node") REFERENCES "NODES_GRAPH_c"("also_bought_this_product_id_NUMBER_OF_NODES") ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS "df_eigenvector_centrality_d" (
	"idf24"	INTEGER,
	"node"	INTEGER,
	"eigenvector_centrality"	REAL,
	PRIMARY KEY("idf24"),
	FOREIGN KEY("node") REFERENCES "NODES_GRAPH_d"("unq_edited_talk_page_user_id_NUMBER_OF_NODES") ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS "df_local_bridges_a" (
	"idf25"	INTEGER,
	"node"	INTEGER,
	"local_bridge"	INTEGER,
	FOREIGN KEY("node") REFERENCES "NODES_GRAPH_a"("id") ON UPDATE CASCADE,
	PRIMARY KEY("idf25")
);
CREATE TABLE IF NOT EXISTS "df_local_bridges_b" (
	"idf26"	INTEGER,
	"node"	INTEGER,
	"local_bridge"	INTEGER,
	FOREIGN KEY("node") REFERENCES "NODES_GRAPH_b"("unq_profile_id") ON UPDATE CASCADE,
	PRIMARY KEY("idf26")
);
CREATE TABLE IF NOT EXISTS "df_metric_means_for_GRAPH_c" (
	"GRAPH_c"	TEXT,
	"means"	REAL
);
CREATE TABLE IF NOT EXISTS "df_triangles_a" (
	"idf27"	INTEGER,
	"node"	INTEGER,
	"number_of_triangles"	INTEGER,
	FOREIGN KEY("node") REFERENCES "NODES_GRAPH_a"("id") ON UPDATE CASCADE,
	PRIMARY KEY("idf27")
);
CREATE TABLE IF NOT EXISTS "df_triangles_b" (
	"idf28"	INTEGER,
	"node"	INTEGER,
	"number_of_triangles"	INTEGER,
	PRIMARY KEY("idf28"),
	FOREIGN KEY("node") REFERENCES "NODES_GRAPH_b"("unq_profile_id") ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS "df_triangles_c" (
	"idf29"	INTEGER,
	"node"	INTEGER,
	"total_degree"	INTEGER,
	"reciprocal_degree"	INTEGER,
	"number_of_directed_triangles"	INTEGER,
	FOREIGN KEY("node") REFERENCES "NODES_GRAPH_c"("also_bought_this_product_id_NUMBER_OF_NODES") ON UPDATE CASCADE,
	PRIMARY KEY("idf29")
);
CREATE TABLE IF NOT EXISTS "df_triangles_d" (
	"idf30"	INTEGER,
	"node"	INTEGER,
	"total_degree"	INTEGER,
	"reciprocal_degree"	INTEGER,
	"number_of_directed_triangles"	INTEGER,
	PRIMARY KEY("idf30"),
	FOREIGN KEY("node") REFERENCES "NODES_GRAPH_d"("unq_edited_talk_page_user_id_NUMBER_OF_NODES") ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS "edges_of_strongly_connected_directed_subgraph_of_GRAPH_c" (
	"deafuid"	INTEGER,
	"from_node"	INTEGER,
	"to_node"	INTEGER,
	FOREIGN KEY("from_node") REFERENCES "Amazon_Product_Co_Purchasing_Network_GRAPH_c"("from_node_id") ON UPDATE CASCADE,
	FOREIGN KEY("to_node") REFERENCES "Amazon_Product_Co_Purchasing_Network_GRAPH_c"("to_node_id") ON UPDATE CASCADE,
	PRIMARY KEY("deafuid")
);
CREATE TABLE IF NOT EXISTS "edges_of_strongly_connected_directed_subgraph_of_GRAPH_d" (
	"idead"	INTEGER,
	"from_node"	INTEGER,
	"to_node"	INTEGER,
	FOREIGN KEY("from_node") REFERENCES "Communication_Network_Wikipedia_GRAPH_d"("editor_user_id") ON UPDATE CASCADE,
	PRIMARY KEY("idead"),
	FOREIGN KEY("to_node") REFERENCES "Communication_Network_Wikipedia_GRAPH_d"("edited_talk_page_user_id") ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS "nodes_of_strongly_connected_directed_subgraph_of_GRAPH_c" (
	"deafid"	INTEGER,
	"Nodes"	INTEGER,
	PRIMARY KEY("deafid"),
	FOREIGN KEY("Nodes") REFERENCES "NODES_GRAPH_c"("also_bought_this_product_id_NUMBER_OF_NODES") ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS "nodes_of_strongly_connected_directed_subgraph_of_GRAPH_d" (
	"deaid"	INTEGER,
	"nodes"	INTEGER,
	FOREIGN KEY("nodes") REFERENCES "NODES_GRAPH_d"("unq_edited_talk_page_user_id_NUMBER_OF_NODES") ON UPDATE CASCADE,
	PRIMARY KEY("deaid")
);
CREATE TABLE IF NOT EXISTS "modularity_of_graphs" (
	"idf98765"	INTEGER,
	"graph"	TEXT,
	"modularity"	REAL,
	PRIMARY KEY("idf98765")
);
CREATE TABLE IF NOT EXISTS "df_closeness_centrality_d2" (
	"idf8"	INTEGER,
	"random_d_node"	INTEGER,
	"closeness_centrality"	REAL,
	FOREIGN KEY("random_d_node") REFERENCES "NODES_GRAPH_d"("unq_edited_talk_page_user_id_NUMBER_OF_NODES") ON UPDATE CASCADE,
	PRIMARY KEY("idf8")
);
CREATE TABLE IF NOT EXISTS "df_average_number_of_triangles_for_undirected_graphs" (
	"graph"	TEXT,
	"average_number_of_triangles"	INTEGER
);
COMMIT;
