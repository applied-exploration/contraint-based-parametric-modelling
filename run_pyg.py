from pyg_homo_dataset import load_homogeneous_dataset

dataset_train_path = 'sequence_data/sg_t16_train.npy'
quantization = {'angle': 127, 'length': 383}
seed = 7


ds_train, weightsgr_train = load_homogeneous_dataset(dataset_train_path, None, quantization, seed, force_entity_categorical_features=False)



def get_parameters(data):
    print("== Data Metrics ==")
    print(f"num_nodes: {data.num_nodes}")
    print(f"num_edges: {data.num_edges}")
    print(f"num_node_features: {data.num_node_features}")
    print(f"Is the graph directed?: {data.is_directed()}")
    print(f"Self loops in graph?: {data.has_self_loops()}")
    print(f"Any isolated nodes?: {data.has_isolated_nodes()}")

single_example = ds_train.__getitem__(0)[0]
get_parameters(single_example)