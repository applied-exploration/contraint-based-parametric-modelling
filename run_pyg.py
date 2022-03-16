from pyg_homo_dataset import load_homogeneous_dataset

dataset_train_path = 'sg_t16_train.npy'
quantization = {'angle': 127, 'length': 383}
seed = 7


ds_train, weightsgr_train = load_homogeneous_dataset(dataset_train_path, None, quantization, seed, entity_features=False, edge_features=False, force_entity_categorical_features=False)


ds_train