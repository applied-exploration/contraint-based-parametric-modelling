from sketchgraphs_models.graph.train.data_loading import load_dataset_and_weights


dataset_train_path = 'sg_t16_train.npy'
quantization = {'angle': 127, 'length': 383}
seed = 7
from torch_geometric.data import Data

ds_train, weightsgr_train = load_dataset_and_weights(dataset_train_path, None, quantization, seed, entity_features=False, edge_features=False, force_entity_categorical_features=False)

first_g = next(iter(ds_train))[0]
data = Data(first_g.node_features, first_g.incidence)
