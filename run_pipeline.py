from sketchgraphs_models.graph import train
import sys

sys.argv = ['prog', '--dataset_train', 'sg_t16_train.npy']
train.main()