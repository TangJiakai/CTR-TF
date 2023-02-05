import tensorflow as tf
import random
import numpy as np
import os


def set_seeds(seed=2023):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)