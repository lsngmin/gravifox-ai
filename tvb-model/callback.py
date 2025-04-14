from tf_keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from config import *

def es():
    return EarlyStopping(
    monitor=MONITER,
    patience=PATIENT,
    restore_best_weights=RESTORE
)

def tb():
    return TensorBoard(
    log_dir= TB_DIR,
    histogram_freq=FREQUENCY
)