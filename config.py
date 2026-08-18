# config.py — All hyperparameters


import os

# Paths 

ROOT          = r'C:\Users\danish\Desktop\pap_cell_project\data'
TRAIN_CSV     = os.path.join(ROOT, 'isbi2025-ps3c-train-dataset.csv')
TRAIN_DIR     = os.path.join(ROOT, 'isbi2025-ps3c-train-dataset')
TEST_DATA_DIR = os.path.join(ROOT, 'isbi2025-ps3c-test-dataset')

# Test CSV 
TEST_CSV_CANDIDATES = [
    os.path.join(ROOT, 'isbi2025-ps3c-test-dataset-annotated.csv'),
    os.path.join(ROOT, 'isbi2025-ps3c-test-dataset.csv'),
]

# Model save paths
MODEL1_CKPT = os.path.join(ROOT, 'checkpoint_model1_final.pth')
MODEL1_BEST = os.path.join(ROOT, 'best_model_1.pth')
MODEL2_CKPT = os.path.join(ROOT, 'checkpoint_model2_final.pth')
MODEL2_BEST = os.path.join(ROOT, 'best_model_v2_2.pth')


# Data
DROP_BOTHCELLS = True
CLASSES        = ['healthy', 'rubbish', 'unhealthy']  # sorted alphabetically
SEED           = 42
VAL_SIZE       = 0.15

# Training
IMG_SIZE_M1  = 300     # Model 1 resolution
IMG_SIZE_M2  = 300     # Model 2 resolution 
BATCH_SIZE   = 32      
EPOCHS       = 35
PATIENCE     = 6
LR           = 2e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.05
