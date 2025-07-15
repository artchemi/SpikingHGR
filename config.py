# NOTE: Данные для обучения
FOLDER_PATH = "Ninapro_DB5"
# STD_MEAN_PATH = "scaling_params.json"

EXERCISES = ["E2", "E3"]
MYO_PREF = "elbow"

# Our: [0, 13, 15, 17, 18, 19, 20, 25, 26]
# MIC-Laboratory/IEEE-NER-2023-EffiE: [0, 13, 15, 17, 18, 25, 26]
GESTURE_INDEXES_B = [0, 13, 15, 18, 19]
GESTURE_INDEXES_C = [34, 38, 43, 46]
GESTURE_INDEXES_MAIN = GESTURE_INDEXES_B + GESTURE_INDEXES_C
CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7]    # NOTE: CAM-MS-RS - [0, 3, 4, 5, 6]
NUM_CLASSES = len(GESTURE_INDEXES_MAIN)

VALID_SIZE = 0.5    # Доля от test
TEST_SIZE = 0.3    # Доля от общей выборки
BATCH_SIZE = 2**7    # ???

INIT_LR= 0.001    # 1e-4

WINDOW_SIZE = 52
STEP_SIZE = 16

EPOCHS = 1000
PATIENCE = 50

# ! Гиперпараметры базовой модели
INPUT_SHAPE_BASE = (WINDOW_SIZE, len(CHANNELS), 1)
FILTERS_SNN = (3, 6)
KERNEL_SIZE_SNN = (1, 3)    # (3, 3)
POOL_SIZE_SNN = (1, 2)
P_DROPOUT_SNN = 0.25
NUM_STEPS = 32    # ???

# ! Параметры LIF нейронов
BETA_LIF = 0.9
LEARN_THRESHOLD_LIF = True
THRS_LIF = 1.

# ! Параметры синаптического нейрона
ALPHA_SYNAPTIC = 0.9
BETA_SYNAPTIC = 0.5 

# ! Гиперпараметры рекуррентной модели
HIDDEN_DIM_RNN_LIF = 20
TIME_STEPS_RNN_LIF = 32
DT = 0.0001

# NOTE: Прочее
GLOBAL_SEED = 42
SAVE_PATH = "checkpoints/model"

