"""Domain constants for activity recognition."""

ACTIVITY_CLASSES = {
    "KhongHanhDong": 0,
    "VayTay": 1,
    "KeoGhe": 2,
    "Dung": 3,
    "Nam": 4,
    "Nga": 5,
    "Ngoi": 6,
    "NhatDo": 7,
}

CLASS_NAMES = list(ACTIVITY_CLASSES.keys())
NUM_CLASSES = len(ACTIVITY_CLASSES)
DEFAULT_CSI_LENGTH = 102
DEFAULT_IMAGE_SIZE = (224, 224)
