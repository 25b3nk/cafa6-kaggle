from .baseline import (
    SimpleProteinClassifier,
    ESM2Classifier,
    ProtBERTClassifier,
    CNNProteinClassifier,
    get_model
)
from .multitask import (
    MultiTaskProteinClassifier,
    MultiTaskESM2Classifier,
    HierarchicalMultiTaskClassifier,
    get_multitask_model
)
