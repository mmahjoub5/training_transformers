from typing import Dict, Type

from src.data.base_preprocessor import BasePreprocessor
from src.data.eli_preprocess import ELI5Preprocessor_CLM, ELI5Preprocessor_QA
from src.data.socratic_dialog_preprocess import SocraticPreprocessor
from src.data.squad_preprocess import SQuADPreprocessor

PREPROCESSOR_REGISTRY: Dict[str, Type[BasePreprocessor]] = {
    "ELI5Preprocessor_QA": ELI5Preprocessor_QA,
    "ELI5Preprocessor_CLM": ELI5Preprocessor_CLM,
    "SocraticPreprocessor": SocraticPreprocessor,
    "SQuADPreprocessor": SQuADPreprocessor,
}
