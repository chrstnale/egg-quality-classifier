import onnxruntime as ort
import numpy as np


def get_eggmodel(model_path: str):
    session = ort.InferenceSession(model_path)
    return session
