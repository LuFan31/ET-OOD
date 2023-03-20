from .base_postprocessor import BasePostprocessor
from .Softmaxprocessor import SoftMaxpostprocessor
from .Tenergyprocessor import Tenergyprocessor

def get_postprocessor(name: str, **kwargs):
    postprocessors = {
        "none": BasePostprocessor,
        "msp": SoftMaxpostprocessor,
        "ene": Tenergyprocessor,
    }
    return postprocessors[name](**kwargs)

