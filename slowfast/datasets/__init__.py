#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .build import DATASET_REGISTRY, build_dataset  # noqa

# Remember to add new dataset here. @DATASET_REGISTRY.register() is not executed otherwise.
from .egtea_gaze import Egteagaze
from .ego4d_gaze import Ego4dgaze
from .egoexo4d_gaze import Egoexo4dgaze

try:
    from .ptv_datasets import Ptvcharades, Ptvkinetics, Ptvssv2  # noqa
except Exception:
    pass
