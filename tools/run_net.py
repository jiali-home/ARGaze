#!/usr/bin/env python3
import torch  # import torch and sklearn at the very beginning to avoid possible incompatible errors
import sklearn

import sys
# sys.path.append('//mnt/data1/jiali/GLC/')
sys.path.append('/mnt/data1/jiali/GLC/')
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from test_gaze_net import test
from train_gaze_net import train
# from stu_tea_train_gaze_net import train
# from student_teacher_training import train


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if cfg.NUM_GPUS > available_gpus:
        if available_gpus == 0:
            raise RuntimeError(
                "NUM_GPUS is set to {} but no CUDA devices are visible. "
                "Check your CUDA setup or lower NUM_GPUS to zero.".format(
                    cfg.NUM_GPUS
                )
            )

        print(
            "[run_net] Requested {} GPUs but only {} available. "
            "Overriding NUM_GPUS to {}.".format(
                cfg.NUM_GPUS, available_gpus, available_gpus
            )
        )
        cfg.defrost()
        cfg.NUM_GPUS = available_gpus
        cfg.freeze()

    # Perform training.
    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)


if __name__ == "__main__":
    main()
