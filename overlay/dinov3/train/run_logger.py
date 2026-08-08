"""训练运行日志器：按运行实例隔离日志文件。

共享代码/共享盘场景下，app.log_module 默认把所有运行写进同一个 logs/app.log，
多任务并行时日志混杂。本模块读取环境变量 DINOV3_RUN_LOG 作为日志文件路径
（由 01_train.sh / run_ablation.sh 设为每次运行的 $OUTPUT_DIR/app.log），
未设置时回退到 logs/app.log（与旧行为一致）。
"""

import logging
import os


def get_run_logger():
    logfile = os.environ.get("DINOV3_RUN_LOG", "logs/app.log")
    logger_ = logging.getLogger("finetune_v2")
    logger_.propagate = False
    if not logger_.hasHandlers():
        logger_.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s [%(processName)s][%(thread)d] %(levelname)s %(message)s"
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger_.addHandler(stream_handler)

        log_dir = os.path.dirname(logfile)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(logfile, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger_.addHandler(file_handler)
    return logger_


logger = get_run_logger()
