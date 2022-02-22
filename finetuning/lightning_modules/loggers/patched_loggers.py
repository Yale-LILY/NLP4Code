import os
import neptune
from typing import Optional, Union
from pytorch_lightning.loggers import NeptuneLogger, CSVLogger, TensorBoardLogger

class PatchedNeptuneLogger(NeptuneLogger):
    def __init__(self, project_name: str, *args, **kwargs):
        api_key = os.getenv('NEPTUNE_API_KEY')
        if api_key is None:
            raise ValueError("Please provide an API key for the neptune logger in the env vars.")
        # exp_name = os.getenv('PL_LOG_DIR').split('/')[0]
        # exp_name = os.getenv('AMLT_JOB_NAME')

        kwargs['api_key'] = api_key
        # kwargs['experiment_id'] = exp_name
        kwargs['project'] = project_name
        kwargs['source_files'] = ['**/*.py', '**/*.yaml', '**/*.sh']

        super().__init__(*args, **kwargs)

class PatchedCSVLogger(CSVLogger):
    def __init__(self, 
        name: Optional[str] = "default",
        version: Optional[Union[int, str]] = None,
        prefix: str = "", 
    ):
        save_dir = os.getenv('PL_LOG_DIR')
        super().__init__(save_dir, name, version, prefix)

class PatchedTensorBoardLogger(TensorBoardLogger):
    def __init__(self,
        name: Optional[str] = "default",
        version: Optional[Union[int, str]] = None,
        log_graph: bool = False,
        default_hp_metric: bool = True,
        prefix: str = "",
        sub_dir: Optional[str] = None,
    ):
        save_dir = os.getenv('PL_LOG_DIR')
        super().__init__(save_dir, name, version, log_graph, default_hp_metric, prefix, sub_dir)