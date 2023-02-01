import os

from typing import Optional, Union, List
from pytorch_lightning.loggers import NeptuneLogger, CSVLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities import rank_zero_only

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

class PatchedWandbLogger(WandbLogger):
    def __init__(self, entity: str, project: str, name: str, log_model: bool, save_code: bool, save_dir: str,
                 tags: List[str] = None, offline: bool = False, *args, **kwargs):
        # try to get the exp name and save dir from env var to support multiple runs
        env_var_name = os.getenv('EXP_NAME')
        if env_var_name is not None:
            name = env_var_name
            save_dir = env_var_name

        # put necessary init vars for wandb
        kwargs['entity'] = entity 
        kwargs['save_code'] = save_code
        kwargs['save_dir'] = save_dir
        kwargs['dir'] = save_dir # fix a bug for wandb logger

        # remove the preceeding folder name
        processed_name = name.split('/')[-1]
        if tags is None:
            kwargs['tags'] = processed_name.split('-')
        else:
            kwargs['tags'] = tags
        
        # fail-safe for uploading the tmp exp for debugging
        if "tmp" in processed_name and not offline:
            print(f"WandbLogger: {processed_name} is a tmp exp so running in offline mode")
            kwargs['offline'] = True
        # TODO: manually disable online logs for now
        kwargs['offline'] = True

        # create the save_dir if it doesn't exist
        print(f"ready to create save_dir: {save_dir}", flush=True)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        super().__init__(name=processed_name, project=project, log_model=log_model, *args, **kwargs)

    @rank_zero_only
    def log_code(self):

        # log the yaml and py files
        root = "."
        print(f"saving all files in {os.path.abspath(root)}")
        result = self.experiment.log_code(root=root, 
                    include_fn=(lambda path: path.endswith(".py") or \
                                       path.endswith(".yaml")),
                    exclude_fn=(lambda path: ".venv" in path or \
                                             "debug-tmp" in path))
        if result is not None:
            print("########################################")
            print("######## Logged code to wandb. #########")
            print("########################################")
        else:
            print("######## logger inited but not successfully saved #########")

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
