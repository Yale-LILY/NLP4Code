from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.cli import LightningCLI

# see https://github.com/PyTorchLightning/pytorch-lightning/issues/10349
import warnings

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

# class MyLightningCLI(LightningCLI):
#     def add_arguments_to_parser(self, parser):
#         parser.link_arguments("data.init_args.transformer_model_name", "model.init_args.transformer_model_name")

cli = LightningCLI(LightningModule, LightningDataModule, 
                   subclass_mode_model=True, subclass_mode_data=True,
                   save_config_callback=None)