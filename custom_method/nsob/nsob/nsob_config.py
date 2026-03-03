"""
Object_nerfstudio configuration file.
"""
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig

from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig

from nerfstudio.engine.trainer import TrainerConfig
from nsob.nsob_trainer import NsobTrainerConfig

from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nsob.nsob_pipeline import NsobPipelineConfig

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nsob.data.nsob_datamanager import NsobDataManager, NsobDataManagerConfig
# from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nsob.data.nsob_dataparser import NsobDataParserConfig

from nerfstudio.data.datasets.depth_dataset import DepthDataset

# from nerfstudio.models.nerfacto import NerfactoModelConfig
from nsob.nsob import NsobModelConfig
from nsob.nsobject import NsobjectModelConfig
from nerfstudio.models.depth_nerfacto import DepthNerfactoModelConfig

from nerfstudio.plugins.types import MethodSpecification

nsob_method = MethodSpecification(
    config=TrainerConfig(
        method_name="nsob",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=NsobPipelineConfig(
            datamanager=NsobDataManagerConfig(
                dataparser=NsobDataParserConfig(),
                sample_only_in_mask=False,
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3",
                    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                    scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
                ),
                masks_on_gpu = True, # Process masks on GPU for speed at the expense of memory
            ),
            model=NsobModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                background_color="white",
                ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Implementation of object_nerf to nerfstudio",
)

nsobject_method = MethodSpecification(
    config=NsobTrainerConfig(
        method_name="nsobject",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=NsobPipelineConfig(
            datamanager=NsobDataManagerConfig(
                dataparser=NsobDataParserConfig(),
                sample_only_in_mask=False,
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3",
                    optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                    scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
                ),
                masks_on_gpu = True, # Process masks on GPU for speed at the expense of memory
            ),
            model=NsobjectModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                background_color="white",
                ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Implementation of object_nerf for object only to nerfstudio",
)

depth_nsob_method = MethodSpecification(
    config=TrainerConfig(
        method_name="depth-nsob",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                _target=VanillaDataManager[DepthDataset],
                dataparser=NsobDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=DepthNerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Implementation of object_nerf to nerfstudio with depth supervision",
)
