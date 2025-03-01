# import os
# from pathlib import Path

# import torch
# import wandb
# from torch.multiprocessing import set_start_method
# from torch.utils.data import DataLoader

# from datasets.build import build_dataset, build_datasets
# from models.build import build_encoder
# from servers.build import get_server_type, build_server
# from clients.build import get_client_type
# from evalers.build import get_evaler_type
# # from trainer.build import build_trainer
# # from trainers import Trainer
# from trainers.build import get_trainer_type

# from utils import initalize_random_seed

# import hydra
# from omegaconf import DictConfig
# import omegaconf
# import coloredlogs, logging
# # import loggings
# logger = logging.getLogger(__name__)
# # coloredlogs.install(fmt='%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s')
# coloredlogs.install(level='INFO', fmt='%(asctime)s %(name)s[%(process)d] %(message)s', datefmt='%m-%d %H:%M:%S')

# wandb.require("service")

# @hydra.main(version_base=None, config_path="configs", config_name="config")
# def main(args : DictConfig) -> None:

#     torch.multiprocessing.set_sharing_strategy('file_system')
#     set_start_method('spawn', True)
#     # pid = os.getpid()
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

#     args.log_dir = Path(args.log_dir)
#     exp_name = args.exp_name if args.remark == "" else f"{args.exp_name}_{args.remark}"
#     args.log_dir = args.log_dir / args.dataset.name / exp_name
#     print(exp_name)
#     if not args.log_dir.exists():
#         args.log_dir.mkdir(parents=True, exist_ok=True)

#     ## Wandb
#     if args.wandb:
#         wandb.init(entity='federated_learning',
#                 project=args.project,
#                 group=f'{args.split.mode}{str(args.split.alpha) if args.split.mode == "dirichlet" else ""}',
#                 job_type=exp_name,
#                 dir=args.log_dir,)
#         wandb.run.name = exp_name
#         wandb.config.update(omegaconf.OmegaConf.to_container(
#             args, resolve=True, throw_on_missing=True
#         ))

#     initalize_random_seed(args)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print("device", device)

#     model = build_encoder(args)
#     client_type = get_client_type(args)
#     server = build_server(args)
#     datasets = build_datasets(args)
#     evaler_type = get_evaler_type(args)

#     trainer_type = get_trainer_type(args)
#     trainer = trainer_type(model=model, client_type=client_type, server=server, evaler_type=evaler_type,
#                            datasets=datasets,
#                            device=device, args=args, config=None)
#     trainer.train()


# if __name__ == '__main__':
#     main()

import os
from pathlib import Path

import torch
import wandb
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader

from datasets.build import build_dataset, build_datasets
from models.build import build_encoder
from servers.build import get_server_type, build_server
from clients.build import get_client_type
from evalers.build import get_evaler_type
from trainers.build import get_trainer_type

from utils import initalize_random_seed

import hydra
from omegaconf import DictConfig
import omegaconf
import coloredlogs, logging

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', fmt='%(asctime)s %(name)s[%(process)d] %(message)s', datefmt='%m-%d %H:%M:%S')

wandb.require("service")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(args: DictConfig) -> None:
    # Set multiprocessing start method to 'spawn' for compatibility with CUDA
    torch.multiprocessing.set_sharing_strategy('file_system')
    set_start_method('spawn', force=True)

    # Ensure CUDA device is set
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.main_gpu  # Use the specified GPU

    # Set up experiment directory
    args.log_dir = Path(args.log_dir)
    exp_name = args.exp_name if args.remark == "" else f"{args.exp_name}_{args.remark}"
    args.log_dir = args.log_dir / args.dataset.name / exp_name
    print(f"Experiment name: {exp_name}")
    if not args.log_dir.exists():
        args.log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize WandB for logging
    if args.wandb:
        wandb.init(
            entity='federated_learning',
            project=args.project,
            group=f'{args.split.mode}{str(args.split.alpha) if args.split.mode == "dirichlet" else ""}',
            job_type=exp_name,
            dir=args.log_dir,
        )
        wandb.run.name = exp_name
        wandb.config.update(omegaconf.OmegaConf.to_container(
            args, resolve=True, throw_on_missing=True
        ))

    # Initialize random seed for reproducibility
    initalize_random_seed(args)

    # Set device to GPU if available
    device = torch.device(f"cuda:{args.main_gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model and move it to the specified device
    model = build_encoder(args).to(device)
    print(f"Model is on device: {next(model.parameters()).device}")

    # Build other components
    client_type = get_client_type(args)
    server = build_server(args)
    datasets = build_datasets(args)
    evaler_type = get_evaler_type(args)

    # Build trainer and start training
    trainer_type = get_trainer_type(args)
    trainer = trainer_type(
        model=model,
        client_type=client_type,
        server=server,
        evaler_type=evaler_type,
        datasets=datasets,
        device=device,
        args=args,
        config=None,
    )

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
