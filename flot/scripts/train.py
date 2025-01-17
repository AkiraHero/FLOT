import os
import time
import torch
import argparse
from tqdm import tqdm
from datetime import datetime
from flot.datasets.generic import Batch
from flot.models.scene_flow import FLOT
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#for distributed
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def compute_epe(est_flow, batch):
    """
    Compute EPE during training.

    Parameters
    ----------
    est_flow : torch.Tensor
        Estimated flow.
    batch : flot.datasets.generic.Batch
        Contains ground truth flow and mask.

    Returns
    -------
    epe : torch.Tensor
        Mean EPE for current batch.

    """

    mask = batch["ground_truth"][0][..., 0]
    true_flow = batch["ground_truth"][1]
    error = est_flow - true_flow
    error = error[mask > 0]
    epe_per_point = torch.sqrt(torch.sum(torch.pow(error, 2.0), -1))
    epe = epe_per_point.mean()

    return epe


def compute_loss(est_flow, batch):
    """
    Compute training loss.

    Parameters
    ----------
    est_flow : torch.Tensor
        Estimated flow.
    batch : flot.datasets.generic.Batch
        Contains ground truth flow and mask.

    Returns
    -------
    loss : torch.Tensor
        Training loss for current batch.

    """

    mask = batch["ground_truth"][0][..., 0]
    true_flow = batch["ground_truth"][1]
    error = est_flow - true_flow
    error = error[mask > 0]
    loss = torch.mean(torch.abs(error))

    return loss


def train(scene_flow, trainloader, delta, optimizer, scheduler, path2log, nb_epochs, rank=0):
    """
    Train scene flow model.

    Parameters
    ----------
    scene_flow : flot.models.FLOT
        FLOT model
    trainloader : flots.datasets.generic.SceneFlowDataset
        Dataset loader.
    delta : int
        Frequency of logs in number of iterations.
    optimizer : torch.optim.Optimizer
        Optimiser.
    scheduler :
        Scheduler.
    path2log : str
        Where to save logs / model.
    nb_epochs : int
        Number of epochs.

    """

    # Log directory
    if not os.path.exists(path2log):
        os.makedirs(path2log)
    writer = SummaryWriter(path2log)

    # Reload state
    total_it = 0
    epoch_start = 0

    # Train
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # scene_flow = scene_flow.to(device, non_blocking=True)
    
    for epoch in range(epoch_start, nb_epochs):

        # Init.
        running_epe = 0
        running_loss = 0

        # Train for 1 epoch
        start = time.time()
        scene_flow = scene_flow.train()
        for it, batch in enumerate(tqdm(trainloader)):

            # Send data to GPU
            batch = batch.to(rank, non_blocking=True)

            # Gradient step
            optimizer.zero_grad()
            est_flow = scene_flow(batch["sequence"])
            loss = compute_loss(est_flow, batch)
            loss.backward()
            optimizer.step()

            # Loss evolution
            running_loss += loss.item()
            running_epe += compute_epe(est_flow, batch).item()
            
            # Logs
            if rank == 0 and it % delta == delta - 1:
                # Print / save logs
                writer.add_scalar("Loss/epe", running_epe / delta, total_it)
                writer.add_scalar("Loss/loss", running_loss / delta, total_it)
                print(
                    "Epoch {0:d} - It. {1:d}: loss = {2:e}".format(
                        epoch, total_it, running_loss / delta
                    )
                )
                print(time.time() - start, "seconds")
                # Re-init.
                running_epe = 0
                running_loss = 0
                start = time.time()

            total_it += 1

        # Scheduler
        scheduler.step()

        # Save model after each epoch
        if isinstance(scene_flow, DDP):
            state = {
                "nb_iter": scene_flow.module.nb_iter,
                "model": scene_flow.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
        else:
            state = {
                "nb_iter": scene_flow.nb_iter,
                "model": scene_flow.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
        if rank == 0:
            torch.save(state, os.path.join(path2log, "model_epch{}.tar".format(epoch)))

    #
    print("Finished Training")

    return None


def my_main(dataset_name, nb_iter, batch_size, max_points, nb_epochs, distributed=False, world_size=2, rank=0):
    """
    Entry point of the script.

    Parameters
    ----------
    dataset_name : str
        Version of FlyingThing3D used for training: 'HPLFlowNet' / 'flownet3d'.
    nb_iter : int
        Number of unrolled iteration of Sinkhorn algorithm in FLOT.
    batch_size : int
        Batch size.
    max_points : int
        Number of points in point clouds.
    nb_epochs : int
        Number of epochs.

    Raises
    ------
    ValueError
        If dataset_name is an unknow dataset.

    """

    # Path to current file
    pathroot = os.path.dirname(__file__)

    # Path to dataset
    if dataset_name.lower() == "HPLFlowNet".lower():
        path2data = os.path.join(
            pathroot, "..", "data", "HPLFlowNet", "FlyingThings3D_subset_processed_35m"
        )
        from flot.datasets.flyingthings3d_hplflownet import FT3D as CurrentDataset

        lr_lambda = lambda epoch: 1.0 if epoch < 50 else 0.1
    elif dataset_name.lower() == "flownet3d".lower():
        path2data = os.path.join(
            pathroot, "..", "data", "flownet3d", "data_processed_maxcut_35_20k_2k_8192"
        )
        from flot.datasets.flyingthings3d_flownet3d import FT3D as CurrentDataset

        lr_lambda = lambda epoch: 1.0 if epoch < 340 else 0.1
    elif dataset_name.lower() == "waymo".lower():
        path2data = "s3://juxiaoliang/dataset/waymo/waymo_sf_processed"
        from flot.datasets.waymo_dataset import WaymoDataset as CurrentDataset
        
        lr_lambda = lambda epoch: 1.0 if epoch < 340 else 0.1
    else:
        raise ValueError("Invalid dataset name: " + dataset_name)
    
    if distributed:
        setup(rank, world_size)
        


    # Training dataset
    ds = CurrentDataset(root_dir=path2data, nb_points=max_points, mode="train")
    
    train_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(ds)
    
    trainloader = DataLoader(
        ds,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False if distributed else True,
        num_workers=6,
        collate_fn=Batch,
        drop_last=True,
        sampler=train_sampler
    )

    # Model
    scene_flow = FLOT(nb_iter=nb_iter)
    
    if distributed:
        scene_flow.to(rank)
        scene_flow = DDP(scene_flow, device_ids=[rank])

    # Optimizer
    optimizer = torch.optim.Adam(scene_flow.parameters(), lr=1e-3)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Log directory
    now = datetime.now().strftime("%y_%m_%d-%H_%M_%S_%f")
    # now += "__Iter_" + str(scene_flow.nb_iter)
    # now += "__Pts_" + str(max_points)
    path2log = os.path.join(pathroot, "..", "experiments", "logs_" + dataset_name, now)

    # Train
    if rank == 0:
        print("Training started. Logs in " + path2log)
    train(scene_flow, trainloader, 500, optimizer, scheduler, path2log, nb_epochs, rank)
    cleanup()
    print("program exit..")
    return None


def my_main_distributed_wrapper(rank, *args):
    my_main(*args, rank=rank)
    



if __name__ == "__main__":

    # Args
    parser = argparse.ArgumentParser(description="Train FLOT.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="HPLFlowNet",
        help="Training dataset. Either HPLFlowNet or " + "flownet3d.",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--nb_epochs", type=int, default=40, help="Number of epochs.")
    parser.add_argument(
        "--nb_points",
        type=int,
        default=2048,
        help="Maximum number of points in point cloud.",
    )
    parser.add_argument(
        "--nb_iter",
        type=int,
        default=1,
        help="Number of unrolled iterations of the Sinkhorn " + "algorithm.",
    )
    parser.add_argument("--distributed", type=bool, default=False)
    parser.add_argument("--ngpus", type=int, default=4, help="GPU num")

    args = parser.parse_args()

    # Launch training
    if not args.distributed:
        my_main(args.dataset, args.nb_iter, args.batch_size, args.nb_points, args.nb_epochs)
    else:
        mp.spawn(my_main_distributed_wrapper, 
                 args=(args.dataset, args.nb_iter, args.batch_size, args.nb_points, args.nb_epochs, True, args.ngpus),
                 nprocs=args.ngpus,
                 join=True
                 )
        
