import argparse
import random
import shutil
import time
from functools import partial
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
import pdb
import logging

from scood.data import get_dataloader, get_ext_dataloader
from scood.evaluation import Evaluator
from scood.postprocessors import get_postprocessor
from scood.networks import ResNet18
from scood.trainers import get_ETtrainer, ETtrainer
from scood.utils import load_yaml, setup_logger



def main(args, config):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # expt_output_dir = (output_dir / config["name"]).mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config)
    config_save_path = output_dir / "config.yml"
    shutil.copy(config_path, config_save_path)

    # setup_logger(str(output_dir))
    logging.basicConfig(filename=str(output_dir)+'/log.txt', level=logging.INFO)

    benchmark = config["dataset"]["labeled"]
    if benchmark == "cifar10":
        num_classes = 10
    elif benchmark == "cifar100" or "ima100":
        num_classes = 100

    # Init Datasets ############################################################
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


    set_seed(3407)
    get_dataloader_default = partial(
        get_dataloader,
        root_dir=args.data_dir,
        benchmark=benchmark,
        num_classes=num_classes,
    )

    labeled_train_loader = get_dataloader_default(
        name=config["dataset"]["labeled"],
        stage="train",
        batch_size=config["dataset"]["labeled_batch_size"],
        shuffle=True,
        num_workers=args.prefetch
    )

    if config['dataset']['unlabeled'] == "none":
        unlabeled_train_loader = None
    else:
        unlabeled_train_loader = get_dataloader_default(
            name=config["dataset"]["unlabeled"],
            stage="train",
            batch_size=config["dataset"]["unlabeled_batch_size"],
            shuffle=True,
            num_workers=args.prefetch
        )
    
    set_seed(3407)
    get_dataloader_ext = partial(
        get_ext_dataloader,
        root_dir=args.data_dir,
        benchmark=benchmark,
        num_classes=num_classes,
    )

    labeled_aug_loader = get_dataloader_ext(
        name=config["dataset"]["labeled"],
        stage="train",
        batch_size=config["dataset"]["labeled_batch_size"],
        shuffle=True,
        num_workers=args.prefetch
    )

    if config['dataset']['unlabeled'] == "none":
        unlabeled_aug_loader = None
    else:
        unlabeled_aug_loader = get_dataloader_ext(
            name=config["dataset"]["unlabeled"],
            stage="train",
            batch_size=config["dataset"]["unlabeled_batch_size"],
            shuffle=True,
            num_workers=args.prefetch
        )


    test_id_loader = get_dataloader_default(
        name=config["dataset"]["labeled"],
        stage="test",
        batch_size=config["dataset"]["test_batch_size"],
        shuffle=False,
        num_workers=args.prefetch
    )

    test_ood_loader_list = []
    for name in config["dataset"]["test_ood"]:
        test_ood_loader = get_dataloader_default(
            name=name,
            stage="test",
            batch_size=config["dataset"]["test_batch_size"],
            shuffle=False,
            num_workers=args.prefetch
        )
        test_ood_loader_list.append(test_ood_loader)


    try:
        num_clusters = config['trainer_args']['num_clusters']
    except KeyError:
        num_clusters = 0

    set_seed(3407)
    net = ResNet18(num_classes=num_classes, dim_aux=num_clusters)

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.ngpu > 0:
        set_seed(3407)
        net.cuda()


    cudnn.benchmark = True 

    set_seed(3407)
    trainer = get_ETtrainer(net,
        labeled_train_loader,
        unlabeled_train_loader,
        labeled_aug_loader,
        unlabeled_aug_loader,
        config['lamda'], 
        config['optim_args'],
        config['trainer_args']
    )

    # Start Training ###########################################################
    set_seed(3407)
    evaluator = Evaluator(net)

    output_dir = Path(args.output_dir)

    begin_epoch = time.time()
    best_accuracy = 0.0
    for epoch in range(0, config["optim_args"]["epochs"]):
        train_metrics = trainer.train_epoch(epoch, output_dir)


        classification_metrics = evaluator.eval_classification(test_id_loader)
        postprocess_args = config["postprocess_args"] if config["postprocess_args"] else {}
        postprocessor = get_postprocessor(config["postprocess"], **postprocess_args)
        evaluator.eval_ood(
            test_id_loader,
            test_ood_loader_list,
            postprocessor=postprocessor,
            method="full",
            dataset_type="scood",
            output_dir = output_dir,
        )

        # Save model
        torch.save(net.state_dict(), output_dir / f"epoch_{epoch}.ckpt")
        if not args.save_all_model:
            # Let us not waste space and delete the previous model
            prev_path = output_dir / f"epoch_{epoch - 1}.ckpt"
            prev_path.unlink(missing_ok=True)

        # save best result
        if classification_metrics["test_accuracy"] >= best_accuracy:
            torch.save(net.state_dict(), output_dir / f"best.ckpt")

            best_accuracy = classification_metrics["test_accuracy"]

        logging.info(
            "Epoch {:3d} | Time {:5d}s | Train Loss {:.4f} | Test Loss {:.3f} | Test Acc {:.2f}".format(
                (epoch + 1),
                int(time.time() - begin_epoch),
                train_metrics["train_loss"],
                classification_metrics["test_loss"],
                100.0 * classification_metrics["test_accuracy"],
            ),
            # flush=True,
        )
        print('Training Completed!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="path to config file",
        default="configs/train/cifar10_ET.yml",
    )
    parser.add_argument(
        "--checkpoint",
        help="specify path to checkpoint if loading from pre-trained model",
        # default="3.1/pretrain.ckpt",
    )
    parser.add_argument(
        "--data_dir",
        help="directory to dataset",
        default="../ICCV21_SCOOD-main/data",
    )
    parser.add_argument(
        "--output_dir",
        help="directory to save experiment artifacts",
        default="output/cifar10",
    )
    parser.add_argument(
        "--save_all_model",
        action="store_true",
        help="whether to save all model checkpoints",
    )
    parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
    parser.add_argument("--prefetch", type=int, default=4, help="pre-fetching threads.")
    
    args = parser.parse_args()

    config = load_yaml(args.config)

    main(args, config)
