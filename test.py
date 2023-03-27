import argparse
from functools import partial
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import os

from scood.data import get_dataloader
from scood.evaluation import Evaluator
from scood.networks import ResNet18
from scood.postprocessors import get_postprocessor
from scood.utils import load_yaml


def main(args, config):
    benchmark = config["id_dataset"]
    if benchmark == "cifar10":
        num_classes = 10
        num_clusters = 1024
    elif benchmark == "cifar100":
        num_classes = 100
        num_clusters = 2048

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
        stage="test",
        interpolation=config["interpolation"],
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=args.prefetch
    )
    
    set_seed(3407)
    test_id_loader = get_dataloader_default(name=config["id_dataset"])
    
    set_seed(3407)
    test_ood_loader_list = []
    for name in config["ood_datasets"]:
        test_ood_loader = get_dataloader_default(name=name)
        test_ood_loader_list.append(test_ood_loader)

    set_seed(3407)
    net = ResNet18(num_classes=num_classes, dim_aux=num_clusters)
    checkpoint = args.checkpoint
    if checkpoint:
        net.load_state_dict(torch.load(checkpoint), strict=False)
        print("Checkpoint Loading Completed!")
    net.eval()

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.ngpu > 0:
        set_seed(3407)
        net.cuda()

    cudnn.benchmark = True # fire on all cylinders
    # #torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False

    print("Starting Evaluation...")
    postprocess_args = config["postprocess_args"] if config["postprocess_args"] else {}
    postprocessor = get_postprocessor(config["postprocess"], **postprocess_args)
    
    set_seed(3407)
    evaluator = Evaluator(net)
    
    output_dir = args.csv_path.split('/')
    if len(output_dir) >= 3:
        output_dir = '/'.join(output_dir[:-1])
    else:
        output_dir = output_dir[0]
    evaluator.eval_ood(
        test_id_loader,
        test_ood_loader_list,
        postprocessor=postprocessor,
        method=config["eval_method"],
        dataset_type=config["dataset_type"],
        csv_path=args.csv_path,
        output_dir=output_dir
    )
    print('Evaluation Completed! Results are saved in "{}"'.format(args.csv_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="path to config file",
        default="configs/test/cifar10.yml",
    )
    parser.add_argument(
        "--checkpoint",
        help="path to model checkpoint",
        default="output/cifar10/best.ckpt",
    )
    parser.add_argument(
        "--data_dir",
        help="directory to dataset",
        default="../SCOOD-OT+PASS/data",
    )
    parser.add_argument(
        "--csv_path",
        help="path to save evaluation results",
        default="results.csv",
    )
    parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
    parser.add_argument("--prefetch", type=int, default=4, help="pre-fetching threads.")

    args = parser.parse_args()

    # Load config file
    config = load_yaml(args.config)

    main(args, config)
