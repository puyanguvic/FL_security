from __future__ import annotations

import argparse
import copy
import logging

import torch
import torch.nn as nn
import torch.optim as optim

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import ParamsType
from nvflare.client.tracking import SummaryWriter
from nvflare.apis.fl_constant import FLMetaKey

from flbench.models.cnn import ModerateCNN
from flbench.tasks.vision.cifar10.dataset import create_data_loaders, create_datasets
from flbench.utils.torch_utils import compute_model_diff, evaluate, get_lr_values, set_seed

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

logging.getLogger("nvflare").setLevel(logging.ERROR)


def main(args):
    set_seed(args.seed)

    model = ModerateCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = None

    flare.init()
    site_name = flare.get_site_name()
    print(f"Create datasets for site {site_name}")

    train_dataset, valid_dataset = create_datasets(site_name, train_idx_root=args.train_idx_root, seed=args.seed)
    train_loader, valid_loader = create_data_loaders(
        train_dataset, valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    summary_writer = SummaryWriter()

    while flare.is_running():
        input_model = flare.receive()
        print(f"\n[Current Round={input_model.current_round}, Site={flare.get_site_name()}]\n")

        if scheduler is None and not args.no_lr_scheduler:
            total_rounds = input_model.total_rounds
            eta_min = args.lr * args.cosine_lr_eta_min_factor
            T_max = total_rounds * args.aggregation_epochs
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
            print(f"{site_name}: CosineAnnealingLR: initial_lr={args.lr}, eta_min={eta_min}, T_max={T_max}")

        model.load_state_dict(input_model.params, strict=True)

        global_model = copy.deepcopy(model)
        for p in global_model.parameters():
            p.requires_grad = False

        model.to(DEVICE)
        global_model.to(DEVICE)

        val_acc_global_model = evaluate(global_model, valid_loader)
        print(f"Global model accuracy on validation set: {100 * val_acc_global_model:.2f}%")
        summary_writer.add_scalar("val_acc_global_model", val_acc_global_model, input_model.current_round)

        steps = args.aggregation_epochs * len(train_loader)

        for epoch in range(args.aggregation_epochs):
            model.train()
            running_loss = 0.0
            for data in train_loader:
                inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / max(1, len(train_loader))
            global_epoch = input_model.current_round * args.aggregation_epochs + epoch
            curr_lr = get_lr_values(optimizer)[0]

            summary_writer.add_scalar("global_round", input_model.current_round, global_epoch)
            summary_writer.add_scalar("global_epoch", global_epoch, global_epoch)
            summary_writer.add_scalar("train_loss", avg_loss, global_epoch)
            summary_writer.add_scalar("learning_rate", curr_lr, global_epoch)

            print(
                f"{site_name}: Epoch [{epoch+1}/{args.aggregation_epochs}] "
                f"- Loss: {avg_loss:.4f} - LR: {curr_lr:.6f}"
            )

            if args.evaluate_local:
                val_acc_local_model = evaluate(model, valid_loader)
                print(f"Local model accuracy on validation set: {100 * val_acc_local_model:.2f}%")
                summary_writer.add_scalar("val_acc_local_model", val_acc_local_model, global_epoch)

            if scheduler is not None:
                scheduler.step()

        print(f"Finished training for current round {input_model.current_round}")

        model_diff, diff_norm = compute_model_diff(model, global_model)
        summary_writer.add_scalar("diff_norm", float(diff_norm), input_model.current_round)

        num_examples = len(train_loader.dataset)

        meta = {
            # NVFlare's default FedAvg recipe uses NUM_STEPS_CURRENT_ROUND as the aggregation weight.
            # For FedAvg this should reflect the amount of local training data.
            FLMetaKey.NUM_STEPS_CURRENT_ROUND: float(num_examples),
            "local_steps_current_round": int(steps),
        }

        output_model = flare.FLModel(
            params=model_diff,
            params_type=ParamsType.DIFF,
            metrics={"accuracy": val_acc_global_model},
            meta=meta,
        )

        flare.send(output_model)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_idx_root", type=str, default="/tmp/flbench_splits", help="Split index root dir")
    p.add_argument("--aggregation_epochs", type=int, default=4, help="Local epochs per FL round")
    p.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    p.add_argument("--no_lr_scheduler", action="store_true", help="Disable LR scheduler")
    p.add_argument("--cosine_lr_eta_min_factor", type=float, default=0.01, help="eta_min factor for cosine LR")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    p.add_argument("--evaluate_local", action="store_true", help="Evaluate local model each epoch")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    args = p.parse_args()

    main(args)
