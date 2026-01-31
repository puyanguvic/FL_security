from __future__ import annotations

import argparse
import copy
import logging

import nvflare.client as flare
import torch
import torch.nn as nn
import torch.optim as optim
from nvflare.apis.fl_constant import FLMetaKey
from nvflare.app_common.abstract.fl_model import ParamsType
from nvflare.client.tracking import SummaryWriter

from flbench.algorithms.scaffold.model import ScaffoldModel
from flbench.utils.torch_utils import compute_model_diff, evaluate_with_loss, get_lr_values, set_seed

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

logging.getLogger("nvflare").setLevel(logging.ERROR)


def main(args):
    set_seed(args.seed)

    from flbench.core.registry import get_task

    task_spec = get_task(args.task)
    model = ScaffoldModel(task_spec.build_model(args.model))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = None
    local_c = None

    flare.init()
    site_name = flare.get_site_name()
    print(f"Create datasets for site {site_name}")

    train_dataset, valid_dataset = task_spec.create_datasets(
        site_name, train_idx_root=args.train_idx_root, seed=args.seed, data_root=args.data_root
    )
    train_loader, valid_loader = task_spec.create_data_loaders(
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

        try:
            model.load_state_dict(input_model.params, strict=True)
        except RuntimeError as e:
            msg = str(e)
            if "size mismatch" in msg:
                raise RuntimeError(
                    "Global model shape mismatch between server and client.\n"
                    f"- client: task={args.task} model={args.model}\n"
                    "This usually happens when an old NVFLARE simulation workspace is reused (stale checkpoint) "
                    "or when server/client tasks differ.\n"
                    "Fix: rerun with a fresh job name or without `--resume` (default clears the sim workspace)."
                ) from e
            raise

        global_model = copy.deepcopy(model)
        for p in global_model.parameters():
            p.requires_grad = False

        model.to(DEVICE)
        global_model.to(DEVICE)

        global_c = model.get_c()
        if local_c is None:
            local_c = {name: torch.zeros_like(param.data) for name, param in model.model.named_parameters()}

        val_loss_global_model, val_acc_global_model = evaluate_with_loss(global_model, valid_loader, criterion)
        print(
            "Global model validation - "
            f"acc: {100 * val_acc_global_model:.2f}% "
            f"loss: {val_loss_global_model:.4f}"
        )
        summary_writer.add_scalar("val_acc_global_model", val_acc_global_model, input_model.current_round)
        summary_writer.add_scalar("val_loss_global_model", val_loss_global_model, input_model.current_round)

        steps = args.aggregation_epochs * len(train_loader)
        steps = max(1, steps)

        for epoch in range(args.aggregation_epochs):
            model.train()
            running_loss = 0.0
            running_correct = 0
            running_total = 0
            for data in train_loader:
                inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                # SCAFFOLD gradient correction: g <- g + c - c_i
                with torch.no_grad():
                    for name, param in model.model.named_parameters():
                        if param.grad is None:
                            continue
                        param.grad.add_(global_c[name] - local_c[name])

                optimizer.step()
                running_loss += loss.item()
                preds = outputs.argmax(dim=1)
                running_correct += (preds == labels).sum().item()
                running_total += labels.numel()

            avg_loss = running_loss / max(1, len(train_loader))
            train_acc = float(running_correct) / float(running_total) if running_total > 0 else 0.0
            global_epoch = input_model.current_round * args.aggregation_epochs + epoch
            curr_lr = get_lr_values(optimizer)[0]

            summary_writer.add_scalar("global_round", input_model.current_round, global_epoch)
            summary_writer.add_scalar("global_epoch", global_epoch, global_epoch)
            summary_writer.add_scalar("train_loss", avg_loss, global_epoch)
            summary_writer.add_scalar("train_acc", train_acc, global_epoch)
            summary_writer.add_scalar("learning_rate", curr_lr, global_epoch)

            print(
                f"{site_name}: Epoch [{epoch+1}/{args.aggregation_epochs}] "
                f"- Loss: {avg_loss:.4f} - LR: {curr_lr:.6f}"
            )

            val_loss_local_model, val_acc_local_model = evaluate_with_loss(model, valid_loader, criterion)
            summary_writer.add_scalar("val_acc_local_model", val_acc_local_model, global_epoch)
            summary_writer.add_scalar("val_loss_local_model", val_loss_local_model, global_epoch)
            if args.evaluate_local:
                print(
                    "Local model validation - "
                    f"acc: {100 * val_acc_local_model:.2f}% "
                    f"loss: {val_loss_local_model:.4f}"
                )

            if scheduler is not None:
                scheduler.step()

        print(f"Finished training for current round {input_model.current_round}")

        # Update local control variate and prepare delta for server
        global_params = dict(global_model.model.named_parameters())
        local_params = dict(model.model.named_parameters())
        delta_c = {}
        scale = 1.0 / (steps * args.lr)
        for name, param in local_params.items():
            delta = -global_c[name] + scale * (global_params[name].data - param.data)
            delta_c[name] = delta
            local_c[name] = local_c[name] + delta

        # Set model's control variate buffers to (c + delta_c)
        new_global_c = {name: global_c[name] + delta_c[name] for name in delta_c}
        model.set_c(new_global_c)

        model_diff, diff_norm = compute_model_diff(model, global_model)
        summary_writer.add_scalar("diff_norm", float(diff_norm), input_model.current_round)

        num_examples = len(train_loader.dataset)

        meta = {
            # NVFlare's default FedAvg recipe uses NUM_STEPS_CURRENT_ROUND as the aggregation weight.
            # For SCAFFOLD this should reflect the amount of local training data.
            FLMetaKey.NUM_STEPS_CURRENT_ROUND: float(num_examples),
            "local_steps_current_round": int(steps),
        }

        output_model = flare.FLModel(
            params=model_diff,
            params_type=ParamsType.DIFF,
            metrics={"accuracy": val_acc_global_model, "loss": val_loss_global_model},
            meta=meta,
        )

        flare.send(output_model)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="vision/cifar10", help="Task key (e.g., vision/cifar10).")
    p.add_argument("--model", type=str, default="cnn/moderate", help="Model key (task resolves it).")
    p.add_argument("--train_idx_root", type=str, default="/tmp/flbench_splits", help="Split index root dir")
    p.add_argument("--aggregation_epochs", type=int, default=4, help="Local epochs per FL round")
    p.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    p.add_argument("--no_lr_scheduler", action="store_true", help="Disable LR scheduler")
    p.add_argument("--cosine_lr_eta_min_factor", type=float, default=0.01, help="eta_min factor for cosine LR")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size")
    p.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    p.add_argument("--evaluate_local", action="store_true", help="Evaluate local model each epoch")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--data_root", type=str, default=None, help="Dataset root (task-dependent)")
    args = p.parse_args()

    main(args)
