# =============================================================================
# Stage4: siteA -> siteB -> siteC -> siteD  Incremental Training
# Model: Baseline_DLRA0914  (RAP with Low-Rank Adapter)
# + Anchor-KD: 用当前数据作探针，锚定历史任务路径的函数响应
# =============================================================================
import copy
import os
import sys
import time
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# =============================================================================
# Project root
# =============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# =============================================================================
# Project imports
# =============================================================================
from dataset import OPTIC_dataset
from iouEval import ioUDiceEval
from models.Baseline_DLRA0914 import (
    Net as Net_RAP,
    non_bottleneck_1d_RAP,
)
from utils.seed import *
from utils.checkpoint import *
from transform import *
from utils.metrics import *

from models.MEEI import _select_and_inherit_by_entropy


# =============================================================================
# 0. 常量
# =============================================================================
DOMAIN_MODULES = [
    'adapters_1', 'adapters_2',
    'bns_1', 'bns_2',
    'bn_ini', 'bn_up',
]

SITE_DIRS = {
    'siteA': "REFUGE/",
    'siteB': "RIM_ONE_r3/",
    'siteC': "Drishti_GS/",
    'siteD': "REFUGE_Valid/",
}


# =============================================================================
# 1. 参数类别判断
# =============================================================================
def is_shared(name):
    domain_specific_keywords = DOMAIN_MODULES + ['output_conv']
    if any(kw in name for kw in domain_specific_keywords):
        return False
    return ('encoder' in name) or ('decoder' in name)


def is_DS_curr(name, current_task):
    if f"decoder.output_conv.{current_task}." in name:
        return True
    for module_name in DOMAIN_MODULES:
        if f"{module_name}.{current_task}." in name:
            return True
    return False


def is_DS_history(name, current_task):
    for keyword in DOMAIN_MODULES:
        if keyword in name:
            if f"{keyword}.{current_task}." not in name:
                return True
    if 'decoder.output_conv' in name:
        if f"decoder.output_conv.{current_task}." not in name:
            return True
    return False


# =============================================================================
# 2. Anchor-KD：用当前数据作探针，锚定历史路径的函数值
# =============================================================================
def anchor_kd_loss(model, model_anchor, inputs, current_task, T=4.0):
    """
    对每个历史任务 t ∈ [0, current_task)：
        teacher = model_anchor(inputs, t)     # 加载时的历史路径（冻结）
        student = model(inputs, t)            # 训练中的历史路径（主干可导）
    两者在输入 inputs 上应保持一致。

    注意：
      - 强制 model.eval() 前向，保护历史 BN 的 running stats 不被污染
      - 只对 logits 做软化 KL，不做任何参数约束
    """
    if current_task == 0:
        return torch.tensor(0.0, device=inputs.device)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            teacher_outs = [
                model_anchor(inputs, t) for t in range(current_task)
            ]

        total = 0.0
        for t in range(current_task):
            student_out = model(inputs, t)
            total = total + F.kl_div(
                F.log_softmax(student_out / T, dim=1),
                F.softmax(teacher_outs[t] / T, dim=1),
                reduction='batchmean',
            ) * (T * T)
        total = total / current_task
    finally:
        if was_training:
            model.train()
    return total


def build_model_anchor(model):
    """
    深拷贝当前 model 作为锚，并完全冻结 + eval。
    """
    anchor = copy.deepcopy(model)
    anchor.eval()
    for p in anchor.parameters():
        p.requires_grad = False
    return anchor


# =============================================================================
# 3. 数据集构建
# =============================================================================
def _build_datasets(args, co_transform, co_transform_val):
    site_dirs = {k: os.path.join(args.rootdir, v) for k, v in SITE_DIRS.items()}

    val_datasets = {
        site: OPTIC_dataset(path, co_transform_val, 'val')
        for site, path in site_dirs.items()
    }

    name = args.dataset_new
    if name not in site_dirs:
        raise ValueError(f"Unknown dataset: {name}")

    print(f'taking {name}')
    train_dataset = OPTIC_dataset(site_dirs[name], co_transform, 'train')
    return train_dataset, val_datasets


# =============================================================================
# 4. Training
# =============================================================================
def train(args, model, model_old):
    best_acc = 0.0

    # ---- TensorBoard ----
    tf_dir = 'runs_{}_{}_{}_{}_{}_step{}_r{}'.format(
        args.dataset_new, args.model, args.num_epochs,
        args.batch_size, args.model_name_suffix,
        args.nb_tasks, args.adapter_rank
    )
    writer = SummaryWriter('Adaptations/' + tf_dir)
    data_name = args.dataset_new

    # ---- Transform ----
    co_transform = MyCoTransform(augment=True, height=args.height, width=args.width)
    co_transform_val = MyCoTransform(augment=False, height=args.height, width=args.width)

    # ---- Dataset / DataLoader ----
    dataset_train, val_datasets = _build_datasets(args, co_transform, co_transform_val)

    worker_init_fn = worker_init_fn_factory(args.seed)

    loader = DataLoader(
        dataset_train,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        worker_init_fn=worker_init_fn,
    )

    # 所有已见域验证 loader
    loader_val = {
        d: DataLoader(val_datasets[d], num_workers=1,
                      batch_size=args.batch_size, shuffle=False)
        for d in args.datasets
    }

    # ---- Loss ----
    weight = None
    if args.cuda and weight is not None:
        weight = weight.cuda()

    criterion = CrossEntropyLoss2d(weight)
    criterion_val = {d: CrossEntropyLoss2d(weight) for d in args.datasets}

    print(type(criterion))
    print('global current_task: ', args.current_task)


    # =========================================================
    # ★ 新增：熵评估 + 域参数继承
    # =========================================================
    best_source_task, entropies = _select_and_inherit_by_entropy(
        model, loader, args,
        max_batches=args.entropy_max_batches,   # 见参数部分
        device='cuda' if args.cuda else 'cpu',
    )

    # 保存一下，方便写日志
    args.__dict__['entropy_source_task'] = best_source_task
    args.__dict__['entropy_values'] = {int(k): float(v) for k, v in entropies.items()}


    # ---- Old model: 完全冻结 ----
    for _, param in model_old.named_parameters():
        param.requires_grad = False
    model_old.eval()

    # ---- Current model: 默认全冻结，按类别解冻 ----
    for name, param in model.named_parameters():
        if is_shared(name):
            param.requires_grad = True
        elif is_DS_curr(name, args.current_task):
            param.requires_grad = True
        elif is_DS_history(name, args.current_task):
            param.requires_grad = False
        else:
            param.requires_grad = False

    # =========================================================================
    # ★ Anchor-KD：构建锚模型
    # =========================================================================
    use_anchor = (args.current_task > 0) and (args.lambda_anchor > 0)
    if use_anchor:
        print("\n=========== ANCHOR-KD SETUP ===========")
        print(f"  lambda_anchor = {args.lambda_anchor}")
        print(f"  anchor_T      = {args.anchor_T}")
        print(f"  anchor_every  = {args.anchor_every}")
        print(f"  anchoring     : tasks {list(range(args.current_task))}")
        model_anchor = build_model_anchor(model)
        print("  model_anchor built (frozen, eval).")
        print("=======================================\n")
    else:
        model_anchor = None
        print("[Anchor-KD] disabled (current_task == 0 or lambda_anchor == 0)")

    # ---- Save paths ----
    savedir = args.savedir
    os.makedirs(savedir, exist_ok=True)

    automated_log_path = os.path.join(savedir, "automated_log.txt")
    modeltxtpath = os.path.join(savedir, "model.txt")

    # ---- Log header ----
    if not os.path.exists(automated_log_path):
        with open(automated_log_path, "w") as f:
            f.write(
                "Epoch\tTrain-loss\tCE-loss\tAnchor-loss\tTest-loss\tTrain-dice\t"
                + "\t".join(f"dice-{d}" for d in args.datasets)
                + "\tlearningRate\n"
            )

    with open(modeltxtpath, "w") as f:
        f.write(str(model))

    # ---- Trainable summary ----
    print("\n================ TRAINABLE PARAMETERS ================\n")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("TRAIN :", name)

    print("\n================ HISTORICAL DOMAIN ================\n")
    for name, _ in model.named_parameters():
        if is_DS_history(name, args.current_task):
            print("FROZEN:", name)
    print("\n=====================================================\n")

    # ---- Optimizer ----
    shared_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and is_shared(name)
    ]
    current_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and is_DS_curr(name, args.current_task)
    ]

    print("\nShared trainable parameters:", len(shared_params))
    print("Current trainable parameters:", len(current_params))

    optimizer = Adam(
        [
            {"params": shared_params, "lr": 5e-6},
            {"params": current_params, "lr": 5e-4},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )

    # ---- Scheduler ----
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: pow((1 - ((epoch - 1) / args.num_epochs)), 0.9),
    )

    # =========================================================================
    # Training loop
    # =========================================================================
    for epoch in range(1, args.num_epochs + 1):
        NUM_CLASSES = args.num_classes[args.current_task]

        print("\n----- TRAINING - EPOCH ---", epoch, "-----")
        scheduler.step(epoch)

        epoch_loss, time_train, e_ce_loss, e_anchor_loss = [], [], [], []
        doDiceTrain = args.diceTrain
        if doDiceTrain:
            diceEvalTrain = ioUDiceEval(NUM_CLASSES)

        # Anchor 权重 warmup（可选）
        if use_anchor and args.anchor_warmup_epochs > 0 and epoch <= args.anchor_warmup_epochs:
            cur_lambda_anchor = args.lambda_anchor * (epoch / args.anchor_warmup_epochs)
        else:
            cur_lambda_anchor = args.lambda_anchor

        print("\nCurrent learning rates:")
        for idx, param_group in enumerate(optimizer.param_groups):
            print(f"  Group {idx}: {param_group['lr']:.10f}")
        if use_anchor:
            print(f"Current lambda_anchor: {cur_lambda_anchor:.6f}")
        usedLr = float(optimizer.param_groups[0]['lr'])

        model.train()

        # ---- Training batches ----
        for step, (images, labels) in enumerate(loader):
            start_time = time.time()

            if args.cuda:
                inputs = images.cuda(non_blocking=True)
                targets = labels.cuda(non_blocking=True)
            else:
                inputs, targets = images, labels

            outputs = model(inputs, args.current_task)
            ce_loss = criterion(outputs, targets[:, 0])

            # =============================================================
            # ★ Anchor-KD
            # =============================================================
            if use_anchor and (step % args.anchor_every == 0):
                anchor_loss = anchor_kd_loss(
                    model, model_anchor, inputs,
                    current_task=args.current_task,
                    T=args.anchor_T,
                )
                total_loss = ce_loss + cur_lambda_anchor * anchor_loss
                anchor_loss_val = anchor_loss.item()
            else:
                total_loss = ce_loss
                anchor_loss_val = 0.0

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss.append(total_loss.item())
            e_ce_loss.append(ce_loss.item())
            e_anchor_loss.append(anchor_loss_val)
            time_train.append(time.time() - start_time)

            if doDiceTrain:
                diceEvalTrain.addBatch(
                    outputs.max(1)[1].unsqueeze(1).data,
                    targets.data,
                )

            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                avg_time = sum(time_train) / len(time_train) / args.batch_size
                kd_str = (
                    f" anchor: {sum(e_anchor_loss)/len(e_anchor_loss):.4f}"
                    if use_anchor else ""
                )
                print(
                    f'loss: {average:0.4f}{kd_str} '
                    f'(epoch: {epoch}, step: {step}) '
                    f'// Avg time/img: {avg_time:.4f} s'
                )

        # ---- Epoch stats ----
        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        average_epoch_loss_ce = sum(e_ce_loss) / len(e_ce_loss)
        average_epoch_loss_anchor = (
            sum(e_anchor_loss) / len(e_anchor_loss) if e_anchor_loss else 0.0
        )
        print('epoch took: ', sum(time_train))

        diceTrain = 0.0
        if doDiceTrain:
            _, iou_classes = diceEvalTrain.getIoU()
            diceTrain, dice_classes = diceEvalTrain.getDice()

        # ---- Validation on all seen domains ----
        print("\n----- VALIDATING - EPOCH", epoch, "-----")

        average_loss_val = {d: 0.0 for d in args.datasets}
        val_iou = {d: 0.0 for d in args.datasets}
        val_dice = {d: 0.0 for d in args.datasets}

        for ind, d in enumerate(args.datasets):
            print(f'validating: {d} (task {ind})')
            average_loss_val[d], val_iou[d], val_dice[d] = eval(
                model, loader_val[d], criterion_val[d],
                ind, args.num_classes[ind], epoch, args.batch_size,
            )

        current_acc = torch.stack(list(val_dice.values())).mean().item()
        loss_val = sum(average_loss_val.values()) / len(average_loss_val)

        print("\n================ VALIDATION RESULT ================\n")
        for d in args.datasets:
            print(f"  {d}: Dice={val_dice[d]:.4f}, IoU={val_iou[d]:.4f}, Loss={average_loss_val[d]:.4f}")
        print(f"Average Dice = {current_acc:.4f}")
        print("\n====================================================\n")

        # ---- TensorBoard ----
        info = {
            'total_train_loss': average_epoch_loss_train,
            'ce_loss_train': average_epoch_loss_ce,
            'anchor_loss_train': average_epoch_loss_anchor,
        }
        for d in args.datasets:
            info[f'val_dice_{d}'] = val_dice[d]
            info[f'val_iou_{d}'] = val_iou[d]
            info[f'val_loss_{d}'] = average_loss_val[d]
        info['average_val_dice'] = current_acc

        for tag, value in info.items():
            writer.add_scalar(tag, value, epoch)

        # ---- Best checkpoint ----
        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)

        suffix = '{}_{}_{}_{}_{}_step{}_r{}'.format(
            args.dataset_new, args.model, args.num_epochs,
            args.batch_size, args.model_name_suffix,
            args.nb_tasks, args.adapter_rank
        )

        filenameCheckpoint = os.path.join(savedir, f'checkpoint_{suffix}.pth.tar')
        filenameBest = os.path.join(savedir, f'model_best_{suffix}.pth.tar')

        save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': str(model),
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            },
            is_best,
            filenameCheckpoint,
            filenameBest,
        )

        if is_best:
            with open(os.path.join(savedir, "best.txt"), "w") as f:
                # 记录域参数继承自哪个域
                f.write(f"current_task = {args.current_task}\n")
                f.write(f"source_task  = {best_source_task}\n")
                for k, v in entropies.items():
                    f.write(f"  history task {k}: avg entropy = {v:.6f}\n")

                f.write(f"Best epoch = {epoch}\n")
                f.write(f"Current domain = {args.dataset_new}\n")
                for d in args.datasets:
                    f.write(f"{d} Dice = {val_dice[d]:.6f}, IoU = {val_iou[d]:.6f}\n")
                f.write(f"Average Dice = {current_acc:.6f}\n")

            print(
                f">>> NEW BEST: Epoch={epoch}, "
                f"Average Dice={current_acc:.4f}"
            )

        # ---- Automated log ----
        with open(automated_log_path, "a") as f:
            f.write(
                f"\n{epoch}\t"
                f"{average_epoch_loss_train:.6f}\t"
                f"{average_epoch_loss_ce:.6f}\t"
                f"{average_epoch_loss_anchor:.6f}\t"
                f"{loss_val:.6f}\t"
                f"{diceTrain:.6f}\t"
                + "\t".join(f"{val_dice[d]:.6f}" for d in args.datasets)
                + f"\t{usedLr:.10f}"
            )

    writer.close()
    return model


# =============================================================================
# 5. Validation
# =============================================================================
def eval(model, dataset_loader, criterion, task, num_classes, epoch, batch_size):
    model.eval()

    epoch_loss_val, time_val = [], []
    num_cls = num_classes

    print('number of classes in current task: ', num_cls)
    print('validating task: ', task)

    ioUDiceEvalVal = ioUDiceEval(num_cls)

    with torch.no_grad():
        for step, (images, labels) in enumerate(dataset_loader):
            start_time = time.time()

            if torch.cuda.is_available():
                inputs = images.cuda(non_blocking=True)
                targets = labels.cuda(non_blocking=True)
            else:
                inputs, targets = images, labels

            outputs = model(inputs, task)
            loss = criterion(outputs, targets[:, 0])

            epoch_loss_val.append(loss.item())
            time_val.append(time.time() - start_time)

            ioUDiceEvalVal.addBatch(
                outputs.max(1)[1].unsqueeze(1).data,
                targets.data,
            )

            if step > 0 and step % 50 == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
                avg_time = sum(time_val) / len(time_val) / batch_size
                print(
                    f'VAL loss: {average:0.4f} '
                    f'(epoch: {epoch}, step: {step}) '
                    f'// Avg time/img: {avg_time:.4f} s'
                )

    average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
    iouVal, _ = ioUDiceEvalVal.getIoU()
    diceVal, _ = ioUDiceEvalVal.getDice()

    return average_epoch_loss_val, iouVal, diceVal


# =============================================================================
# 6. Checkpoint loading helpers
# =============================================================================
def _copy_domain_params(checkpoint_state, current_state, new_dict_load,
                        previous_task, current_task):
    copied_count = 0

    for k, v in checkpoint_state.items():
        if 'decoder.output_conv' in k:
            continue

        for module_name in DOMAIN_MODULES:
            pattern = rf'\.{module_name}\.{previous_task}\.'
            if not re.search(pattern, k):
                continue

            nkey = re.sub(
                rf'\.{module_name}\.{previous_task}\.',
                f'.{module_name}.{current_task}.',
                k,
            )
            if nkey in current_state and current_state[nkey].shape == v.shape:
                new_dict_load[nkey] = v
                copied_count += 1
            break

    return copied_count


def _load_previous_stage_checkpoint(args, model, model_old):
    print("\n================================================")
    print("Loading Stage3 checkpoint:")
    print(args.state)
    print("================================================\n")

    if not os.path.exists(args.state):
        raise FileNotFoundError(f"Checkpoint not found:\n{args.state}")

    saved_model = torch.load(
        args.state,
        map_location='cuda' if args.cuda else 'cpu',
    )
    checkpoint_state = saved_model['state_dict']

    # 1. old model
    missing_keys_old, unexpected_keys_old = model_old.load_state_dict(
        checkpoint_state, strict=False,
    )
    print("old model Missing keys:", missing_keys_old)
    print("old model Unexpected keys:", unexpected_keys_old)

    # 2. common params
    print("\nLoading common parameters from previous checkpoint...")
    current_state = model.state_dict()
    new_dict_load = {}
    common_count = 0

    for k, v in checkpoint_state.items():
        if k in current_state and current_state[k].shape == v.shape:
            new_dict_load[k] = v
            common_count += 1
    print("Common parameters loaded:", common_count)

    # # 3. task{previous} -> task{current} domain 参数复制
    # previous_task = args.current_task - 1
    # copied_count = _copy_domain_params(
    #     checkpoint_state, current_state, new_dict_load,
    #     previous_task, args.current_task,
    # )
    # print(f"Task{previous_task} -> Task{args.current_task} domain parameters copied:", copied_count)

    # 4. 加载到当前模型
    missing_keys, unexpected_keys = model.load_state_dict(
        new_dict_load, strict=False,
    )

    print("\n================ MODEL LOADING ================\n")
    # print(f"Initialized new domain {args.current_task} from previous domain {previous_task}")
    print("\nMissing keys:")
    for key in missing_keys:
        print("  ", key)
    print("\nUnexpected keys:")
    for key in unexpected_keys:
        print("  ", key)
    print("\n================================================\n")

    return None, common_count, 0, missing_keys, unexpected_keys


# =============================================================================
# 7. Main
# =============================================================================
def main(args):
    setup_seed(args.seed)

    print('\ndataset_new: ', args.dataset_new)
    print('datasets   : ', args.datasets)
    print('current task: ', args.current_task)

    savedir = args.savedir
    os.makedirs(savedir, exist_ok=True)

    with open(os.path.join(savedir, 'opts.txt'), "w") as f:
        f.write(str(args))

    # ---- Stage4 sanity check ----
    assert args.current_task == 3, \
        "This script is Stage4 A -> B -> C -> D, therefore current_task must be 3."
    assert args.dataset_new == 'siteD', \
        "Stage4 current domain must be siteD."
    assert args.datasets == ['siteA', 'siteB', 'siteC', 'siteD'], \
        "Stage4 datasets should be [siteA, siteB, siteC, siteD]."
    assert args.num_classes == [3, 3, 3, 3], \
        "Stage4 expects num_classes=[3,3,3,3]."
    assert args.num_classes_old == [3, 3, 3], \
        "Stage4 expects num_classes_old=[3,3,3]."
    assert args.nb_tasks == 4, \
        "Stage4 expects nb_tasks=4."

    # ---- Build models ----
    print("\n================ MODEL CREATION ================\n")
    print("num_classes     =", args.num_classes)
    print("num_classes_old =", args.num_classes_old)
    print("nb_tasks        =", args.nb_tasks)
    print("current_task    =", args.current_task)

    model = Net_RAP(args.num_classes, args.nb_tasks, adapter_rank=args.adapter_rank)
    model_old = Net_RAP(args.num_classes_old, args.nb_tasks - 1, adapter_rank=args.adapter_rank)

    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
        model_old = torch.nn.DataParallel(model_old).cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    # ---- Load Stage3 checkpoint ----
    if not args.state:
        raise ValueError("Stage4 requires --state to load the Stage3/siteABC checkpoint.")

    _load_previous_stage_checkpoint(args, model, model_old)

    print("Stage4 initialization:")
    print("  Previous domains : task0 / siteA, task1 / siteB, task2 / siteC")
    print("  Current domain   : task3 / siteD")
    print("  Shared           : loaded from siteABC checkpoint")
    print("  Domain-0,1,2     : loaded from siteABC checkpoint (frozen)")
    # print("  Domain-3         : copied from Domain-2")
    print("  Fusion           : shared + current（无历史分支）")
    print("  Anchor-KD        : 用当前 batch 作探针，锚定历史路径\n")

    # ---- Train ----
    print('Loaded model from checkpoint provided.')
    start_time = time.time()
    model = train(args, model, model_old)
    end_time = time.time()

    total_minutes = (end_time - start_time) / 60.0
    print("\n========== TRAINING FINISHED ===========")
    print(f"Total training time: {total_minutes:.2f} minutes")


# =============================================================================
# 8. Arguments
# =============================================================================
def parse_args():
    parser = ArgumentParser()

    # CUDA / Model
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--model', default='Baseline_DLRA0914')

    # Stage4: siteA -> siteB -> siteC -> siteD
    parser.add_argument('--dataset_new', default='siteD')
    parser.add_argument('--datasets', nargs="+", required=False,
                        default=['siteA', 'siteB', 'siteC', 'siteD'])

    # Classes
    parser.add_argument('--num_classes', type=int, nargs='+', default=[3, 3, 3, 3])
    parser.add_argument('--num_classes_old', type=int, nargs='+', default=[3, 3, 3])

    # Incremental task
    parser.add_argument('--nb_tasks', type=int, default=4)
    parser.add_argument('--current_task', type=int, default=3)

    # Stage3 checkpoint（siteABC）
    parser.add_argument(
        '--state',
        default=(
            '/home/jnu/fcl_exp/class1_CL/ECANet_V5_EMA/save/fundus_results/Baseline_DLRA0914/siteABC_384/'
            'model_best_siteC_Baseline_DLRA0914_100_8_IL_step3_r16.pth.tar'
        ),
    )

    # Data
    parser.add_argument('--rootdir', default='/home/jnu/fcl_data/MDIL2025/Fundus/')
    parser.add_argument('--height', type=int, default=384)
    parser.add_argument('--width', type=int, default=384)

    # Training
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--steps_loss', type=int, default=50)
    parser.add_argument('--steps_plot', type=int, default=50)
    parser.add_argument('--adapter_rank', type=int, default=16)

    # ---- Anchor-KD ----
    parser.add_argument('--lambda_anchor', type=float, default=1.0,
                        help='Anchor-KD 损失权重')
    parser.add_argument('--anchor_T', type=float, default=4.0,
                        help='Anchor-KD 蒸馏温度')
    parser.add_argument('--anchor_every', type=int, default=1,
                        help='每 N 个 step 做一次 Anchor-KD（节省算力）')
    parser.add_argument('--anchor_warmup_epochs', type=int, default=5,
                        help='λ_anchor 从 0 升到目标值的 epoch 数；0 表示不 warmup')

    # Save
    parser.add_argument(
        '--savedir',
        default='../save/fundus_results/Baseline_DLRA_MEEI0916/siteABCD_384_distil/',
    )
    parser.add_argument('--epochs_save', type=int, default=0)

    # Legacy arguments (Stage4 实际不使用)
    parser.add_argument('--lambdac', type=float, default=0.0)
    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--pretrainedEncoder')
    parser.add_argument('--diceTrain', action='store_true', default=False)
    parser.add_argument('--diceVal', action='store_true', default=True)
    parser.add_argument('--resume', action='store_true')

    # Misc
    parser.add_argument('--model_name_suffix', default='IL')
    parser.add_argument('--seed', type=int, default=2025)

    parser.add_argument('--entropy_max_batches', type=int, default=20,
                        help='熵评估时使用的新域 batch 数（0 或负数表示全量）')

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())