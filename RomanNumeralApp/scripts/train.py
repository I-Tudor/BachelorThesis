import os
import sys
import yaml
import torch
import random
import logging
import numpy as np
import pandas as pd
import torch.optim as optim
import wandb

from tqdm import tqdm
from os.path import join as ospj
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from source.loss import MultiTaskLoss
from source.data import TheoryTabDataset
from source.metrics import MultiTaskMetrics
from source.constants import TASKS, LABEL_SIZES, LABEL_DOMAINS
from source.models import Frog, BiGRU, AudioAugmentedNet, RNATransformer


# Reproducibility

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
logging.info(f'Using device: {device}')


# Train / validate loops

def train(model, train_loader, criterion, optimizer):
    train_loss = 0.0
    train_metrics = MultiTaskMetrics('train')

    model.train()
    for batch_idx, (features, labels) in (
        pbar := tqdm(enumerate(train_loader), total=len(train_loader), unit='batch')
    ):
        labels   = labels.to(device)
        features = [feature.to(device) for feature in features]

        outputs = model(features)
        loss    = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping prevents occasional large gradients from
        # destabilising the uncertainty loss or the key head's wider input.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss += loss.item()
        train_metrics.update(outputs, labels)

        pbar.set_description(
            f'Train loss: {train_loss/(batch_idx+1):.4f}'
        )

    train_loss /= len(train_loader)
    return train_loss, train_metrics.compute()


def validate(model, valid_loader, criterion):
    valid_loss = 0.0
    valid_metrics = MultiTaskMetrics('valid')

    model.eval()
    with torch.no_grad():
        for batch_idx, (features, labels) in (
            pbar := tqdm(enumerate(valid_loader), total=len(valid_loader), unit='batch')
        ):
            labels   = labels.to(device)
            features = [feature.to(device) for feature in features]

            outputs = model(features)
            loss    = criterion(outputs, labels)

            valid_loss += loss.item()
            valid_metrics.update(outputs, labels)

            pbar.set_description(
                f'Val loss: {valid_loss/(batch_idx+1):.4f}'
            )

    valid_loss /= len(valid_loader)
    return valid_loss, valid_metrics.compute()


# Main training loop

def main(model, train_loader, valid_loader, config):
    wandb.init(
        project="parc-rna",
        name=config.get("run_name", None),
        config=config,
    )
    wandb.watch(model, log="gradients", log_freq=100)

    criterion = MultiTaskLoss(
        TASKS,
        device=device,
        label_smoothing=config.get('label_smoothing', 0.0),
        use_equiv_loss=config.get('use_equiv_loss', False),
        add_chord_change_head=config.get('add_chord_change_head', False),
        # Key criterion
        use_tonal_key_loss=config.get('use_tonal_key_loss', True),
        tonal_key_alpha=config.get('tonal_key_alpha', 0.15),
        # Boundary-upweighted loss
        # Frames at chord transitions receive `boundary_weight` × the normal
        # CE loss weight, encouraging the model to commit decisively at
        # boundaries rather than smearing predictions across them.
        # boundary_margin controls how many extra frames on each side are also
        # upweighted (at 46ms/frame, margin=1 → ±46ms around each boundary).
        boundary_weight=config.get('boundary_weight', 1.0),
        boundary_margin=config.get('boundary_margin', 1),
        # Within-segment consistency loss
        # Penalises prediction variance between adjacent same-chord frames.
        # Stabilises flat chord regions and reduces within-chord glitches.
        # Added as a scaled additive term after the averaged CE loss, so
        # consistency_weight=0.05 contributes ~5% of the main loss.
        consistency_weight=config.get('consistency_weight', 0.0),
        consistency_tasks=config.get('consistency_tasks', None),
    )

    # Split optimiser: key head gets lower LR + higher weight decay
    optimizer_cfg = config['optimizer']

    if hasattr(model, 'key_head_parameters'):
        key_lr     = optimizer_cfg.get('lr', 5e-4) * 0.4
        key_wd     = optimizer_cfg.get('weight_decay', 0.01) * 5
        optimizer  = optim.AdamW([
            {
                "params"      : model.non_key_parameters(),
                "lr"          : optimizer_cfg.get('lr', 5e-4),
                "weight_decay": optimizer_cfg.get('weight_decay', 0.01),
            },
            {
                "params"      : model.key_head_parameters(),
                "lr"          : key_lr,
                "weight_decay": key_wd,
            },
        ])
        logging.info(
            f"Split optimiser: base LR={optimizer_cfg['lr']:.0e}, "
            f"key-head LR={key_lr:.0e}, key-head WD={key_wd:.4f}"
        )
    else:
        optimizer = optim.AdamW(model.parameters(), **optimizer_cfg)

    # Learning rate schedule: linear warmup -> cosine decay
    num_epochs    = config['num_epochs']
    warmup_epochs = config.get('warmup_epochs', 10)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(num_epochs - warmup_epochs, 1),
        eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    df_results       = pd.DataFrame()
    best_rn_conv_acc = -torch.inf

    for epoch in range(num_epochs):
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}]")
        train_loss, train_metrics = train(model, train_loader, criterion, optimizer)
        valid_loss, valid_metrics = validate(model, valid_loader, criterion)

        # Unconditional scheduler step (fixes the frozen-LR bug from the
        # original codebase where step() was inside the best-model block).
        scheduler.step()

        current_lrs = [pg['lr'] for pg in optimizer.param_groups]
        logging.info(f"  LR: {current_lrs}")

        metrics = {
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            **train_metrics,
            **valid_metrics,
        }
        wandb.log(metrics, step=epoch)

        df_results = pd.concat([df_results, pd.DataFrame(metrics, index=[epoch])])
        df_results.to_csv(ospj(config['output_dir'], 'metrics.csv'), index=False)

        if valid_metrics['valid_rn_conv_acc'] > best_rn_conv_acc:
            best_rn_conv_acc = valid_metrics['valid_rn_conv_acc']
            logging.info(
                f'*** New best validation rn_conv accuracy: {best_rn_conv_acc:.4f} ***'
            )
            checkpoint = {
                'epoch'    : epoch,
                'model'    : model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, ospj(config['output_dir'], 'best_model.ckpt'))

    wandb.finish()


def cli():
    parser = ArgumentParser()
    parser.add_argument(
        '--yaml', type=str, required=True,
        help='Path to the configuration YAML file'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = cli()
    with open(args.yaml, 'r') as f:
        config = yaml.safe_load(f)

    if config['model'] == 'NaiveBaseline':
        logging.info('NaiveBaseline model does not require training. Exiting.')
        sys.exit(0)

    logging.info(f"Output directory set to: {config['output_dir']}")
    os.makedirs(config['output_dir'], exist_ok=True)

    label_step = 4 if config['model'] == 'Frog' else 1

    train_set = TheoryTabDataset(
        'train',
        tasks=TASKS,
        label_step=label_step,
        augment=config.get('augment', False),
        **config['data']
    )

    valid_set = TheoryTabDataset(
        'valid',
        tasks=TASKS,
        label_step=label_step,
        **config['data']
    )

    train_loader = DataLoader(train_set, shuffle=True,  **config['dataloader'])
    valid_loader = DataLoader(valid_set, shuffle=False, **config['dataloader'])

    logging.info('Size of each dataloader:')
    logging.info(f'  - Train dataloader: {len(train_loader)}')
    logging.info(f'  - Validation dataloader: {len(valid_loader)}\n')

    in_channels = (84,) if config['data']['use_semitone_spectrum'] else (12, 12)

    model_factory = {
        'Frog'             : Frog(in_channels, LABEL_SIZES).to(device),
        'BiGRU'            : BiGRU(in_channels, LABEL_SIZES).to(device),
        'AudioAugmentedNet': AudioAugmentedNet(in_channels, LABEL_SIZES).to(device),
        'RNATransformer'   : RNATransformer(
            in_channels, LABEL_SIZES,
            **config.get('model_kwargs', {})
        ).to(device),
    }

    model = model_factory.get(config['model'])
    if model is None:
        raise ValueError(f"No pattern found for model {config['model']}")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Number of model parameters: {num_params:_}\n')

    main(model, train_loader, valid_loader, config)

    train_set.close()
    valid_set.close()

    logging.info('Job finished!')