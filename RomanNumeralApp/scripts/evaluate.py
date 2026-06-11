import json
import yaml
import torch
import random
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
from os.path import join as ospj
from argparse import ArgumentParser
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

from source.loss import MultiTaskLoss
from source.data import TheoryTabDataset
from source.models import Frog, BiGRU, AudioAugmentedNet, NaiveBaseline, RNATransformer
from source.constants import (
    TASKS,
    GENRES,
    LABEL_SIZES,
    COMPLEXITIES,
    LABEL_DOMAINS,
    PCSETS_FILEPATH,
    SPLITS_FILEPATH,
    LABEL_PADDING_VALUE,
)

with open(PCSETS_FILEPATH, 'r') as f:
    pcsets = json.load(f)
    get_rn_pitch_classes = np.vectorize(lambda key, rn: pcsets[key][rn])

with open(SPLITS_FILEPATH, 'r') as f:
    splits = json.load(f)
    test_ids_by_level = {
        'song': splits['song']['test'],
        'artist': splits['artist']['test'],
        'theorytab': splits['theorytab']['test']
    }

global_key_domain = np.array(LABEL_DOMAINS['global_key'])
roman_numeral_domain = np.array(LABEL_DOMAINS['roman_numeral'])

# Fixing random seeds for reproducibility
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device: {device}')


def boostrap_ci(scores, ci):
    lower = np.percentile(scores, ((1 - ci) / 2) * 100)
    upper = np.percentile(scores, (1 - (1 - ci) / 2) * 100)
    
    return np.mean(scores), (lower, upper)


def get_predictions(model, test_set):
    test_loss = 0.0
    masks = {task: [] for task in TASKS}
    y_preds = {task: [] for task in TASKS}
    y_trues = {task: [] for task in TASKS}

    criterion = MultiTaskLoss(tasks=TASKS, device=device)

    model.eval()
    with torch.no_grad():
        for i in (
            pbar := tqdm(range(len(test_set)), total=len(test_set), unit='sequence')
        ):
            features, labels = test_set[i]
            labels = torch.from_numpy(labels).unsqueeze(0).to(device)
            features = [torch.from_numpy(feature).unsqueeze(0).to(device) for feature in features]
        
            outputs = model(features)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    
            labels = labels.squeeze(0).cpu().numpy()
            outputs = {task: output.squeeze(0).cpu().numpy() for task, output in outputs.items()}

            for j, task in enumerate(outputs):
                y_trues[task].append(labels[j])
                y_preds[task].append(outputs[task].argmax(axis=-1))
                masks[task].append(labels[j] != LABEL_PADDING_VALUE)
            
            pbar.set_description(
                f'Test loss: {test_loss/(i+1):.4f}'
            )

    test_loss /= len(test_set)
    masks = {task: np.array(values) for task, values in masks.items()}
    y_preds = {task: np.array(values) for task, values in y_preds.items()}
    y_trues = {task: np.array(values) for task, values in y_trues.items()}
    
    return test_loss, y_preds, y_trues, masks


def evaluate(model, test_set, n_bootstraps=1000, ci=0.95):
    test_loss, y_preds, y_trues, masks = get_predictions(model, test_set)

    # Computing Equivalence-Aware variables
    print('Computing pcsets for labels...')
    y_true_keys = global_key_domain[y_trues['global_key']]
    y_true_rns = roman_numeral_domain[y_trues['roman_numeral']]
    y_true_pitch_classes = get_rn_pitch_classes(y_true_keys, y_true_rns)

    print('Computing pcsets for predictions...')
    y_pred_keys = global_key_domain[y_preds['global_key']]
    y_pred_rns = roman_numeral_domain[y_preds['roman_numeral']]
    y_pred_pitch_classes = get_rn_pitch_classes(y_pred_keys, y_pred_rns)
    
    scores = defaultdict(list)
    rng = np.random.RandomState(RANDOM_SEED)
    
    for _ in tqdm(range(n_bootstraps), desc='Bootstrap CI'):
        matches = {}
        idxs = rng.choice(len(test_set), size=len(test_set), replace=True)
        
        for task in TASKS:
            mask = masks[task][idxs].flatten()
            y_true = y_trues[task][idxs].flatten()
            y_pred = y_preds[task][idxs].flatten()
            
            if task == 'roman_numeral':
                rn_mask = mask
            
            matches[task] = (y_pred == y_true)
            scores[f'test_{task}_acc'].append(np.mean(y_pred[mask] == y_true[mask]))
            
            macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
                y_true[mask], y_pred[mask], average='macro', zero_division=0
            )
            
            scores[f'test_{task}_f1'].append(macro_f1)
            scores[f'test_{task}_recall'].append(macro_recall)
            scores[f'test_{task}_precision'].append(macro_precision)
        
        # Task combination accuracies
        rn_alt_matches = matches['global_key'] * matches['roman_numeral'] * matches['inversion']
        scores['test_rn_alt_acc'].append(np.mean(rn_alt_matches[rn_mask]))
        
        rn_conv_matches = matches['global_key']
        for task in ['root_scale_degree', 'tonicization', 'quality', 'inversion', 'root_pitch_class']:
            rn_conv_matches *= matches[task]
        
        degree_matches = matches['root_scale_degree'] * matches['tonicization']
        
        scores['test_degree_acc'].append(np.mean(degree_matches[rn_mask]))
        scores['test_rn_conv_acc'].append(np.mean(rn_conv_matches[rn_mask]))

        # Evaluation-Aware version
        y_true_pcs = y_true_pitch_classes[idxs].flatten()
        y_pred_pcs = y_pred_pitch_classes[idxs].flatten()
        
        mask = masks['roman_numeral'][idxs].flatten()
        scores['test_equiv_rn_acc'].append(np.mean(y_pred_pcs[mask] == y_true_pcs[mask]))

        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true_pcs[mask], y_pred_pcs[mask], average='macro', zero_division=0
        )

        scores['test_equiv_rn_macro_f1'].append(macro_f1)
        scores['test_equiv_rn_macro_recall'].append(macro_recall)
        scores['test_equiv_rn_macro_precision'].append(macro_precision)
    
    metrics = {'test_loss': test_loss}
    for task, values in scores.items():
        mean, (lower, upper) = boostrap_ci(values, ci)
        metrics[task] = mean
        metrics[f'{task}_lower_ci'] = lower
        metrics[f'{task}_upper_ci'] = upper
    
    df_metrics = pd.DataFrame(metrics, index=[0])
    return df_metrics
    

def cli():
    parser = ArgumentParser()
    parser.add_argument('--yaml', type=str, required=True, help='Path to the configuration YAML file')
    parser.add_argument('--n_boostraps', type=int, default=300, help='Number of bootstrap iterations for confidence intervals')
    parser.add_argument('--ci', type=float, default=0.95, help='Confidence interval for bootstrap')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to the model checkpoint file (optional)')

    parser.add_argument('--run_genre_eval', action='store_true', help='Run evaluation per genre')
    parser.add_argument('--run_complexity_eval', action='store_true', help='Run evaluation per complexity')

    return parser.parse_args()


if __name__ == '__main__':
    args = cli()
    with open(args.yaml, 'r') as f:
        config = yaml.safe_load(f)

    in_channels = (84,) if config['data']['use_semitone_spectrum'] else (12, 12)
    model_factory = {
        'NaiveBaseline': NaiveBaseline(device),
        'Frog': Frog(in_channels, LABEL_SIZES).to(device),
        'BiGRU': BiGRU(in_channels, LABEL_SIZES).to(device),
        'AudioAugmentedNet': AudioAugmentedNet(in_channels, LABEL_SIZES).to(device),
        'RNATransformer': RNATransformer(in_channels, LABEL_SIZES, **config.get('model_kwargs', {})).to(device)
    }

    model = model_factory.get(config['model'])
    if model is None:
        raise ValueError(f"No pattern found for model {config['model']}")
    
    if config['model'] != 'NaiveBaseline':
        checkpoint_path = args.checkpoint if args.checkpoint else ospj(config['output_dir'], 'best_model.ckpt')
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        load_message = model.load_state_dict(checkpoint['model'], strict=False)
        logging.info(f'Load state dict message: {load_message}')

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Number of model parameters: {num_params:_}\n')

    label_step = 4 if config['model'] == 'Frog' else 1
    
    test_set = TheoryTabDataset(
        'test',
        tasks=TASKS,
        label_step=label_step,
        **config['data']
    )

    logging.info('Running overall evaluation...')
    metrics = evaluate(model, test_set, n_bootstraps=args.n_boostraps, ci=args.ci)
    metrics.to_csv(ospj(config['output_dir'], 'eval_results.csv'), index=False)
    test_set.close()
    
    if args.run_genre_eval:
        logging.info('Running per-genre evaluation...')
        genre_metrics = pd.DataFrame()
        for genre in GENRES:
            test_set = TheoryTabDataset(
                'test',
                tasks=TASKS,
                label_step=label_step,
                wanted_genres=[genre],
                **config['data']
            )

            metrics = evaluate(model, test_set, n_bootstraps=args.n_boostraps, ci=args.ci)
            genre_metrics = pd.concat([genre_metrics, metrics], ignore_index=True)
            test_set.close()

        genres = pd.DataFrame({'genre': GENRES})
        genre_metrics = pd.concat([genres, genre_metrics], axis=1)
        genre_metrics.to_csv(ospj(config['output_dir'], 'genre_eval_results.csv'), index=False)
    
    if args.run_complexity_eval:
        logging.info('Running per-complexity evaluation...')
        complexity_metrics = pd.DataFrame()
        for complexity in COMPLEXITIES:
            test_set = TheoryTabDataset(
                'test',
                tasks=TASKS,
                label_step=label_step,
                wanted_complexities=[complexity],
                **config['data']
            )

            metrics = evaluate(model, test_set, n_bootstraps=args.n_boostraps, ci=args.ci)
            complexity_metrics = pd.concat([complexity_metrics, metrics], ignore_index=True)
            test_set.close()

        complexities = pd.DataFrame({'complexity': COMPLEXITIES})
        complexity_metrics = pd.concat([complexities, complexity_metrics], axis=1)
        complexity_metrics.to_csv(ospj(config['output_dir'], 'complexity_eval_results.csv'), index=False)

    logging.info('Evaluations completed.')
