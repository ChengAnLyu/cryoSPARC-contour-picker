"""Train a small capsid appearance profile from a private manifest.

The manifest must identify clean screenshots, valid image pixels, and partial
positive marker clusters. Other unmarked pixels remain unknown. Explicit
confirmed-blank supervision can add a separate negative pool; absence of marks
alone never assigns a negative target. The default preserves legacy PU training.
Run with --dataset manifest.json --folds folds.json --output output_directory.
Requires PyTorch for training; runtime inference uses NumPy and SciPy only.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import capsid_prior
from capsid_prior import FEATURE_NAMES, PROFILE_FORMAT, appearance_features, load_profile, score_image


def sha_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_deck_records(manifest):
    """Record each original deck without mistaking the latest deck for all data."""
    records = manifest.get('source_decks')
    if records is None:
        records = [{'path': manifest.get('source_deck', ''),
                    'sha256': manifest.get('source_sha256'),
                    'case_ids': [case['case_id'] for case in manifest['cases']]}]
    if not isinstance(records, list) or not records:
        raise ValueError('Training requires nonempty source-deck provenance.')
    result, represented = [], []
    for record in records:
        digest = record.get('sha256')
        ids = record.get('case_ids')
        if (not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)
                or not isinstance(ids, list) or not ids
                or any(type(cid) is not int for cid in ids)):
            raise ValueError('Invalid source-deck hash or case assignment.')
        represented.extend(ids)
        result.append({'name': Path(record.get('path') or record.get('uploaded_name') or '').name,
                       'sha256': digest, 'case_ids': sorted(ids)})
    if sorted(represented) != sorted(case['case_id'] for case in manifest['cases']):
        raise ValueError('Source decks must assign each training case exactly once.')
    return result


def _confirmed_region_ids(ids, annotations):
    """Validate explicit positive IDs without inventing labels from size."""
    if (not isinstance(ids, list) or any(type(value) is not int or value < 1 for value in ids)
            or len(ids) != len(set(ids))):
        raise ValueError('Confirmed positive region IDs must be unique positive integers.')
    if (not isinstance(annotations, list)
            or any(not isinstance(item, dict) or type(item.get('region_id')) is not int
                   or item['region_id'] < 1 for item in annotations)):
        raise ValueError('Annotated regions must have positive integer IDs.')
    annotated = [item['region_id'] for item in annotations]
    if len(annotated) != len(set(annotated)) or not set(ids) <= set(annotated):
        raise ValueError('Confirmed positive IDs must identify existing, unique annotated regions.')
    return list(ids)


def validate_training_supervision(case):
    """Require explicit user confirmation for every supervised training case."""
    record = case.get('training_supervision')
    if not isinstance(record, dict):
        raise ValueError('Explicit training_supervision is required for every supervised case.')
    if type(record.get('confirmed_blank')) is not bool:
        raise ValueError('confirmed_blank must be an explicit boolean.')
    if record.get('positive_scope') != 'large_capsids' or record.get('unmarked_regions') != 'unknown':
        raise ValueError('Supervision must confirm large_capsids and leave unmarked regions unknown.')
    confirmation = record.get('confirmation')
    if not ((isinstance(confirmation, str) and confirmation.strip())
            or (isinstance(confirmation, dict) and confirmation)):
        raise ValueError('Training supervision requires nonempty user-confirmation provenance.')
    try:
        json.dumps(confirmation, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('User-confirmation provenance must contain finite JSON data.') from exc
    annotations = case.get('annotations', {}).get('regions')
    ids = _confirmed_region_ids(record.get('confirmed_positive_region_ids'), annotations)
    if record['confirmed_blank'] and (ids or annotations):
        raise ValueError('A confirmed blank cannot also contain positive annotations.')
    return {'confirmed_blank': record['confirmed_blank'], 'confirmed_positive_region_ids': ids,
            'positive_scope': 'large_capsids', 'confirmation': confirmation, 'unmarked_regions': 'unknown'}


def training_positive_mask(regions, annotations, valid, policy='substantial',
                           confirmed_region_ids=None):
    """Select conservative centers only; an unmarked pixel is never negative."""
    if policy not in ('substantial', 'all-marked', 'confirmed-large'):
        raise ValueError('Unsupported positive selection policy.')
    if regions.shape != valid.shape or regions.ndim != 2:
        raise ValueError('Training regions and valid-image mask must have the same two-dimensional shape.')
    if policy == 'confirmed-large':
        ids = _confirmed_region_ids(confirmed_region_ids, annotations)
        if not set(ids) <= set(np.unique(regions)):
            raise ValueError('A confirmed positive region is absent from the region bitmap.')
    else:
        ids = [item['region_id'] for item in annotations
               if policy == 'all-marked' or item['substantial_cluster']]
    return ndi.binary_erosion(np.isin(regions, ids), iterations=6) & valid, ids


def training_masks(regions, annotations, valid, policy='substantial', supervision=None):
    """Split valid pixels into P/U/N without turning unknown pixels negative.

    Positive locations can also belong to U, as required by the PU mixture.
    Confirmed-blank cases contribute only N. Legacy callers omit supervision.
    """
    ids = None if supervision is None else supervision['confirmed_positive_region_ids']
    positive, selected = training_positive_mask(regions, annotations, valid, policy, ids)
    if supervision is not None and not set(selected) <= set(ids):
        raise ValueError('A positive policy selected unconfirmed regions; use confirmed-large.')
    negative = np.zeros(valid.shape, dtype=bool)
    if supervision is not None and supervision['confirmed_blank']:
        if positive.any() or selected or annotations:
            raise ValueError('Confirmed blank and positive supervision conflict.')
        negative = valid.copy()
    unlabeled = valid & ~negative
    return positive, unlabeled, negative, selected


def sample_training_indices(rng, positive, unlabeled, negative):
    """Keep legacy P/U draws identical; independently bound confirmed N rows."""
    if positive.shape != unlabeled.shape or positive.shape != negative.shape:
        raise ValueError('P/U/N sample masks must have matching dimensions.')
    result = []
    for mask, limit in ((positive, 2500), (unlabeled, 5000), (negative, 5000)):
        rows = np.flatnonzero(mask)
        if len(rows) > limit:
            rows = rng.choice(rows, limit, replace=False)
        result.append(rows)
    return tuple(result)


def validate_training_options(positive_prior, negative_weight):
    if (isinstance(positive_prior, (bool, np.bool_)) or not math.isfinite(positive_prior)
            or not 0 < positive_prior < 1):
        raise ValueError('positive-prior must be finite and strictly between zero and one.')
    if (isinstance(negative_weight, (bool, np.bool_)) or not math.isfinite(negative_weight)
            or negative_weight < 0):
        raise ValueError('negative-weight must be finite and nonnegative.')


def validate_folds(cases, folds):
    """Keep all supervised pools for one screenshot group on the same side."""
    if not isinstance(folds, list) or not folds:
        raise ValueError('Training requires nonempty grouped folds.')
    all_ids, held_out = set(cases), []
    for fold in folds:
        train_list, test_list = fold['train_cases'], fold['test_cases']
        for ids in (train_list, test_list):
            if (not isinstance(ids, list) or any(type(cid) is not int for cid in ids)
                    or len(ids) != len(set(ids))):
                raise ValueError('Fold case IDs must be unique integers.')
        train, test = set(train_list), set(test_list)
        if not train or not test or train & test or train | test != all_ids:
            raise ValueError('Each fold must separate every case into training or held-out data.')
        train_groups = {cases[c]['evaluation_group'] for c in train}
        test_groups = {cases[c]['evaluation_group'] for c in test}
        if train_groups & test_groups:
            raise ValueError('Overlapping screenshot groups cross a training/held-out split.')
        held_out.extend(test)
    if sorted(held_out) != sorted(all_ids):
        raise ValueError('Every case must be held out exactly once.')


def training_loss(positive_logits, unlabeled_logits, negative_logits=None, *,
                  positive_prior=.15, negative_weight=0.):
    """Compute nnPU risk plus optional confirmed-negative logistic loss."""
    import torch
    from torch.nn.functional import softplus
    validate_training_options(positive_prior, negative_weight)
    if not positive_logits.numel() or not unlabeled_logits.numel():
        raise ValueError('Training loss requires positive and unlabeled logits.')
    if negative_weight and (negative_logits is None or not negative_logits.numel()):
        raise ValueError('Positive negative-weight requires confirmed-negative logits.')
    positive_logistic = softplus(-positive_logits).mean()
    unlabeled_negative_logistic = softplus(unlabeled_logits).mean()
    positive_negative_logistic = softplus(positive_logits).mean()
    positive_risk = positive_prior * positive_logistic
    negative_risk = unlabeled_negative_logistic - positive_prior * positive_negative_logistic
    clamped_nnpu = positive_risk + torch.relu(negative_risk)
    confirmed_negative_risk = (softplus(negative_logits).mean()
                               if negative_logits is not None and negative_logits.numel()
                               else positive_risk.new_zeros(()))
    weighted_negative_risk = negative_weight * confirmed_negative_risk
    # Keep the legacy expression exact when no extra objective is requested.
    loss = clamped_nnpu + weighted_negative_risk if negative_weight else clamped_nnpu
    return loss, {'positive_logistic': positive_logistic,
                  'unlabeled_negative_logistic': unlabeled_negative_logistic,
                  'positive_negative_logistic': positive_negative_logistic,
                  'positive_risk': positive_risk,
                  'unlabeled_minus_positive_risk': negative_risk,
                  'clamped_nnpu_total': clamped_nnpu,
                  'confirmed_negative_risk': confirmed_negative_risk,
                  'weighted_negative_risk': weighted_negative_risk,
                  'clamped_total': loss}


def sample_positive_rows(rng, case_lengths, batch_size, mode='pixel'):
    """Draw original row indices, optionally giving each positive case equal weight."""
    lengths = np.asarray(case_lengths, dtype=np.int64)
    if lengths.ndim != 1 or np.any(lengths < 0) or not lengths.sum() or batch_size < 1:
        raise ValueError('Positive sampling requires nonempty nonnegative case lengths and a positive batch size.')
    if mode == 'pixel':
        return rng.integers(int(lengths.sum()), size=batch_size)
    if mode != 'case-balanced':
        raise ValueError('Unsupported positive sampling mode.')
    eligible = np.flatnonzero(lengths)
    selected = eligible[rng.integers(len(eligible), size=batch_size)]
    offsets = np.cumsum(np.r_[0, lengths[:-1]])
    return offsets[selected] + (rng.random(batch_size) * lengths[selected]).astype(np.int64)


def sample_negative_rows(rng, case_lengths, batch_size):
    """Give each confirmed-blank screenshot equal expected negative weight."""
    return sample_positive_rows(rng, case_lengths, batch_size, 'case-balanced')


def pool_draw_summary(train_cases, lengths, draws, label):
    """Describe actual row draws, including cases with no rows in this pool."""
    lengths, draws = np.asarray(lengths), np.asarray(draws)
    if (label not in ('positive', 'unlabeled', 'negative') or lengths.ndim != 1 or draws.ndim != 1
            or len(train_cases) != len(lengths) or np.any(lengths < 0)
            or int(lengths.sum()) != len(draws) or np.any(draws < 0)):
        raise ValueError('Invalid training-pool draw evidence.')
    offsets = np.cumsum(np.r_[0, lengths])
    summary = {}
    for index, cid in enumerate(train_cases):
        part = draws[offsets[index]:offsets[index + 1]]
        summary[str(cid)] = {f'available_{label}_rows': len(part), f'{label}_draws': int(part.sum()),
                             f'unique_{label}_rows_drawn': int(np.count_nonzero(part)),
                             'max_reuse_of_one_row': int(part.max()) if len(part) else 0}
    return summary, offsets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--folds', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--positive-policy', choices=('substantial', 'all-marked', 'confirmed-large'), default='substantial',
                        help='Use substantial groups, all marked cores, or explicit confirmed large-region IDs.')
    parser.add_argument('--positive-sampling', choices=('pixel', 'case-balanced'), default='pixel',
                        help='Draw positives uniformly by available pixel, or uniformly by positive screenshot then pixel.')
    parser.add_argument('--positive-prior', type=float, default=.15,
                        help='Assumed positive fraction in the unlabeled pool; default 0.15.')
    parser.add_argument('--negative-weight', type=float, default=0.,
                        help='Additional confirmed-blank negative loss weight; 0 retains the PU objective.')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    try:
        validate_training_options(args.positive_prior, args.negative_weight)
    except ValueError as exc:
        parser.error(str(exc))
    import torch
    from torch import nn
    data_root = args.dataset.resolve().parent
    output = args.output.resolve()
    for folder in (output, output / 'samples', output / 'scores'):
        folder.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.dataset.read_text())
    folds = json.loads(args.folds.read_text())['folds']
    cases = {case['case_id']: case for case in manifest['cases']}
    source_decks = source_deck_records(manifest)
    validate_folds(cases, folds)
    supervised = args.positive_policy == 'confirmed-large' or args.negative_weight > 0
    supervision = {cid: validate_training_supervision(case) for cid, case in cases.items()} if supervised else {}

    config = {'steps': args.steps, 'positive_class_prior': args.positive_prior, 'learning_rate': .001,
              'weight_decay': .01, 'positive_batch': 256, 'unlabeled_batch': 1024,
              'negative_batch': 256, 'negative_weight': args.negative_weight,
              'negative_sampling': 'case-balanced', 'negative_samples_per_case': 5000,
              'explicit_supervision_enabled': supervised,
              'normalizer_source': 'training unlabeled rows only',
              'hidden_widths': [32, 16], 'activation': 'tanh', 'positive_erosion_px': 6,
              'positive_samples_per_case': 2500, 'unlabeled_samples_per_case': 5000,
              'sample_seed_base': 260000, 'fold_seed_base': 260000, 'full_seed': 260999,
              'positive_policy': args.positive_policy, 'positive_sampling': args.positive_sampling}
    base_provenance = {'manifest_sha256': sha_file(args.dataset),
                       'folds_sha256': sha_file(args.folds),
                       'feature_module_sha256': sha_file(capsid_prior.__file__),
                       'trainer_sha256': sha_file(__file__), 'source_decks': source_decks, 'config': config}

    def relative(name):
        path = (data_root / name).resolve()
        if not path.is_relative_to(data_root) or not path.is_file():
            raise ValueError(f'Invalid dataset-relative file: {name}')
        return path

    samples = {}
    sample_provenance = {}
    source_paths = {}
    for cid, case in sorted(cases.items()):
        clean = relative(case['images']['input']['path'])
        valid_path = relative(case['valid_image_pixels']['path'])
        positive_path = relative(case['valid_image_pixels']['positive_regions_path'])
        case_supervision = supervision.get(cid)
        positive_ids = (case_supervision['confirmed_positive_region_ids']
                        if args.positive_policy == 'confirmed-large' else
                        [item['region_id'] for item in case['annotations']['regions']
                         if args.positive_policy == 'all-marked' or item['substantial_cluster']])
        provenance = {**base_provenance, 'case_id': cid, 'input_sha256': sha_file(clean),
                      'valid_mask_sha256': sha_file(valid_path),
                      'positive_regions_sha256': sha_file(positive_path),
                      'positive_region_ids': positive_ids,
                      'training_supervision': case_supervision}
        sig = fingerprint(provenance)
        dest = output / 'samples' / f'case_{cid:02d}.npz'
        if dest.exists():
            with np.load(dest, allow_pickle=False) as old:
                if str(old['fingerprint']) != sig:
                    raise ValueError(f'Stale sample cache: {dest}. Choose a new output directory.')
                keys = ('p', 'u', 'n', 'p_indices', 'u_indices', 'n_indices')
                if not set(keys) <= set(old.files):
                    raise ValueError(f'Incomplete P/U/N sample cache: {dest}. Choose a new output directory.')
                samples[cid] = {key: old[key] for key in keys}
        else:
            rgb = np.asarray(Image.open(clean).convert('RGB'))
            features, valid = appearance_features(rgb)
            roi = np.asarray(Image.open(valid_path)) > 0
            regions = np.asarray(Image.open(positive_path))
            if roi.shape != valid.shape or regions.shape != valid.shape:
                raise ValueError('Dataset masks and clean screenshot have different dimensions.')
            valid &= roi
            positive, unlabeled, negative, checked_ids = training_masks(
                regions, case['annotations']['regions'], valid, args.positive_policy, case_supervision)
            assert positive_ids == checked_ids
            rng = np.random.default_rng(config['sample_seed_base'] + cid)
            pp, uu, negative_indices = sample_training_indices(rng, positive, unlabeled, negative)
            flat = features.reshape(-1, len(FEATURE_NAMES))
            samples[cid] = {'p': flat[pp], 'u': flat[uu], 'n': flat[negative_indices],
                            'p_indices': pp, 'u_indices': uu, 'n_indices': negative_indices}
            np.savez_compressed(dest, **samples[cid], fingerprint=sig)
        provenance['sample_counts'] = {key: len(samples[cid][key]) for key in ('p', 'u', 'n')}
        sample_provenance[str(cid)] = provenance
        source_paths[cid] = clean
        print('prepared', cid, 'P', len(samples[cid]['p']), 'U', len(samples[cid]['u']),
              'N', len(samples[cid]['n']), flush=True)
    data_fingerprint = fingerprint(sample_provenance)
    (output / 'training-provenance.json').write_text(json.dumps({
        **base_provenance, 'data_fingerprint': data_fingerprint,
        'sample_provenance': sample_provenance}, indent=2) + '\n')
    torch.set_num_threads(2)

    def fit(train_cases, seed, destination):
        model_fingerprint = fingerprint({'data': data_fingerprint, 'train_cases': train_cases,
                                         'seed': seed, 'base': base_provenance})
        if destination.exists():
            model = json.loads(destination.read_text())
            if model['training'].get('model_fingerprint') != model_fingerprint:
                raise ValueError(f'Stale model cache: {destination}. Choose a new output directory.')
            draws_path = destination.with_name(destination.stem + '-draws.npz')
            if (not draws_path.is_file()
                    or model['training'].get('positive_draws_sha256') != sha_file(draws_path)):
                raise ValueError(f'Incomplete model sampling evidence: {destination}. Choose a new output directory.')
            load_profile(destination)
            return model
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        p = np.concatenate([samples[c]['p'] for c in train_cases])
        u = np.concatenate([samples[c]['u'] for c in train_cases])
        n = np.concatenate([samples[c]['n'] for c in train_cases])
        case_lengths = [len(samples[c]['p']) for c in train_cases]
        unlabeled_case_lengths = [len(samples[c]['u']) for c in train_cases]
        negative_case_lengths = [len(samples[c]['n']) for c in train_cases]
        if not len(p) or not len(u):
            raise ValueError('Each training fold needs marked-positive and unlabeled samples.')
        if args.negative_weight and not len(n):
            raise ValueError('Each supervised-negative training fold needs confirmed-blank samples.')
        # Keep normalizers restricted to this fold's unlabeled pool. Neither
        # held-out examples nor their confirmed negative pixels contribute.
        mean, scale = u.mean(0), np.maximum(u.std(0), .05)
        p = torch.from_numpy(np.clip((p - mean) / scale, -8, 8).astype(np.float32))
        u = torch.from_numpy(np.clip((u - mean) / scale, -8, 8).astype(np.float32))
        n = torch.from_numpy(np.clip((n - mean) / scale, -8, 8).astype(np.float32))
        net = nn.Sequential(nn.Linear(23, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1))
        optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.01)
        positive_draws = np.zeros(len(p), np.int64)
        unlabeled_draws = np.zeros(len(u), np.int64)
        negative_draws = np.zeros(len(n), np.int64)
        loss_trace = []
        for step in range(args.steps):
            rows = sample_positive_rows(rng, case_lengths, 256, args.positive_sampling)
            np.add.at(positive_draws, rows, 1)
            fp = net(p[rows]).view(-1)
            unlabeled_rows = rng.integers(len(u), size=1024)
            np.add.at(unlabeled_draws, unlabeled_rows, 1)
            fu = net(u[unlabeled_rows]).view(-1)
            fn = None
            if args.negative_weight:
                negative_rows = sample_negative_rows(rng, negative_case_lengths, config['negative_batch'])
                np.add.at(negative_draws, negative_rows, 1)
                fn = net(n[negative_rows]).view(-1)
            loss, components = training_loss(fp, fu, fn, positive_prior=args.positive_prior,
                                             negative_weight=args.negative_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0 or step == args.steps - 1:
                loss_trace.append({'step': step + 1,
                                   **{name: float(value.detach()) for name, value in components.items()}})
        draw_summary, offsets = pool_draw_summary(train_cases, case_lengths, positive_draws, 'positive')
        unlabeled_summary, unlabeled_offsets = pool_draw_summary(
            train_cases, unlabeled_case_lengths, unlabeled_draws, 'unlabeled')
        negative_summary, negative_offsets = pool_draw_summary(
            train_cases, negative_case_lengths, negative_draws, 'negative')
        draws_path = destination.with_name(destination.stem + '-draws.npz')
        np.savez_compressed(draws_path, positive_row_draws=positive_draws,
                            unlabeled_row_draws=unlabeled_draws, negative_row_draws=negative_draws,
                            train_case_ids=np.asarray(train_cases), positive_case_offsets=offsets,
                            unlabeled_case_offsets=unlabeled_offsets, negative_case_offsets=negative_offsets)
        confirmations = {fingerprint(supervision[c]['confirmation']): supervision[c]['confirmation']
                         for c in train_cases if c in supervision}
        supervision_record = {
            'enabled': supervised, 'negative_loss_applied': args.negative_weight > 0,
            'confirmed_blank_case_ids': [c for c in train_cases if c in supervision and supervision[c]['confirmed_blank']],
            'confirmed_positive_region_ids': {str(c): supervision[c]['confirmed_positive_region_ids']
                                             for c in train_cases if c in supervision},
            'confirmation_records': confirmations,
            'case_confirmation_ids': {str(c): fingerprint(supervision[c]['confirmation'])
                                      for c in train_cases if c in supervision},
            'unmarked_regions': 'unknown',
            'negative_scope': 'valid pixels in explicitly confirmed blank images only',
        }
        positive_description = {'substantial': '6px-eroded substantial marked regions',
                                'all-marked': '6px-eroded cores of every marked region, including small confident groups',
                                'confirmed-large': '6px-eroded cores of explicitly confirmed large-capsid region IDs'}
        model = {'format': PROFILE_FORMAT, 'feature_names': list(FEATURE_NAMES),
                 'mean': mean.tolist(), 'scale': scale.tolist(),
                 'layers': [{'weight': layer.weight.detach().numpy().tolist(),
                             'bias': layer.bias.detach().numpy().tolist()}
                            for layer in net if isinstance(layer, nn.Linear)],
                 'training': {'method': ('nonnegative positive-unlabeled logistic risk with confirmed-negative logistic loss'
                                         if args.negative_weight else 'nonnegative positive-unlabeled logistic risk'),
                              'assumed_positive_prior': args.positive_prior,
                              'negative_weight': args.negative_weight, 'seed': seed, 'steps': args.steps,
                              'case_ids': train_cases, 'positive_sample_count': len(p),
                              'unlabeled_sample_count': len(u), 'negative_sample_count': len(n),
                              'labels_exhaustive': False,
                              'unlabeled_are_negatives': False,
                              'positive_selection': positive_description[args.positive_policy]
                                                     + '; confidence selected, not random positives',
                              'positive_sampling': args.positive_sampling,
                              'negative_sampling': 'case-balanced',
                              'training_supervision': supervision_record,
                              'positive_draw_summary': draw_summary,
                              'unlabeled_draw_summary': unlabeled_summary,
                              'negative_draw_summary': negative_summary,
                              'positive_draws_sha256': sha_file(draws_path),
                              'sampling_draws_sha256': sha_file(draws_path),
                              'loss_trace': loss_trace,
                              'score_is_calibrated_probability': False,
                              'source_decks': source_decks,
                              'data_fingerprint': data_fingerprint,
                              'model_fingerprint': model_fingerprint,
                              'provenance': base_provenance,
                              'validation_scope': 'screenshot-group internal development validation; acquisition grouping unknown'}}
        if len(source_decks) == 1:
            model['training']['source_deck_sha256'] = source_decks[0]['sha256']
        destination.write_text(json.dumps(model, indent=2) + '\n')
        load_profile(destination)
        return model

    for index, fold in enumerate(folds):
        model_path = output / f'prior-fold-{index}.json'
        fit(fold['train_cases'], 260000 + index, model_path)
        profile = load_profile(model_path)
        for cid in fold['test_cases']:
            score_path = output / 'scores' / f'case_{cid:02d}.npy'
            metadata = score_path.with_suffix('.json')
            score_signature = fingerprint({'model_sha256': profile.sha256,
                                           'input_sha256': sha_file(source_paths[cid]),
                                           'feature_module_sha256': base_provenance['feature_module_sha256']})
            if score_path.exists():
                if (not metadata.exists() or json.loads(metadata.read_text()).get('fingerprint') != score_signature
                        or json.loads(metadata.read_text()).get('score_sha256') != sha_file(score_path)):
                    raise ValueError(f'Stale score cache: {score_path}. Choose a new output directory.')
            else:
                rgb = np.asarray(Image.open(source_paths[cid]).convert('RGB'))
                scores = score_image(rgb, profile)
                np.save(score_path, scores)
                metadata.write_text(json.dumps({'fingerprint': score_signature,
                                                'case_id': cid, 'fold': index,
                                                'model_sha256': profile.sha256,
                                                'score_sha256': sha_file(score_path)}, indent=2) + '\n')
            print('scored heldout', index, cid, flush=True)
    fit(sorted(cases), 260999, output / 'capsid-profile.json')
    print('completed all folds and full profile', flush=True)


if __name__ == '__main__':
    main()
