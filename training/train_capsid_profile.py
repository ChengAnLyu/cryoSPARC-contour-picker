"""Train a small positive-unlabeled appearance profile from a private manifest.

The manifest must identify clean screenshots, valid image pixels, and partial
positive marker clusters. No outside pixels are assigned a negative target.
Run with --dataset manifest.json --folds folds.json --output output_directory.
Requires PyTorch for training; runtime inference uses NumPy and SciPy only.
"""
from pathlib import Path
import argparse
import hashlib
import json
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


def training_positive_mask(regions, annotations, valid, policy='substantial'):
    """Select conservative centers only; an unmarked pixel is never negative."""
    if policy not in ('substantial', 'all-marked'):
        raise ValueError('Unsupported positive selection policy.')
    ids = [item['region_id'] for item in annotations
           if policy == 'all-marked' or item['substantial_cluster']]
    if regions.shape != valid.shape or regions.ndim != 2:
        raise ValueError('Training regions and valid-image mask must have the same two-dimensional shape.')
    return ndi.binary_erosion(np.isin(regions, ids), iterations=6) & valid, ids


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--folds', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--positive-policy', choices=('substantial', 'all-marked'), default='substantial',
                        help='Use earlier substantial marker groups, or conservative cores from every marked group.')
    parser.add_argument('--positive-sampling', choices=('pixel', 'case-balanced'), default='pixel',
                        help='Draw positives uniformly by available pixel, or uniformly by positive screenshot then pixel.')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
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
    all_ids = set(cases)
    held_out = []
    for fold in folds:
        train, test = set(fold['train_cases']), set(fold['test_cases'])
        if train & test or train | test != all_ids:
            raise ValueError('Each fold must separate every case into training or held-out data.')
        train_groups = {cases[c]['evaluation_group'] for c in train}
        test_groups = {cases[c]['evaluation_group'] for c in test}
        if train_groups & test_groups:
            raise ValueError('Overlapping screenshot groups cross a training/held-out split.')
        held_out.extend(test)
    if sorted(held_out) != sorted(all_ids):
        raise ValueError('Every case must be held out exactly once.')

    config = {'steps': args.steps, 'positive_class_prior': .15, 'learning_rate': .001,
              'weight_decay': .01, 'positive_batch': 256, 'unlabeled_batch': 1024,
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
        positive_ids = [item['region_id'] for item in case['annotations']['regions']
                        if args.positive_policy == 'all-marked' or item['substantial_cluster']]
        provenance = {**base_provenance, 'case_id': cid, 'input_sha256': sha_file(clean),
                      'valid_mask_sha256': sha_file(valid_path),
                      'positive_regions_sha256': sha_file(positive_path),
                      'positive_region_ids': positive_ids}
        sig = fingerprint(provenance)
        dest = output / 'samples' / f'case_{cid:02d}.npz'
        if dest.exists():
            with np.load(dest, allow_pickle=False) as old:
                if str(old['fingerprint']) != sig:
                    raise ValueError(f'Stale sample cache: {dest}. Choose a new output directory.')
                samples[cid] = {key: old[key] for key in ('p', 'u', 'p_indices', 'u_indices')}
        else:
            rgb = np.asarray(Image.open(clean).convert('RGB'))
            features, valid = appearance_features(rgb)
            roi = np.asarray(Image.open(valid_path)) > 0
            regions = np.asarray(Image.open(positive_path))
            if roi.shape != valid.shape or regions.shape != valid.shape:
                raise ValueError('Dataset masks and clean screenshot have different dimensions.')
            valid &= roi
            positive, checked_ids = training_positive_mask(regions, case['annotations']['regions'], valid,
                                                          args.positive_policy)
            assert positive_ids == checked_ids
            rng = np.random.default_rng(config['sample_seed_base'] + cid)
            pp, uu = np.flatnonzero(positive), np.flatnonzero(valid)
            if len(pp) > 2500:
                pp = rng.choice(pp, 2500, replace=False)
            if len(uu) > 5000:
                uu = rng.choice(uu, 5000, replace=False)
            flat = features.reshape(-1, len(FEATURE_NAMES))
            samples[cid] = {'p': flat[pp], 'u': flat[uu], 'p_indices': pp, 'u_indices': uu}
            np.savez_compressed(dest, **samples[cid], fingerprint=sig)
        sample_provenance[str(cid)] = provenance
        source_paths[cid] = clean
        print('prepared', cid, len(samples[cid]['p']), len(samples[cid]['u']), flush=True)
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
        case_lengths = [len(samples[c]['p']) for c in train_cases]
        offsets = np.cumsum(np.r_[0, case_lengths])
        if not len(p) or not len(u):
            raise ValueError('Each training fold needs marked-positive and unlabeled samples.')
        mean, scale = u.mean(0), np.maximum(u.std(0), .05)
        p = torch.from_numpy(np.clip((p - mean) / scale, -8, 8).astype(np.float32))
        u = torch.from_numpy(np.clip((u - mean) / scale, -8, 8).astype(np.float32))
        net = nn.Sequential(nn.Linear(23, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1))
        optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.01)
        positive_draws = np.zeros(len(p), np.int64)
        loss_trace = []
        for step in range(args.steps):
            rows = sample_positive_rows(rng, case_lengths, 256, args.positive_sampling)
            np.add.at(positive_draws, rows, 1)
            fp = net(p[rows]).view(-1)
            fu = net(u[rng.integers(len(u), size=1024)]).view(-1)
            positive_risk = .15 * nn.functional.softplus(-fp).mean()
            negative_risk = nn.functional.softplus(fu).mean() - .15 * nn.functional.softplus(fp).mean()
            loss = positive_risk + torch.relu(negative_risk)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 100 == 0 or step == args.steps - 1:
                loss_trace.append({'step': step + 1, 'positive_risk': float(positive_risk.detach()),
                                   'unlabeled_minus_positive_risk': float(negative_risk.detach()),
                                   'clamped_total': float(loss.detach())})
        draws_path = destination.with_name(destination.stem + '-draws.npz')
        np.savez_compressed(draws_path, positive_row_draws=positive_draws,
                            train_case_ids=np.asarray(train_cases), positive_case_offsets=offsets)
        draw_summary = {}
        for index, cid in enumerate(train_cases):
            draws = positive_draws[offsets[index]:offsets[index + 1]]
            draw_summary[str(cid)] = {'available_positive_rows': len(draws), 'positive_draws': int(draws.sum()),
                                      'unique_positive_rows_drawn': int(np.count_nonzero(draws)),
                                      'max_reuse_of_one_row': int(draws.max()) if len(draws) else 0}
        model = {'format': PROFILE_FORMAT, 'feature_names': list(FEATURE_NAMES),
                 'mean': mean.tolist(), 'scale': scale.tolist(),
                 'layers': [{'weight': layer.weight.detach().numpy().tolist(),
                             'bias': layer.bias.detach().numpy().tolist()}
                            for layer in net if isinstance(layer, nn.Linear)],
                 'training': {'method': 'nonnegative positive-unlabeled logistic risk',
                              'assumed_positive_prior': .15, 'seed': seed, 'steps': args.steps,
                              'case_ids': train_cases, 'positive_sample_count': len(p),
                              'unlabeled_sample_count': len(u), 'labels_exhaustive': False,
                              'unlabeled_are_negatives': False,
                              'positive_selection': ('6px-eroded substantial marked regions' if args.positive_policy == 'substantial'
                                                     else '6px-eroded cores of every marked region, including small confident groups')
                                                     + '; confidence selected, not random positives',
                              'positive_sampling': args.positive_sampling,
                              'positive_draw_summary': draw_summary,
                              'positive_draws_sha256': sha_file(draws_path),
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
