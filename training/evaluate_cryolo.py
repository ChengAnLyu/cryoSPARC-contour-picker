"""Evaluate real crYOLO CBOX results through the production contour pipeline.

The reference labels are partial picking extents. Metrics cover known cores and
explicitly confirmed blanks; they are not exhaustive particle recall or precision.
Run crYOLO on the exported PNGs first. Missing output is an error, not a blank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recognition import RecognitionConfig
from cryolo_recognition import CryoloRecognizer, read_cbox


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def evaluate(manifest_path, cbox_directory, weights, config_path, profile, threshold,
             output, case_ids=None, scope='in-sample final-model diagnostic'):
    manifest_path = Path(manifest_path)
    dataset = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    config = RecognitionConfig(str(weights), backend='cryolo',
                               cryolo_config=str(config_path), automatic_profile='capsid-learned',
                               capsid_prior_path=str(profile), cryolo_threshold=threshold)
    recognizer = CryoloRecognizer(config)
    records = []
    all_regions = []
    selected_ids = set(case_ids) if case_ids is not None else {c['case_id'] for c in dataset['cases']}
    if not selected_ids or not selected_ids <= {c['case_id'] for c in dataset['cases']}:
        raise ValueError('Case selection must identify existing dataset cases.')
    try:
        for case in dataset['cases']:
            cid = case['case_id']
            if cid not in selected_ids:
                continue
            rgb = np.asarray(Image.open(root / case['images']['input']['path']).convert('RGB'))
            stem = f'case_{cid:03d}'
            cbox_path = Path(cbox_directory) / (stem + '.cbox')
            boxes = read_cbox(cbox_path, rgb.shape[:2], threshold)
            started = time.perf_counter()
            candidates, evidence, carbon, area, appearance = recognizer._refine(rgb, boxes, None)
            elapsed = time.perf_counter() - started
            union = np.zeros(rgb.shape[:2], dtype=bool)
            for candidate in candidates:
                union |= candidate.mask
            valid = np.asarray(Image.open(root / case['valid_image_pixels']['path'])) > 0
            regions = np.asarray(Image.open(root / case['valid_image_pixels']['positive_regions_path']))
            refs = []
            for rid in case['training_supervision']['confirmed_positive_region_ids']:
                core = ndi.binary_erosion(regions == rid, iterations=6) & valid
                size = int(core.sum())
                if not size:
                    raise ValueError(f'Empty confirmed core: case {cid} region {rid}')
                covered = int((core & union).sum())
                item = {'case_id': cid, 'region_id': rid, 'core_pixels': size,
                        'covered_pixels': covered, 'coverage': covered / size}
                refs.append(item)
                all_regions.append(item)
            records.append({'case_id': cid, 'confirmed_blank': case['training_supervision']['confirmed_blank'],
                            'confirmed_regions': refs, 'raw_boxes_above_threshold': len(boxes),
                            'final_contours': len(candidates), 'selected_pixels': int(union.sum()),
                            'refinement_seconds': elapsed, 'cbox_sha256': sha256(cbox_path),
                            'carbon_guard': carbon, 'area_filter': area,
                            'candidate_evidence': evidence})
    finally:
        recognizer.close()
    blanks = [r for r in records if r['confirmed_blank']]
    core_pixels = sum(r['core_pixels'] for r in all_regions)
    covered = sum(r['covered_pixels'] for r in all_regions)
    result = {'scope': scope, 'model_sha256': sha256(weights), 'config_sha256': sha256(config_path),
              'appearance_profile_sha256': sha256(profile), 'manifest_sha256': sha256(manifest_path),
              'threshold': threshold, 'minimum_contour_area_px': config.min_capsid_area_px,
              'summary': {'cases': len(records), 'confirmed_regions': len(all_regions),
                          'core_pixels': core_pixels, 'covered_core_pixels': covered,
                          'core_pixel_coverage': covered / core_pixels if core_pixels else None,
                          'regions_at_least_half_covered': sum(r['coverage'] >= .5 for r in all_regions),
                          'regions_at_least_90_percent_covered': sum(r['coverage'] >= .9 for r in all_regions),
                          'confirmed_blank_cases': len(blanks),
                          'unwanted_contours_on_confirmed_blanks': sum(r['final_contours'] for r in blanks),
                          'confirmed_blanks_with_any_contour': sum(r['final_contours'] > 0 for r in blanks),
                          'refinement_seconds': sum(r['refinement_seconds'] for r in records)},
              'limitations': ['Known marked-core recovery only; incomplete annotations cannot measure exhaustive recall or precision.',
                              'Reference picking extents are not exact biological boundaries.',
                              'Only explicitly confirmed blanks contribute to unwanted-contour counts.',
                              'Prediction time must be measured separately; refinement time excludes crYOLO inference.',
                              'The reused appearance profile was fitted on all 160 examples; detector-only held-out predictions do not make this full contour pipeline held out.'],
              'cases': records}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'cbox-directory', 'weights', 'config', 'profile', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--threshold', type=float, default=.3)
    parser.add_argument('--case-ids', type=int, nargs='+')
    parser.add_argument('--scope', default='in-sample final-model diagnostic')
    args = parser.parse_args()
    result = evaluate(args.manifest, args.cbox_directory, args.weights, args.config,
                      args.profile, args.threshold, args.output, args.case_ids, args.scope)
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
