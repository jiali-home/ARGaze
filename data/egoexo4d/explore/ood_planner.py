#!/usr/bin/env python3
"""
OOD Feasibility & Split Planner

Evaluates feasibility for OOD-Site / OOD-Task / OOD-Participant splits using
control-variable constraints, recommends candidates, assigns splits, and
optionally subsamples after splitting.

Inputs: CSV with columns for task, site, participant (auto-inferred or specified).
Outputs (under --outdir, default: explore/ood):
  - feasibility_site.csv, feasibility_task.csv, feasibility_participant.csv
  - ood_recommendations.json
  - split_assignments.csv (split column: TRAIN/VAL/TEST_IID/TEST_OOD_SITE/TEST_OOD_TASK/TEST_OOD_PARTICIPANT)
  - split_summary.csv (counts per split)
  - post_split_checks/*.csv (coverage checks per OOD type)
  - if subsampling: split_assignments_subsampled.csv + summary

Example:
  python explore/ood_planner.py \
    --csv /path/to/your.csv \
    --task-col task_name --site-col university_name --participant-col participant_uid \
    --iid-frac 0.10 --val-frac 0.10 --seed 42 \
    --ood-site-min-task-sites 3 --ood-task-min-sites 3 --ood-task-min-participants 5 \
    --subsample-after 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------- Utilities --------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _series_for(df: pd.DataFrame, colname: str) -> pd.Series:
    matches = [i for i, c in enumerate(df.columns) if c == colname]
    if not matches:
        raise KeyError(f"Column not found: {colname}")
    s = df.iloc[:, matches[0]].copy()
    s.name = colname
    return s


def _infer_column(df: pd.DataFrame, candidates: List[str], provided: Optional[str], label: str) -> str:
    if provided:
        if provided in df.columns:
            return provided
        raise ValueError(f"Provided {label} column '{provided}' not found in CSV. Available: {list(df.columns)}")
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not infer {label} column. Looked for any of: {candidates}. Available columns: {list(df.columns)}"
    )


def _stratified_indices(groups: pd.Series, n: int, rng: np.random.Generator) -> np.ndarray:
    """Return indices for a stratified sample across group labels, proportional to group sizes.

    After the initial proportional allocation, if the result is short of ``n`` due to rounding
    or small groups, top-up by assigning remaining slots to groups that still have capacity.
    """
    if n <= 0:
        return np.array([], dtype=int)
    if n >= len(groups):
        return groups.index.to_numpy()
    counts = groups.value_counts()
    props = counts / counts.sum()
    per_group = (props * n).round().astype(int)
    # ensure at least 1 if possible
    per_group = per_group.clip(lower=1)
    # adjust down to n if needed
    while per_group.sum() > n:
        candidates = per_group[per_group > 1].index
        if len(candidates) == 0:
            break
        g = rng.choice(candidates)
        per_group[g] -= 1
    # initial sampling
    sel_idx: List[int] = []
    selected_counts: Dict[object, int] = {}
    selected_per_group: Dict[object, set] = {}
    for g, k in per_group.items():
        idx = groups[groups == g].index.to_numpy()
        k_eff = min(k, len(idx))
        if k_eff > 0:
            sel = rng.choice(idx, size=k_eff, replace=False)
            chosen = sel.tolist()
            sel_idx.extend(chosen)
            selected_counts[g] = k_eff
            selected_per_group[g] = set(map(int, chosen))
        else:
            selected_counts[g] = 0
            selected_per_group[g] = set()
    # top-up until reaching n or saturating capacity
    def _cap(g):
        return int(counts.get(g, 0)) - int(selected_counts.get(g, 0))
    while len(sel_idx) < n:
        candidates = [g for g in counts.index if _cap(g) > 0]
        if not candidates:
            break
        g = rng.choice(candidates)
        all_idx = groups[groups == g].index.to_numpy()
        remaining = [i for i in all_idx if i not in selected_per_group[g]]
        if not remaining:
            counts = counts.drop(index=g)
            continue
        pick = int(rng.choice(remaining))
        sel_idx.append(pick)
        selected_per_group[g].add(pick)
        selected_counts[g] = selected_counts.get(g, 0) + 1
    return np.array(sel_idx, dtype=int)


# -------- Feasibility checks --------

@dataclass
class OODSiteConfig:
    min_task_sites: int = 3     # each task from held-out site should exist in at least this many other sites
    min_task_samples_other: int = 10  # minimum samples for that task in other sites (total)
    min_task_coverage_frac: float = 0.8  # fraction of site tasks that meet the above
    # Optional additional guards to avoid picking tiny sites as OOD candidates
    min_site_tasks: int = 0  # minimum distinct tasks within site (0 = disabled)
    min_site_samples: int = 0  # minimum total samples within site (0 = disabled)


@dataclass
class OODTaskConfig:
    min_sites: int = 3
    min_participants: int = 5
    min_total_samples: int = 50


@dataclass
class OODParticipantConfig:
    min_task_overlap_frac: float = 0.7  # fraction of participant's tasks seen in others
    min_site_overlap_frac: float = 0.7  # fraction of participant's sites seen in others
    min_total_samples: int = 20
    # Optional: ensure participants are not degenerate
    min_unique_tasks_for_part: int = 0
    min_unique_sites_for_part: int = 0


def feasibility_site(df: pd.DataFrame, task_col: str, site_col: str, cfg: OODSiteConfig) -> pd.DataFrame:
    # Precompute aggregates
    task_site = df.groupby([task_col, site_col]).size().rename('samples').reset_index()
    task_total_by_site = task_site.groupby(site_col)['samples'].sum().rename('site_samples')
    # For each site, compute coverage
    rows = []
    sites = task_site[site_col].unique().tolist()
    for s in sites:
        sub = task_site[task_site[site_col] == s]
        tasks_s = sub[task_col].unique().tolist()
        covered_tasks = 0
        covered_samples = 0
        total_samples_site = int(task_total_by_site.loc[s]) if s in task_total_by_site.index else 0
        for t in tasks_s:
            others = task_site[(task_site[task_col] == t) & (task_site[site_col] != s)]
            distinct_sites = others[site_col].nunique()
            other_samples = int(others['samples'].sum()) if not others.empty else 0
            # mark covered
            if distinct_sites >= cfg.min_task_sites and other_samples >= cfg.min_task_samples_other:
                covered_tasks += 1
                # count samples of this task within site toward covered sample mass
                covered_samples += int(sub[sub[task_col] == t]['samples'].sum())
        total_tasks_site = len(tasks_s)
        coverage_frac = covered_tasks / total_tasks_site if total_tasks_site else np.nan
        covered_sample_frac = covered_samples / total_samples_site if total_samples_site else np.nan
        meets_cov = bool(coverage_frac >= cfg.min_task_coverage_frac if not np.isnan(coverage_frac) else False)
        meets_size = (total_tasks_site >= cfg.min_site_tasks) and (total_samples_site >= cfg.min_site_samples)
        rows.append({
            site_col: s,
            'site_tasks': total_tasks_site,
            'site_samples': total_samples_site,
            'covered_tasks': covered_tasks,
            'coverage_frac': coverage_frac,
            'covered_sample_frac': covered_sample_frac,
            'meets_req': bool(meets_cov and meets_size),
        })
    return pd.DataFrame(rows).sort_values(['meets_req', 'coverage_frac', 'covered_sample_frac', 'site_samples'], ascending=[False, False, False, False])


def feasibility_task(df: pd.DataFrame, task_col: str, site_col: str, participant_col: str, cfg: OODTaskConfig) -> pd.DataFrame:
    g = df.groupby(task_col)
    rows = []
    for t, sub in g:
        n_sites = sub[site_col].nunique()
        n_parts = sub[participant_col].nunique()
        total = len(sub)
        ok = (n_sites >= cfg.min_sites) and (n_parts >= cfg.min_participants) and (total >= cfg.min_total_samples)
        rows.append({task_col: t, 'sites': n_sites, 'participants': n_parts, 'samples': total, 'meets_req': bool(ok)})
    return pd.DataFrame(rows).sort_values(['meets_req', 'samples', 'sites', 'participants'], ascending=[False, False, False, False])


def feasibility_participant(df: pd.DataFrame, task_col: str, site_col: str, participant_col: str, cfg: OODParticipantConfig) -> pd.DataFrame:
    rows = []
    for p, sub in df.groupby(participant_col):
        tasks_p = set(sub[task_col].unique())
        sites_p = set(sub[site_col].unique())
        others = df[df[participant_col] != p]
        tasks_others = set(others[task_col].unique())
        sites_others = set(others[site_col].unique())
        task_overlap_frac = (len(tasks_p & tasks_others) / len(tasks_p)) if tasks_p else np.nan
        site_overlap_frac = (len(sites_p & sites_others) / len(sites_p)) if sites_p else np.nan
        total = len(sub)
        n_tasks = len(tasks_p)
        n_sites = len(sites_p)
        ok = (
            (total >= cfg.min_total_samples)
            and (n_tasks >= cfg.min_unique_tasks_for_part)
            and (n_sites >= cfg.min_unique_sites_for_part)
            and (task_overlap_frac >= cfg.min_task_overlap_frac if not np.isnan(task_overlap_frac) else False)
            and (site_overlap_frac >= cfg.min_site_overlap_frac if not np.isnan(site_overlap_frac) else False)
        )
        rows.append({participant_col: p, 'samples': total, 'unique_tasks': n_tasks, 'unique_sites': n_sites, 'task_overlap_frac': task_overlap_frac, 'site_overlap_frac': site_overlap_frac, 'meets_req': bool(ok)})
    return pd.DataFrame(rows).sort_values(['meets_req', 'samples', 'task_overlap_frac', 'site_overlap_frac'], ascending=[False, False, False, False])


# -------- Split planning --------

S_TRAIN = 'TRAIN'
S_VAL = 'VAL'
S_IID = 'TEST_IID'
S_OOD_SITE = 'TEST_OOD_SITE'
S_OOD_TASK = 'TEST_OOD_TASK'
S_OOD_PART = 'TEST_OOD_PARTICIPANT'


def plan_splits(
    df: pd.DataFrame,
    task_col: str,
    site_col: str,
    participant_col: str,
    feas_site: pd.DataFrame,
    feas_task: pd.DataFrame,
    feas_part: pd.DataFrame,
    fs_cfg: OODSiteConfig = OODSiteConfig(),
    ft_cfg: OODTaskConfig = OODTaskConfig(),
    fp_cfg: OODParticipantConfig = OODParticipantConfig(),
    iid_frac: float = 0.10,
    val_frac: float = 0.10,
    iid_mode: str = 'ratio',  # 'ratio' or 'fraction'
    iid_ratio_train: int = 7,
    iid_ratio_val: int = 1,
    iid_ratio_test: int = 2,
    seed: int = 42,
    n_ood_tasks: int = 1,
    n_ood_sites: int = 1,
    n_ood_parts: int = 1,
    cap_ood_to_iid: bool = True,
    group_series: Optional[pd.Series] = None,
    ood_task_max_samples: int = 0,
    ood_site_max_samples: int = 0,
    ood_part_max_samples: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    rng = np.random.default_rng(seed)
    df = df.copy()

    # Priority: Task > Site > Participant
    # 1) OOD-Task: pick top-K tasks meeting requirements
    task_candidates = feas_task[feas_task['meets_req']].copy()
    if ood_task_max_samples and ood_task_max_samples > 0:
        tmp = task_candidates[task_candidates['samples'] <= ood_task_max_samples]
        if not tmp.empty:
            task_candidates = tmp
    task_pick_df = task_candidates.sort_values(['samples', 'sites', 'participants'], ascending=[False, False, False]).head(max(0, int(n_ood_tasks)))
    ood_tasks = task_pick_df[task_col].astype(str).tolist()

    # 2) OOD-Site: recompute feasibility on remaining after removing selected OOD tasks
    rem_for_site = df[~df[task_col].astype(str).isin(ood_tasks)] if len(ood_tasks) > 0 else df
    feas_site_rem = feasibility_site(rem_for_site, task_col, site_col, fs_cfg)
    site_candidates = feas_site_rem[feas_site_rem['meets_req']].copy()
    if ood_site_max_samples and ood_site_max_samples > 0 and 'site_samples' in site_candidates.columns:
        tmp = site_candidates[site_candidates['site_samples'] <= ood_site_max_samples]
        if not tmp.empty:
            site_candidates = tmp
    site_pick_df = site_candidates.sort_values(['coverage_frac', 'covered_sample_frac', 'site_samples'], ascending=[False, False, False]).head(max(0, int(n_ood_sites)))
    ood_sites = site_pick_df[site_col].astype(str).tolist()

    # 3) OOD-Participant: recompute feasibility after removing OOD tasks and OOD sites
    rem_for_part = rem_for_site[~rem_for_site[site_col].astype(str).isin(ood_sites)] if len(ood_sites) > 0 else rem_for_site
    part_candidates = feasibility_participant(rem_for_part, task_col, site_col, participant_col, fp_cfg)
    part_candidates = part_candidates[part_candidates['meets_req']]
    if ood_part_max_samples and ood_part_max_samples > 0:
        tmp = part_candidates[part_candidates['samples'] <= ood_part_max_samples]
        if not tmp.empty:
            part_candidates = tmp
    part_pick_df = part_candidates.sort_values(['samples', 'task_overlap_frac', 'site_overlap_frac'], ascending=[False, False, False]).head(max(0, int(n_ood_parts)))
    ood_parts = part_pick_df[participant_col].astype(str).tolist()

    # Assign splits with priority: TASK > SITE > PART
    split = pd.Series(S_TRAIN, index=df.index, name='split')
    if len(ood_tasks) > 0:
        split[df[task_col].astype(str).isin(ood_tasks)] = S_OOD_TASK
    if len(ood_sites) > 0:
        split[(df[site_col].astype(str).isin(ood_sites)) & (split == S_TRAIN)] = S_OOD_SITE
    if len(ood_parts) > 0:
        split[(df[participant_col].astype(str).isin(ood_parts)) & (split == S_TRAIN)] = S_OOD_PART

    # IID + VAL from remaining TRAIN pool
    train_pool_idx = split[split == S_TRAIN].index
    n_pool = len(train_pool_idx)
    if iid_mode == 'ratio':
        r_train, r_val, r_iid = max(0, int(iid_ratio_train)), max(0, int(iid_ratio_val)), max(0, int(iid_ratio_test))
        total_r = r_train + r_val + r_iid
        if total_r <= 0:
            r_train, r_val, r_iid = 7, 1, 2
            total_r = 10
        tgt_iid = int(round(n_pool * (r_iid / total_r)))
        tgt_val = int(round(n_pool * (r_val / total_r)))
    else:
        # fraction mode: base on remaining TRAIN pool size (after OOD assignment)
        tgt_iid = int(round(n_pool * iid_frac))
        tgt_val = int(round(n_pool * val_frac))
    if group_series is not None:
        # group-level sampling by take
        gser = group_series.loc[train_pool_idx]
        grp_to_idx: Dict[object, List[int]] = {}
        for idx, g in zip(gser.index, gser.values):
            grp_to_idx.setdefault(g, []).append(int(idx))
        grp_order = list(grp_to_idx.keys())
        rng.shuffle(grp_order)
        # IID groups
        iid_groups: List[object] = []
        total = 0
        for g in grp_order:
            sz = len(grp_to_idx[g])
            if total + sz <= tgt_iid:
                iid_groups.append(g)
                total += sz
            if total >= tgt_iid:
                break
        iid_idx = []
        for g in iid_groups:
            iid_idx.extend(grp_to_idx[g])
        split.loc[iid_idx] = S_IID
        # VAL groups from remaining
        rem_groups = [g for g in grp_order if g not in iid_groups]
        val_groups: List[object] = []
        total = 0
        for g in rem_groups:
            sz = len(grp_to_idx[g])
            if total + sz <= tgt_val:
                val_groups.append(g)
                total += sz
            if total >= tgt_val:
                break
        val_idx = []
        for g in val_groups:
            val_idx.extend(grp_to_idx[g])
        split.loc[val_idx] = S_VAL
    else:
        # sample-level stratified fallback
        tgt_iid = min(tgt_iid, n_pool)
        iid_idx = _stratified_indices(pd.Series(
            df.loc[train_pool_idx, task_col].astype(str) + '||' + df.loc[train_pool_idx, site_col].astype(str),
            index=train_pool_idx
        ), tgt_iid, rng)
        split.loc[iid_idx] = S_IID
        rem_after_iid = split[split == S_TRAIN].index
        tgt_val = min(tgt_val, len(rem_after_iid))
        val_idx = _stratified_indices(pd.Series(
            df.loc[rem_after_iid, task_col].astype(str) + '||' + df.loc[rem_after_iid, site_col].astype(str),
            index=rem_after_iid
        ), tgt_val, rng)
        split.loc[val_idx] = S_VAL

    df_out = df.copy()
    df_out['split'] = split

    # NOTE: OOD capping is handled in main() after optional group unification

    # Summaries
    summary = df_out['split'].value_counts().rename_axis('split').reset_index(name='samples')
    # Picks: provide both list-based and legacy singletons (first element or None)
    picks: Dict[str, object] = {
        'ood_tasks': ood_tasks,
        'ood_sites': ood_sites,
        'ood_participants': ood_parts,
        'ood_task': (ood_tasks[0] if len(ood_tasks) > 0 else None),
        'ood_site': (ood_sites[0] if len(ood_sites) > 0 else None),
        'ood_participant': (ood_parts[0] if len(ood_parts) > 0 else None),
    }
    return df_out, picks


def _cap_ood_splits(
    df_out: pd.DataFrame,
    task_col: str,
    site_col: str,
    group_col: Optional[str],
    seed: int,
    absolute_cap: int = 0,
) -> pd.DataFrame:
    """Cap each OOD split size to at most TEST_IID size.

    If ``group_col`` is provided and present in ``df_out``, cap by whole groups to avoid
    splitting groups (e.g., take) across kept/dropped samples. Otherwise, perform stratified
    per-sample capping by ``task||site``.
    """
    rng = np.random.default_rng(seed)
    out = df_out.copy()
    n_iid_final = int((out['split'] == S_IID).sum())
    if n_iid_final <= 0:
        return out
    for ood_label in [S_OOD_TASK, S_OOD_SITE, S_OOD_PART]:
        mask = out['split'] == ood_label
        n_ood = int(mask.sum())
        target_base = n_iid_final
        if absolute_cap and absolute_cap > 0:
            target_base = min(target_base, int(absolute_cap))
        if n_ood <= target_base:
            continue
        idx_all = out[mask].index
        target = target_base
        if group_col and (group_col in out.columns):
            # cap by whole groups
            sub = out.loc[idx_all, [group_col]]
            grp_sizes = sub[group_col].value_counts()
            groups = grp_sizes.index.to_numpy().copy()
            rng.shuffle(groups)
            keep_groups: List[object] = []
            total = 0
            for g in groups:
                sz = int(grp_sizes.loc[g])
                if total + sz <= target:
                    keep_groups.append(g)
                    total += sz
                if total >= target:
                    break
            keep_idx = out.index[(mask) & (out[group_col].isin(keep_groups))]
            drop_idx = idx_all.difference(keep_idx)
        else:
            # stratify by task||site per sample
            groups = pd.Series(
                out.loc[idx_all, task_col].astype(str) + '||' + out.loc[idx_all, site_col].astype(str),
                index=idx_all
            )
            keep_idx = pd.Index(_stratified_indices(groups, target, rng))
            drop_idx = idx_all.difference(keep_idx)
        if len(drop_idx) > 0:
            out = out.drop(index=drop_idx)
    return out


def _unify_splits_by_group(df_with_split: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Ensure all rows of the same group (e.g., take/video) share one split to avoid leakage.
    Priority order: OOD-Task > OOD-Site > OOD-Participant > TEST_IID > VAL > TRAIN.
    """
    priority = [S_OOD_TASK, S_OOD_SITE, S_OOD_PART, S_IID, S_VAL, S_TRAIN]
    # Compute group -> highest-priority split present in that group
    pr_map = {s: i for i, s in enumerate(priority)}
    grp = df_with_split[[group_col, 'split']].copy()
    # Pick the split with minimal priority index per group
    grp['pri'] = grp['split'].map(pr_map).fillna(len(priority)).astype(int)
    best = grp.sort_values('pri').drop_duplicates(subset=[group_col], keep='first')
    # Map back to rows
    best_map = dict(zip(best[group_col], best['split']))
    out = df_with_split.copy()
    out['split'] = out[group_col].map(best_map)
    return out


def post_split_checks(df_out: pd.DataFrame, task_col: str, site_col: str, participant_col: str, outdir: str) -> None:
    checks_dir = os.path.join(outdir, 'post_split_checks')
    _ensure_dir(checks_dir)

    # OOD-Site: For held-out site, verify tasks exist widely in TRAIN,
    # ignoring any tasks that were selected as OOD-Task.
    if (df_out['split'] == S_OOD_SITE).any():
        held_sites = df_out.loc[df_out['split'] == S_OOD_SITE, site_col].unique().tolist()
        train = df_out[df_out['split'] == S_TRAIN]
        ood_task_set = set(df_out.loc[df_out['split'] == S_OOD_TASK, task_col].unique().tolist())
        for s in held_sites:
            tasks = [t for t in df_out.loc[df_out[site_col] == s, task_col].unique().tolist() if t not in ood_task_set]
            rows = []
            for t in tasks:
                sites_in_train = train[train[task_col] == t][site_col].nunique()
                samples_in_train = train[train[task_col] == t].shape[0]
                rows.append({task_col: t, 'sites_in_train': sites_in_train, 'samples_in_train': samples_in_train})
            pd.DataFrame(rows).sort_values(['sites_in_train', 'samples_in_train'], ascending=[False, False]).to_csv(
                os.path.join(checks_dir, f'ood_site_tasks_train_coverage__{s}.csv'), index=False
            )

    # OOD-Task: Recap feasibility-like stats over the full dataset (since held-out entirely)
    if (df_out['split'] == S_OOD_TASK).any():
        held_tasks = df_out.loc[df_out['split'] == S_OOD_TASK, task_col].unique().tolist()
        for t in held_tasks:
            sub = df_out[df_out[task_col] == t]
            n_sites = sub[site_col].nunique()
            n_parts = sub[participant_col].nunique()
            total = len(sub)
            pd.DataFrame([
                {task_col: t, 'sites': n_sites, 'participants': n_parts, 'samples': total}
            ]).to_csv(os.path.join(checks_dir, f'ood_task_feasibility_recap__{t}.csv'), index=False)

    # OOD-Participant: For held-out participant, check overlap of their tasks/sites in TRAIN by others
    if (df_out['split'] == S_OOD_PART).any():
        held_parts = df_out.loc[df_out['split'] == S_OOD_PART, participant_col].unique().tolist()
        train = df_out[df_out['split'] == S_TRAIN]
        for p in held_parts:
            tasks_p = set(df_out.loc[df_out[participant_col] == p, task_col].unique())
            sites_p = set(df_out.loc[df_out[participant_col] == p, site_col].unique())
            tasks_others = set(train[task_col].unique())
            sites_others = set(train[site_col].unique())
            with open(os.path.join(checks_dir, f'ood_participant_overlap__{p}.txt'), 'w') as f:
                f.write(f'task_overlap_frac={len(tasks_p & tasks_others)/len(tasks_p) if tasks_p else np.nan}\n')
                f.write(f'site_overlap_frac={len(sites_p & sites_others)/len(sites_p) if sites_p else np.nan}\n')


def subsample_after_split(df_out: pd.DataFrame, task_col: str, site_col: str, frac: float, seed: int) -> pd.DataFrame:
    if not (0 < frac <= 1):
        return df_out
    rng = np.random.default_rng(seed)
    result = []
    for split_name, sub in df_out.groupby('split'):
        n = int(round(len(sub) * frac))
        if n >= len(sub):
            result.append(sub)
            continue
        # stratify by (task, site)
        key = sub[task_col].astype(str) + '||' + sub[site_col].astype(str)
        idx = _stratified_indices(pd.Series(key.values, index=sub.index), n, rng)
        result.append(sub.loc[idx])
    return pd.concat(result, axis=0).sort_index()


# -------- Main --------

def _draw_site_task_heatmap(pivot: pd.DataFrame, outpath: str, highlight_site: Optional[str] = None, highlight_task: Optional[str] = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Choose plotting backend
    try:
        import seaborn as sns  # type: ignore
    except Exception:
        sns = None

    # Ensure 2D
    if pivot.empty:
        fig = plt.figure(figsize=(12, 12))
        plt.title("Site × Task heatmap (empty)")
        plt.savefig(outpath, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return

    fig_w = max(8, 0.4 * pivot.shape[1])
    fig_h = max(6, 0.3 * pivot.shape[0])
    fig_w = 12
    fig_h = 12
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if sns is not None:
        sns.heatmap(pivot, cmap="YlGnBu", ax=ax)
    else:
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns.astype(str), rotation=90)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index.astype(str))
    ax.set_title("Site × Task (samples)")

    # Highlight selected OOD site (row) and task (column)
    n_rows, n_cols = pivot.shape
    if highlight_site is not None and highlight_site in pivot.index:
        r = list(pivot.index).index(highlight_site)
        rect = patches.Rectangle((-0.5, r - 0.5), n_cols, 1, linewidth=2.0, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(n_cols - 0.5, r - 0.4, "OOD-Site", color='red', fontsize=10, ha='right', va='bottom')
    if highlight_task is not None and highlight_task in pivot.columns:
        c = list(pivot.columns).index(highlight_task)
        rect = patches.Rectangle((c - 0.5, -0.5), 1, n_rows, linewidth=2.0, edgecolor='orange', facecolor='none')
        ax.add_patch(rect)
        ax.text(c - 0.4, -0.5, "OOD-Task", color='orange', fontsize=10, ha='left', va='top', rotation=90)

    plt.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _plot_split_distributions(
    splits_df: pd.DataFrame,
    task_col: str,
    site_col: str,
    participant_col: str,
    outdir: str,
    top_k: int = 30,
) -> None:
    """Create bar plots of per-split frequency distributions for task/site/participant.

    Writes PNGs under ``outdir/distributions`` and corresponding CSVs of full frequencies.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    dist_dir = os.path.join(outdir, 'distributions')
    _ensure_dir(dist_dir)

    for split_name, sub in splits_df.groupby('split'):
        freqs = {
            'task': sub[task_col].value_counts(),
            'site': sub[site_col].value_counts(),
            'participant': sub[participant_col].value_counts(),
        }
        # Save full frequency tables
        for key, vc in freqs.items():
            vc.rename('count').to_frame().to_csv(os.path.join(dist_dir, f'{split_name}__{key}_freq.csv'))
        # Plot top-k bars for visibility
        for key, vc in freqs.items():
            top = vc.head(top_k)
            if top.empty:
                continue
            fig, ax = plt.subplots(figsize=(max(6, min(18, 0.3 * len(top))), 6))
            top.plot(kind='bar', ax=ax)
            ax.set_title(f"{split_name}: Top-{min(top_k, len(vc))} {key}s")
            ax.set_ylabel('count')
            ax.set_xlabel(key)
            plt.xticks(rotation=60, ha='right')
            plt.tight_layout()
            fig.savefig(os.path.join(dist_dir, f'{split_name}__{key}_top{top_k}.png'), dpi=200, bbox_inches='tight')
            plt.close(fig)


def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p.astype(float)
    q = q.astype(float)
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum <= 0 and q_sum <= 0:
        return 0.0
    if p_sum <= 0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / (p_sum + eps)
    if q_sum <= 0:
        q = np.ones_like(q) / len(q)
    else:
        q = q / (q_sum + eps)
    m = 0.5 * (p + q)
    def _kl(a, b):
        a = np.clip(a, eps, None)
        b = np.clip(b, eps, None)
        return float(np.sum(a * np.log(a / b)))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _cosine_similarity(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    num = float(np.dot(p, q))
    den = float(np.linalg.norm(p) * np.linalg.norm(q))
    if den <= eps:
        return 0.0
    return num / den


def _plot_similarity_heatmap(mat_df: pd.DataFrame, outpath: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns  # type: ignore
    except Exception:
        sns = None
    fig, ax = plt.subplots(figsize=(10, 8))
    if sns is not None:
        sns.heatmap(mat_df, annot=False, cmap="YlGnBu", ax=ax)
    else:
        im = ax.imshow(mat_df.values, aspect="auto", cmap="YlGnBu")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(mat_df.shape[1]))
        ax.set_xticklabels(mat_df.columns.astype(str), rotation=90)
        ax.set_yticks(range(mat_df.shape[0]))
        ax.set_yticklabels(mat_df.index.astype(str))
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _analyze_split_similarity(
    splits_df: pd.DataFrame,
    task_col: str,
    site_col: str,
    participant_col: str,
    key: str,
    outdir: str,
) -> None:
    """Compute pairwise JS divergence, cosine similarity, and Jaccard overlap for a field.

    key: one of 'task', 'site', 'participant'
    """
    import os
    dist_dir = os.path.join(outdir, 'distributions')
    _ensure_dir(dist_dir)
    # Load freq tables from already saved CSVs if present, otherwise compute
    name_map = {
        'task': None,
        'site': None,
        'participant': None,
    }
    assert key in name_map
    # Build frequency tables per split
    freq_by_split: Dict[str, pd.Series] = {}
    for split_name, sub in splits_df.groupby('split'):
        if key == 'task':
            vc = sub[task_col].value_counts()
        elif key == 'site':
            vc = sub[site_col].value_counts()
        else:
            vc = sub[participant_col].value_counts()
        freq_by_split[str(split_name)] = vc
    # Union of categories
    categories = sorted(set().union(*[set(s.index.astype(str)) for s in freq_by_split.values()]))
    splits = sorted(freq_by_split.keys())
    # Build count matrix
    mat = np.zeros((len(splits), len(categories)), dtype=float)
    for i, sname in enumerate(splits):
        vc = freq_by_split[sname]
        for j, cat in enumerate(categories):
            mat[i, j] = float(vc.get(cat, 0))
    # Pairwise metrics
    n = len(splits)
    js = np.zeros((n, n), dtype=float)
    cos = np.zeros((n, n), dtype=float)
    jac = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            js[i, j] = _js_divergence(mat[i], mat[j])
            cos[i, j] = _cosine_similarity(mat[i], mat[j])
            ai = set([categories[k] for k, v in enumerate(mat[i]) if v > 0])
            aj = set([categories[k] for k, v in enumerate(mat[j]) if v > 0])
            un = len(ai | aj)
            inter = len(ai & aj)
            jac[i, j] = (inter / un) if un > 0 else 1.0
    # DataFrames
    js_df = pd.DataFrame(js, index=splits, columns=splits)
    cos_df = pd.DataFrame(cos, index=splits, columns=splits)
    jac_df = pd.DataFrame(jac, index=splits, columns=splits)
    # Save CSVs
    js_df.to_csv(os.path.join(dist_dir, f'similarity_{key}_js.csv'))
    cos_df.to_csv(os.path.join(dist_dir, f'similarity_{key}_cosine.csv'))
    jac_df.to_csv(os.path.join(dist_dir, f'similarity_{key}_jaccard.csv'))
    # Heatmaps
    _plot_similarity_heatmap(js_df, os.path.join(dist_dir, f'similarity_{key}_js.png'), f'{key} JS divergence (lower=more similar)')
    _plot_similarity_heatmap(cos_df, os.path.join(dist_dir, f'similarity_{key}_cosine.png'), f'{key} Cosine similarity (higher=more similar)')
    _plot_similarity_heatmap(jac_df, os.path.join(dist_dir, f'similarity_{key}_jaccard.png'), f'{key} Jaccard overlap (support)')


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='OOD feasibility and split planner')
    ap.add_argument('--csv', default="/mnt/data2/jiali/GLC/output/aria_video_analysis_v2/aria_video_clips_for_benchmark.csv", help='Path to input CSV')
    ap.add_argument('--outdir', default='explore/ood_group_by_take', help='Output directory (default: explore/ood)')
    ap.add_argument('--task-col', dest='task_col', default=None)
    ap.add_argument('--site-col', dest='site_col', default=None)
    ap.add_argument('--participant-col', dest='participant_col', default=None)

    # OOD constraints
    ap.add_argument('--ood-site-min-task-sites', type=int, default=3)
    ap.add_argument('--ood-site-min-task-samples', type=int, default=10)
    ap.add_argument('--ood-site-min-coverage-frac', type=float, default=0.8)
    ap.add_argument('--ood-site-min-site-tasks', type=int, default=0, help='Min distinct tasks within a site to be considered for OOD-Site (0 = disabled)')
    ap.add_argument('--ood-site-min-site-samples', type=int, default=0, help='Min total samples within a site to be considered for OOD-Site (0 = disabled)')

    ap.add_argument('--ood-task-min-sites', type=int, default=3)
    ap.add_argument('--ood-task-min-participants', type=int, default=5)
    ap.add_argument('--ood-task-min-samples', type=int, default=50)

    ap.add_argument('--ood-part-min-total-samples', type=int, default=20)
    ap.add_argument('--ood-part-min-task-overlap-frac', type=float, default=0.7)
    ap.add_argument('--ood-part-min-site-overlap-frac', type=float, default=0.7)
    ap.add_argument('--ood-part-min-unique-tasks', type=int, default=0, help='Min unique tasks for a participant (0 = disabled)')
    ap.add_argument('--ood-part-min-unique-sites', type=int, default=0, help='Min unique sites for a participant (0 = disabled)')

    # Split sizes
    ap.add_argument('--iid-frac', type=float, default=0.10)
    ap.add_argument('--val-frac', type=float, default=0.10)
    ap.add_argument('--seed', type=int, default=42)

    # IID split mode
    ap.add_argument('--iid-mode', choices=['ratio', 'fraction'], default='ratio', help='How to split remaining pool into TRAIN/VAL/TEST_IID (default: ratio)')
    ap.add_argument('--iid-ratio-train', type=int, default=7, help='Train ratio when iid-mode=ratio (default: 7)')
    ap.add_argument('--iid-ratio-val', type=int, default=1, help='Val ratio when iid-mode=ratio (default: 1)')
    ap.add_argument('--iid-ratio-test', type=int, default=2, help='Test IID ratio when iid-mode=ratio (default: 2)')

    # How many OOD items to pick (priority: task > site > participant)
    ap.add_argument('--num-ood-tasks', type=int, default=1, help='Number of OOD tasks to select (default: 1)')
    ap.add_argument('--num-ood-sites', type=int, default=1, help='Number of OOD sites to select (default: 1)')
    ap.add_argument('--num-ood-participants', type=int, default=1, help='Number of OOD participants to select (default: 1)')
    # OOD size preferences (optional)
    ap.add_argument('--ood-task-max-samples', type=int, default=0, help='Prefer OOD tasks with <= this many samples (0 = no preference)')
    ap.add_argument('--ood-site-max-samples', type=int, default=0, help='Prefer OOD sites with <= this many total site samples (0 = no preference)')
    ap.add_argument('--ood-part-max-samples', type=int, default=0, help='Prefer OOD participants with <= this many samples (0 = no preference)')

    # Subsampling
    ap.add_argument('--subsample-after', type=float, default=1.0, help='Fraction to keep after split (0-1]. e.g., 0.3 = 30%')
    ap.add_argument('--min-train-samples', type=int, default=0, help='Minimum TRAIN sample count after grouping; backfill from IID by takes if needed')
    ap.add_argument('--min-val-samples', type=int, default=0, help='Minimum VAL sample count after grouping; backfill from IID by takes if needed')
    ap.add_argument('--cap-ood-to-test-iid', action='store_true', default=True, help='Cap each OOD split size to at most TEST_IID size (default: enabled)')
    ap.add_argument('--no-cap-ood-to-test-iid', dest='cap_ood_to_test_iid', action='store_false', help='Disable capping OOD splits to TEST_IID size')
    ap.add_argument('--cap-ood-absolute', type=int, default=0, help='Absolute max per OOD split (0 = disabled). Applied after grouping.')
    ap.add_argument('--group-by-take', action='store_true', default=True, help='Unify splits by take/video to avoid leakage (default: enabled)')
    ap.add_argument('--no-group-by-take', dest='group_by_take', action='store_false', help='Disable group-by-take unification')
    ap.add_argument('--group-col', default=None, help='Column name to group by (default: auto: take_name or derived from video_path)')

    args = ap.parse_args(argv)

    if not os.path.isfile(args.csv):
        print(f'ERROR: CSV not found: {args.csv}', file=sys.stderr)
        return 2

    df = pd.read_csv(args.csv)
    if df.empty:
        print('ERROR: CSV has no rows', file=sys.stderr)
        return 2

    # Columns
    task_col = _infer_column(df, ["task_name", "parent_task_name", "task", "task_id", "parent_task_id"], args.task_col, 'task')
    # prefer names over IDs
    if task_col in {"task_id", "parent_task_id"}:
        for c in ["task_name", "parent_task_name", "task"]:
            if c in df.columns:
                task_col = c
                break
    site_col = _infer_column(df, ["site_id", "site", "location", "location_id", "venue", "scene_id", "university_name"], args.site_col, 'site')
    participant_col = _infer_column(df, ["participant_id", "participant_uid", "subject_id", "person_id", "user_id", "participant", "subject"], args.participant_col, 'participant')

    outdir = args.outdir
    _ensure_dir(outdir)

    # Feasibility tables
    fs_cfg = OODSiteConfig(
        min_task_sites=args.ood_site_min_task_sites,
        min_task_samples_other=args.ood_site_min_task_samples,
        min_task_coverage_frac=args.ood_site_min_coverage_frac,
        min_site_tasks=args.ood_site_min_site_tasks,
        min_site_samples=args.ood_site_min_site_samples,
    )
    ft_cfg = OODTaskConfig(min_sites=args.ood_task_min_sites, min_participants=args.ood_task_min_participants, min_total_samples=args.ood_task_min_samples)
    fp_cfg = OODParticipantConfig(
        min_task_overlap_frac=args.ood_part_min_task_overlap_frac,
        min_site_overlap_frac=args.ood_part_min_site_overlap_frac,
        min_total_samples=args.ood_part_min_total_samples,
        min_unique_tasks_for_part=args.ood_part_min_unique_tasks,
        min_unique_sites_for_part=args.ood_part_min_unique_sites,
    )

    feas_site = feasibility_site(df, task_col, site_col, fs_cfg)
    feas_task = feasibility_task(df, task_col, site_col, participant_col, ft_cfg)
    feas_part = feasibility_participant(df, task_col, site_col, participant_col, fp_cfg)

    feas_site.to_csv(os.path.join(outdir, 'feasibility_site.csv'), index=False)
    feas_task.to_csv(os.path.join(outdir, 'feasibility_task.csv'), index=False)
    feas_part.to_csv(os.path.join(outdir, 'feasibility_participant.csv'), index=False)

    # Determine group column early and build series for group-aware sampling
    group_col = None
    group_series = None
    if args.group_by_take:
        group_col = args.group_col
        if group_col is None:
            if 'take_name' in df.columns:
                group_col = 'take_name'
            elif 'video_path' in df.columns:
                def _derive_take(p: str) -> str:
                    try:
                        parts = str(p).split('/takes/')
                        if len(parts) >= 2:
                            tail = parts[1]
                            return tail.split('/')[0]
                    except Exception:
                        pass
                    return str(p)
                group_series = df['video_path'].map(_derive_take)
                group_col = '_take_derived'
            else:
                group_col = None
        if group_series is None and group_col is not None and group_col in df.columns:
            group_series = df[group_col]

    # Plan splits using group-aware IID/VAL sampling
    splits_df, picks = plan_splits(
        df, task_col, site_col, participant_col,
        feas_site, feas_task, feas_part,
        fs_cfg=fs_cfg, ft_cfg=ft_cfg, fp_cfg=fp_cfg,
        iid_frac=args.iid_frac, val_frac=args.val_frac,
        iid_mode=args.iid_mode, iid_ratio_train=args.iid_ratio_train, iid_ratio_val=args.iid_ratio_val, iid_ratio_test=args.iid_ratio_test,
        seed=args.seed,
        n_ood_tasks=args.num_ood_tasks, n_ood_sites=args.num_ood_sites, n_ood_parts=args.num_ood_participants,
        cap_ood_to_iid=False,
        group_series=group_series,
        ood_task_max_samples=getattr(args, 'ood_task_max_samples', 0),
        ood_site_max_samples=getattr(args, 'ood_site_max_samples', 0),
        ood_part_max_samples=getattr(args, 'ood_part_max_samples', 0),
    )

    # Optional: unify by take/video to avoid leakage
    if args.group_by_take and group_col is not None:
        if group_series is None and group_col in df.columns:
            group_series = df[group_col]
        if group_series is not None:
            splits_df[group_col] = group_series
            splits_df = _unify_splits_by_group(splits_df, group_col)

    # Optional: cap OOD sizes after (optional) group unification
    if args.cap_ood_to_test_iid:
        splits_df = _cap_ood_splits(
            splits_df,
            task_col,
            site_col,
            group_col if (args.group_by_take and group_col and (group_col in splits_df.columns)) else None,
            args.seed,
            args.cap_ood_absolute,
        )

    # Enforce minimum TRAIN/VAL sample counts by moving whole takes back from IID
    if args.group_by_take and group_col and (group_col in splits_df.columns):
        def _enforce_min(split_name: str, min_samples: int) -> None:
            if min_samples <= 0:
                return
            cur = int((splits_df['split'] == split_name).sum())
            if cur >= min_samples:
                return
            need = min_samples - cur
            # candidate IID groups ordered by size (desc)
            iids = splits_df[splits_df['split'] == S_IID]
            grp_sizes = iids[group_col].value_counts()
            moved = 0
            for g, sz in grp_sizes.items():
                splits_df.loc[(splits_df['split'] == S_IID) & (splits_df[group_col] == g), 'split'] = split_name
                moved += int(sz)
                if cur + moved >= min_samples:
                    break
        _enforce_min(S_TRAIN, args.min_train_samples)
        _enforce_min(S_VAL, args.min_val_samples)

    splits_df.to_csv(os.path.join(outdir, 'split_assignments.csv'), index=False)
    picks_path = os.path.join(outdir, 'ood_recommendations.json')
    with open(picks_path, 'w') as f:
        json.dump(picks, f, indent=2)

    # Summaries
    summary = splits_df['split'].value_counts().rename_axis('split').reset_index(name='samples')
    summary.to_csv(os.path.join(outdir, 'split_summary.csv'), index=False)
    # Extended summary
    ext_rows = []
    for sname, sub in splits_df.groupby('split'):
        ext_rows.append({
            'split': sname,
            'samples': len(sub),
            'unique_tasks': int(sub[task_col].nunique()),
            'unique_sites': int(sub[site_col].nunique()),
            'unique_participants': int(sub[participant_col].nunique()),
        })
    pd.DataFrame(ext_rows).sort_values('samples', ascending=False).to_csv(os.path.join(outdir, 'split_summary_extended.csv'), index=False)

    # Per-take summary
    if args.group_by_take and group_col and (group_col in splits_df.columns):
        rows = []
        for sname, sub in splits_df.groupby('split'):
            take_sizes = sub[group_col].value_counts()
            if not take_sizes.empty:
                rows.append({
                    'split': sname,
                    'takes': int(take_sizes.shape[0]),
                    'take_size_mean': float(take_sizes.mean()),
                    'take_size_median': float(take_sizes.median()),
                    'take_size_min': int(take_sizes.min()),
                    'take_size_max': int(take_sizes.max()),
                })
            else:
                rows.append({'split': sname, 'takes': 0, 'take_size_mean': 0.0, 'take_size_median': 0.0, 'take_size_min': 0, 'take_size_max': 0})
        pd.DataFrame(rows).to_csv(os.path.join(outdir, 'split_takes_summary.csv'), index=False)

    # Post-split checks
    post_split_checks(splits_df, task_col, site_col, participant_col, outdir)

    # Per-split distributions (task/site/participant)
    _plot_split_distributions(splits_df, task_col, site_col, participant_col, outdir)

    # Similarity analysis across splits (task/site/participant)
    try:
        _analyze_split_similarity(splits_df, task_col, site_col, participant_col, 'task', outdir)
        _analyze_split_similarity(splits_df, task_col, site_col, participant_col, 'site', outdir)
        _analyze_split_similarity(splits_df, task_col, site_col, participant_col, 'participant', outdir)
    except Exception as e:
        # Do not fail the pipeline on plotting issues
        print(f"WARN: similarity analysis failed: {e}")

    # Sanity heatmaps with OOD highlights (overall and TRAIN-only)
    # Overall Site × Task
    overall_pivot = pd.crosstab(df[site_col], df[task_col]).astype(int)
    _draw_site_task_heatmap(
        overall_pivot,
        os.path.join(outdir, 'sanity_heatmap_site_by_task_overall.png'),
        # highlight_site=picks.get('ood_site'),
        # highlight_task=picks.get('ood_task'),
    )
    # TRAIN-only Site × Task
    train_df = splits_df[splits_df['split'] == S_TRAIN]
    if not train_df.empty:
        train_pivot = pd.crosstab(train_df[site_col], train_df[task_col]).astype(int)
        _draw_site_task_heatmap(
            train_pivot,
            os.path.join(outdir, 'sanity_heatmap_site_by_task_train.png'),
            # highlight_site=picks.get('ood_site'),
            # highlight_task=picks.get('ood_task'),
        )

    # Optional subsampling after split
    if args.subsample_after < 1.0:
        subs = subsample_after_split(splits_df, task_col, site_col, frac=args.subsample_after, seed=args.seed)
        subs.to_csv(os.path.join(outdir, 'split_assignments_subsampled.csv'), index=False)
        subs['split'].value_counts().rename_axis('split').reset_index(name='samples').to_csv(os.path.join(outdir, 'split_summary_subsampled.csv'), index=False)

    # Small README
    with open(os.path.join(outdir, 'README.txt'), 'w') as f:
        f.write('Files generated by ood_planner.py\n')
        f.write('- feasibility_site.csv\n- feasibility_task.csv\n- feasibility_participant.csv\n')
        f.write('- ood_recommendations.json\n- split_assignments.csv\n- split_summary.csv\n')
        f.write('- post_split_checks/*\n')
        f.write('- sanity_heatmap_site_by_task_overall.png\n')
        f.write('- sanity_heatmap_site_by_task_train.png\n')
        if args.subsample_after < 1.0:
            f.write('- split_assignments_subsampled.csv\n- split_summary_subsampled.csv\n')
        f.write(f"\nDetected columns: task={task_col}, site={site_col}, participant={participant_col}\n")
        f.write(f"Args: iid_frac={args.iid_frac}, val_frac={args.val_frac}, subsample_after={args.subsample_after}\n")

    print('Done. Outputs written to:', os.path.abspath(outdir))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


# python explore/ood_planner.py --num-ood-tasks 3 --num-ood-sites 2 --num-ood-participants 2 --outdir explore/ood
# python explore/ood_planner.py --num-ood-tasks 3 --num-ood-sites 2 --num-ood-participants 2 --group-by-take
