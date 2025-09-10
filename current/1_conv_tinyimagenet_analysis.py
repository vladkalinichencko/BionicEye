"""Strict analysis over Optuna + ClearML (minimal, self‑contained).

For each Optuna trial, we fetch the LAST reported ClearML metric values and plot slice-like
per-parameter charts. Objective is strictly the LAST `val_loss` (lower is better).
We also fetch the LAST `val_acc` and keep it in the dataframe for reference.

Visual style:
- Points: light blue.
- Statistical overlays (mean ± std band + mean line): pale red.
"""
# %%
import os
import optuna
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from clearml import Task
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Config
STUDY = os.environ.get('STUDY', 'tiny_imagenet_sweep')
DB = os.environ.get('DB', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sweep.db')))
PROJECT = os.environ.get('CLEARML_PROJECT', 'BionicEye')
OBJ = 'val_loss'  # main objective: strictly use LAST val_loss
ACC = 'val_acc'   # also fetch LAST val_acc for reference
SYNC_FROM_CLEARML = os.environ.get('SYNC_FROM_CLEARML', '0') in ('1', 'true', 'True')
# Optional: after fetching from ClearML, persist into Optuna user-attrs via JSON API only.
SAVE_TO_OPTUNA = os.environ.get('SAVE_TO_OPTUNA', '0') in ('1', 'true', 'True')

def _extract_last(data):
    # ClearML scalars structure: dict(series -> {x: [...], y: [...]}) or other compact forms
    if isinstance(data, dict):
        first = next(iter(data.values())) if data else None
        if isinstance(first, dict) and first.get('y'):
            return first['y'][-1]
        if isinstance(first, (list, tuple)) and first:
            cand = first[-1]
            return cand[1] if isinstance(cand, (list, tuple)) else cand
        return first
    if isinstance(data, (list, tuple)) and data:
        cand = data[-1]
        return cand[1] if isinstance(cand, (list, tuple)) else cand
    return data


def _df_from_optuna(study: optuna.Study, trials: list[optuna.trial.FrozenTrial]) -> pd.DataFrame:
    rows = []
    for t in trials:
        # read user attributes written earlier: 'last_val_loss' (required), 'last_val_acc' (optional)
        last_loss = t.user_attrs.get('last_val_loss')
        if last_loss is None:
            continue
        r = t.params.copy()
        r['_trial'] = t.number
        r['_obj'] = float(last_loss)
        last_acc = t.user_attrs.get('last_val_acc')
        if last_acc is not None:
            try:
                r['_val_acc_last'] = float(last_acc)
            except Exception:
                pass
        rows.append(r)
    df = pd.DataFrame(rows)
    return df


def _sync_last_from_clearml(trials: list[optuna.trial.FrozenTrial]) -> pd.DataFrame:
    rows = []

    # 1) Try pre-fetch all tasks by regex (fast path). If none found, fall back to exact-name lookup per trial.
    project_arg = PROJECT if PROJECT else None
    trial_to_task: dict[int, str] = {}
    try:
        all_name_pattern = r"^" + re.escape("Tiny ImageNet Sweep (optuna) / trial_") + r"\d{4}$"
        task_rows = Task.query_tasks(
            project_name=project_arg,
            task_name=all_name_pattern,
            additional_return_fields=["name"],
        ) or []
        name_re = re.compile(r"trial_(\d{4})$")
        for r in task_rows:
            tid = r.get('id') if isinstance(r, dict) else None
            name = r.get('name') if isinstance(r, dict) else None
            if not tid or not name:
                continue
            m = name_re.search(name)
            if not m:
                continue
            tn = int(m.group(1))
            trial_to_task[tn] = tid
    except Exception:
        trial_to_task = {}

    def fetch_one_exact(tn: int, tparams: dict) -> dict | None:
        expected_name = f"Tiny ImageNet Sweep (optuna) / trial_{tn:04d}"
        task = None
        try:
            # Prefer id from prefetch, otherwise resolve by exact name
            if tn in trial_to_task:
                task = Task.get_task(task_id=trial_to_task[tn])
            else:
                task = Task.get_task(project_name=project_arg, task_name=expected_name)
        except Exception:
            return None
        if task is None:
            return None
        try:
            scalars = task.get_reported_scalars()
        except Exception:
            return None
        if OBJ not in scalars:
            return None
        try:
            last_obj = _extract_last(scalars[OBJ])
            obj_val = float(last_obj)
        except Exception:
            return None
        acc_val = None
        if ACC in scalars:
            try:
                last_acc = _extract_last(scalars[ACC])
                acc_val = float(last_acc)
            except Exception:
                acc_val = None
        out = tparams.copy()
        out['_trial'] = tn
        out['_obj'] = obj_val
        if acc_val is not None:
            out['_val_acc_last'] = acc_val
        return out

    with ThreadPoolExecutor(max_workers=int(os.environ.get('CLEARML_FETCH_WORKERS', '8'))) as ex:
        futs = {}
        for t in trials:
            futs[ex.submit(fetch_one_exact, t.number, t.params.copy())] = t.number
        for f in as_completed(futs):
            res = f.result()
            if res is not None:
                rows.append(res)

    return pd.DataFrame(rows)


def main():
    storage_url = f"sqlite:///{DB}"
    # Load study
    study = optuna.load_study(study_name=STUDY, storage=storage_url)
    trials = list(study.trials)
    # Build dataframe of last metrics
    if SYNC_FROM_CLEARML:
        df = _sync_last_from_clearml(trials)
    else:
        df = _df_from_optuna(study, trials)
    if df.empty:
        return
    # Optionally persist to Optuna user-attrs (JSON field) without any direct SQL
    if SYNC_FROM_CLEARML and SAVE_TO_OPTUNA:
        number_to_id = {t.number: t._trial_id for t in trials}
        opt_storage = study._storage
        for row in df.to_dict(orient='records'):
            tn = int(row.get('_trial'))
            tid = number_to_id.get(tn)
            if tid is None:
                continue
            new_loss = float(row.get('_obj'))
            new_acc = float(row.get('_val_acc_last')) if '_val_acc_last' in row and row.get('_val_acc_last') is not None else None
            try:
                opt_storage.set_trial_user_attr(tid, 'last_val_loss', new_loss)
                if new_acc is not None:
                    opt_storage.set_trial_user_attr(tid, 'last_val_acc', new_acc)
            except Exception:
                # Ignore persistence errors silently
                pass
    params = [c for c in df.columns if not c.startswith('_')]

    # Outlier handling before plotting
    df_plot = df.copy()
    y_all = pd.to_numeric(df_plot['_obj'], errors='coerce')
    df_plot = df_plot.loc[np.isfinite(y_all)]
    y_all = df_plot['_obj'].astype(float)

    OUTLIER = os.environ.get('OUTLIER', 'mad')  # 'none' | 'mad' | 'quantile'
    removed = 0
    if OUTLIER.lower() == 'mad' and len(y_all) > 0:
        med = float(np.median(y_all))
        mad = float(np.median(np.abs(y_all - med)))
        if mad > 0:
            MAD_K = float(os.environ.get('MAD_K', '10'))
            kappa = 1.4826 * mad
            low = max(0.0, med - MAD_K * kappa)
            high = med + MAD_K * kappa
            mask = (y_all >= low) & (y_all <= high)
            removed = int((~mask).sum())
            df_plot = df_plot.loc[mask]
            y_all = df_plot['_obj'].astype(float)
        else:
            # all equal -> nothing to remove
            pass
    elif OUTLIER.lower() == 'quantile' and len(y_all) > 0:
        qbot = float(os.environ.get('BOTTOM_Q', '0.0'))
        qtop = float(os.environ.get('TOP_Q', '0.01'))
        lo = y_all.quantile(qbot)
        hi = y_all.quantile(1.0 - qtop)
        mask = (y_all >= lo) & (y_all <= hi)
        removed = int((~mask).sum())
        df_plot = df_plot.loc[mask]
        y_all = df_plot['_obj'].astype(float)

    # simple slice-like plot: one row per parameter
    n = len(params)
    fig = make_subplots(rows=n, cols=1, subplot_titles=params, vertical_spacing=0.06)
    y = y_all.values

    for i, p in enumerate(params, start=1):
        xref = 'x' if i == 1 else f'x{i}'
        yref = 'y' if i == 1 else f'y{i}'

        # Decide if numeric should be treated as categorical (low cardinality) or continuous (binning)
        is_num = pd.api.types.is_numeric_dtype(df_plot[p])
        uniq_ct = int(df_plot[p].nunique(dropna=True))
        num_as_categ_max = int(os.environ.get('NUMERIC_AS_CATEG_MAX', '20'))
        treat_as_categ = (not is_num) or (is_num and uniq_ct <= num_as_categ_max)

        custom = np.stack([df_plot['_trial'].values], axis=-1)

        if treat_as_categ:
            # Factorize categories; for numeric values keep sorted order
            if is_num:
                uniq_vals = np.array(sorted(df_plot[p].astype(float).unique()))
                idx_map = {v: k for k, v in enumerate(uniq_vals)}
                series_num = df_plot[p].astype(float)
                cats = series_num.map(idx_map).astype(int).values
                ticktext = [str(v) for v in uniq_vals]
            else:
                cats, uniq = pd.factorize(df_plot[p].astype(str))
                uniq_vals = np.array(list(uniq))
                ticktext = list(uniq_vals)

            # points with jitter
            jitter = np.random.uniform(-0.12, 0.12, size=len(cats))
            xj = cats + jitter
            cat_text = np.array(ticktext)[cats]
            fig.add_trace(
                go.Scatter(
                    x=xj,
                    y=y,
                    mode='markers',
                    marker=dict(color='rgba(102,178,255,0.9)', size=5),
                    text=cat_text,
                    customdata=custom,
                    hovertemplate=f"trial %{{customdata[0]}}<br>{p}: %{{text}}<br>val_loss: %{{y:.6g}}<extra></extra>",
                ),
                row=i, col=1,
            )

            # Per-category mean/std as full-width horizontal overlays (one red band+line per category)
            kmax = int(len(ticktext))
            x_min = -0.5
            x_max = kmax - 0.5
            fig.update_xaxes(tickmode='array', tickvals=list(range(kmax)), ticktext=ticktext, range=[x_min, x_max], row=i, col=1)
            for k in range(kmax):
                gmask = (cats == k)
                if not np.any(gmask):
                    continue
                yg = y[gmask]
                m = float(np.mean(yg))
                s = float(np.std(yg))
                # Band (mean ± std)
                fig.add_trace(
                    go.Scatter(
                        x=[x_min, x_max, x_max, x_min, x_min],
                        y=[m - s, m - s, m + s, m + s, m - s],
                        fill='toself',
                        fillcolor='rgba(255, 80, 80, 0.10)',
                        line=dict(width=0),
                        hoverinfo='skip',
                        showlegend=False,
                    ),
                    row=i, col=1,
                )
                # Mean line
                fig.add_trace(
                    go.Scatter(
                        x=[x_min, x_max],
                        y=[m, m],
                        mode='lines',
                        line=dict(color='rgba(220,30,30,0.95)', width=1.5),
                        hoverinfo='skip',
                        showlegend=False,
                    ),
                    row=i, col=1,
                )

        else:
            # Continuous numeric: scatter and per-bin mean/std overlays
            xv = df_plot[p].astype(float).values
            fig.add_trace(
                go.Scatter(
                    x=xv,
                    y=y,
                    mode='markers',
                    marker=dict(color='rgba(102,178,255,0.9)', size=5),
                    customdata=custom,
                    hovertemplate=f"trial %{{customdata[0]}}<br>{p}: %{{x}}<br>val_loss: %{{y:.6g}}<extra></extra>",
                ),
                row=i, col=1,
            )
            if xv.size:
                xmin, xmax = float(np.min(xv)), float(np.max(xv))
                bins = int(os.environ.get('NUM_BINS', '12'))
                min_bin_n = int(os.environ.get('MIN_BIN_N', '1'))
                if xmax > xmin:
                    edges = np.linspace(xmin, xmax, bins + 1)
                    # assign each point to bin index
                    bi = np.clip(np.digitize(xv, edges) - 1, 0, bins - 1)
                    for b in range(bins):
                        mask = (bi == b)
                        if mask.sum() < min_bin_n:
                            continue
                        x0, x1 = edges[b], edges[b + 1]
                        yg = y[mask]
                        m = float(np.mean(yg))
                        s = float(np.std(yg))
                        fig.add_shape(type='rect', xref=xref, x0=x0, x1=x1, yref=yref, y0=m - s, y1=m + s,
                                      fillcolor='rgba(255,99,71,0.08)', line_width=0, row=i, col=1)
                        fig.add_shape(type='line', xref=xref, x0=x0, x1=x1, yref=yref, y0=m, y1=m,
                                      line=dict(color='rgba(200,80,60,0.95)', width=1), row=i, col=1)

    # Taller subplots for readability (approx square/tall panels)
    per_panel = int(os.environ.get('PANEL_HEIGHT', '320'))
    layout_kwargs = dict(height=max(700, per_panel * n), width=int(os.environ.get('FIG_WIDTH', '1000')), showlegend=False, title=f"{STUDY} — LAST {OBJ}")
    # Optional log-scale for y
    if os.environ.get('Y_LOG', '0') in ('1', 'true', 'True'):
        layout_kwargs['yaxis_type'] = 'log'
        for r in range(2, n + 1):
            layout_kwargs[f'yaxis{r}_type'] = 'log'
    fig.update_layout(**layout_kwargs)
    fig.show()


if __name__ == '__main__':
    main()
# %%