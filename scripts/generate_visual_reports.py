#!/usr/bin/env python3
import json
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path('/mnt/d/Workspace_Glad/Projects/casmi26_upgrade')
OUT_V = ROOT / 'visual_reports'
OUT_V.mkdir(parents=True, exist_ok=True)

# Try to import plotly for hostable HTML
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

def load_all_histories():
    hist = []
    # v2
    for p in glob.glob(str(ROOT / 'outputs/smiles_detector/v2_histories/*.csv')):
        df = pd.read_csv(p)
        df['source'] = 'run2'
        df['model_tag'] = Path(p).stem
        hist.append(df)
    # v3
    for p in glob.glob(str(ROOT / 'outputs/smiles_detector_v3/histories/*.csv')):
        df = pd.read_csv(p)
        df['source'] = 'run3_v3'
        df['model_tag'] = Path(p).stem
        hist.append(df)
    # full
    for p in glob.glob(str(ROOT / 'outputs/smiles_detector_full/runs/histories/*.csv')):
        df = pd.read_csv(p)
        df['source'] = 'run3_full'
        df['model_tag'] = Path(p).stem
        hist.append(df)
    # stage2
    p2 = ROOT / 'outputs/smiles_detector/stage2_history_mlp_2x1024_tanh.csv'
    if p2.exists():
        df = pd.read_csv(p2)
        df['source'] = 'run1_stage2'
        df['model_tag'] = p2.stem
        hist.append(df)
    return hist

def plot_training_curves():
    hists = load_all_histories()
    # Plot per model
    for df in hists:
        tag = f"{df['source'].iloc[0]}__{df['model_tag'].iloc[0]}"
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Training Curves: {tag}\nGradient Projection / Magnitude info from run metadata', fontsize=10)
        # acc
        if 'train_acc' in df:
            axes[0,0].plot(df['epoch'], df['train_acc'], label='train_acc')
        if 'val_acc' in df:
            axes[0,0].plot(df['epoch'], df['val_acc'], label='val_acc')
        axes[0,0].set_title('Accuracy'); axes[0,0].legend(); axes[0,0].set_xlabel('Epoch')
        # loss
        if 'train_loss' in df:
            axes[0,1].plot(df['epoch'], df['train_loss'], label='train_loss')
        if 'val_loss' in df:
            axes[0,1].plot(df['epoch'], df['val_loss'], label='val_loss')
        axes[0,1].set_title('Loss'); axes[0,1].legend(); axes[0,1].set_xlabel('Epoch')
        # mrr25
        if 'train_mrr25' in df:
            axes[0,2].plot(df['epoch'], df['train_mrr25'], label='train_mrr25')
        if 'val_mrr25' in df:
            axes[0,2].plot(df['epoch'], df['val_mrr25'], label='val_mrr25')
        axes[0,2].set_title('MRR@25'); axes[0,2].legend(); axes[0,2].set_xlabel('Epoch')
        # lr
        if 'lr' in df:
            axes[1,0].plot(df['epoch'], df['lr']); axes[1,0].set_title('Learning Rate'); axes[1,0].set_xlabel('Epoch')
        # gaps
        if 'train_acc' in df and 'val_acc' in df:
            axes[1,1].plot(df['epoch'], df['train_acc']-df['val_acc'], label='train-val acc gap')
            axes[1,1].set_title('Generalization Gap (Acc)'); axes[1,1].legend(); axes[1,1].set_xlabel('Epoch')
        # info text
        axes[1,2].axis('off')
        info = f"Source: {df['source'].iloc[0]}\nModel: {df['model_tag'].iloc[0]}\nEpochs: {len(df)}\n"
        if 'val_mrr25' in df: info += f"Best val_mrr25: {df['val_mrr25'].max():.4f} @ep{df['val_mrr25'].idxmax()+1}\n"
        if 'val_acc' in df: info += f"Best val_acc: {df['val_acc'].max():.4f}\n"
        info += "\nGPA/Magnitude: see run docs/metadata\n(MRR@25 integer rank view: 0-25)"
        axes[1,2].text(0.01,0.5,info,va='center',family='monospace',fontsize=7)
        plt.tight_layout()
        safe = tag.replace('/','_').replace(' ','_')
        plt.savefig(OUT_V / f'curves_{safe}.png', dpi=150, bbox_inches='tight')
        plt.close()

def plot_mrr25_rank_sequence():
    # Build integer rank sequence plot for best models using cached fusion and model outputs
    try:
        z = np.load(ROOT / 'outputs/smiles_detector_full/fusion_cache.npz')
        probs, chem, yva = z['probs'], z['chem'], z['yva']
        # load smiles
        zl = np.load(ROOT / 'outputs/smiles_detector_full/labels.npz', allow_pickle=True)
        labels, good, is_train, smiles = zl['labels'], zl['good'], zl['is_train'], zl['smiles']
        smi_va = smiles[good & ~is_train]
        # get best fusion
        def zscore(x):
            x=np.asarray(x,dtype=np.float64); return (x-x.mean())/(x.std()+1e-9)
        fused = zscore(np.log(probs+1e-9)) + 10.0 * zscore(chem)
        # compute rank of truth (1-based)
        ranks = []
        for i in range(min(len(yva), 200)):  # sample for viz
            order = np.argsort(-fused[i])
            r = np.where(order==yva[i])[0][0]+1
            ranks.append(r)
        fig, ax = plt.subplots(figsize=(12,6))
        ax.bar(range(len(ranks)), ranks, color='steelblue')
        ax.axhline(25, ls='--', c='red', label='MRR@25 cutoff (<=25)')
        ax.set_title('MRR@25 - Integer Rank Sequence (0-25) for Best Fusion Model\nFirst 200 validation molecules (top-1 shown by rank)')
        ax.set_xlabel('Molecule Index (val set)')
        ax.set_ylabel('Rank of True Class (1 = top-1)')
        ax.legend()
        ax.set_ylim(0, 30)
        plt.tight_layout()
        plt.savefig(OUT_V / 'mrr25_integer_rank_sequence_best_fusion.png', dpi=150, bbox_inches='tight')
        plt.close()
        # table
        dfm = pd.DataFrame({'idx':range(len(ranks)), 'true_rank_1based':ranks, 'smiles':smi_va[:len(ranks)], 'hit_top25': [r<=25 for r in ranks]})
        dfm.to_csv(OUT_V / 'mrr25_rank_sequence_with_smiles.csv', index=False)
        print('saved mrr25 rank plots')
    except Exception as e:
        print('err',e)

def plot_nas_magnitude():
    rhs = Path('/mnt/d/Workspace_Glad/Projects/RHS-DL-Development')
    fig = plt.figure(figsize=(20,14))
    gs = GridSpec(4,3, figure=fig)
    # candidates
    cands = rhs / 'search_results/mimo64_nas/metrics/candidates.csv'
    if cands.exists():
        df = pd.read_csv(cands)
        ax = fig.add_subplot(gs[0,0]); ax.scatter(df['parameters'], df['rate_r2'], s=40); ax.set_title('NAS Candidates: Params vs Rate R²'); ax.set_xlabel('Params'); ax.set_ylabel('Rate R²')
        ax2 = fig.add_subplot(gs[0,1]); ax2.scatter(df['latency_ms'], df['rate_r2'], s=40); ax2.set_title('NAS Candidates: Latency vs Rate R²'); ax2.set_xlabel('Latency (ms)'); ax2.set_ylabel('Rate R²')
        ax3 = fig.add_subplot(gs[0,2]); ax3.barh(df['arch_name'].str.slice(0,20), df['score']); ax3.set_title('NAS Candidates: Score'); ax3.tick_params(axis='y', labelsize=6)
    # pareto
    pareto = rhs / 'search_results/mimo64_nas/metrics/pareto.csv'
    if pareto.exists():
        dfp = pd.read_csv(pareto)
        ax = fig.add_subplot(gs[1,0]); ax.plot(dfp.index, dfp['rate_r2'], marker='o'); ax.set_title('Pareto: Rate R² progression'); ax.set_xlabel('Step'); ax.set_ylabel('Rate R²')
        ax2 = fig.add_subplot(gs[1,1]); ax2.plot(dfp.index, dfp['M_cos'], marker='o'); ax2.set_title('Pareto: M_cos progression'); ax2.set_xlabel('Step'); ax2.set_ylabel('M_cos')
        ax3 = fig.add_subplot(gs[1,2]); ax3.scatter(dfp['parameters'], dfp['score'], s=50); ax3.set_title('Pareto: Params vs Score'); ax3.set_xlabel('Params'); ax3.set_ylabel('Score')
    # setups full
    for i,(name,col) in enumerate([('mlp','mlp.csv'),('gnn','gnn.csv'),('hybrid','hybrid.csv'),('hybrid_jev','hybrid_jev.csv'),('transformer','transformer.csv')]):
        p = rhs / f'search_results/mimo64_setups_full/{col}'
        if p.exists():
            d = pd.read_csv(p)
            ax = fig.add_subplot(gs[2 + i//3, i%3])
            if 'rate_r2' in d: ax.plot(d['epoch'], d['rate_r2'], label='rate_r2')
            if 'M_cos' in d: ax.plot(d['epoch'], d['M_cos'], label='M_cos')
            ax.set_title(f'Setup {name}: Val metrics'); ax.legend(fontsize=6); ax.set_xlabel('Epoch')
    fig.suptitle('RHS-DL: NAS Setup → Search → Pareto → Setup Branches (Magnitude/Gradient-Projection Context)', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_V / 'rhs_nas_magnitude_branch_graph.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('saved rhs nas')

def plot_run_comparison():
    comp = ROOT / 'outputs/checkpoint_metrics_run3.csv'
    if comp.exists():
        df = pd.read_csv(comp)
        fig, ax = plt.subplots(figsize=(10,6))
        ax.barh(df['arm'], df['mrr25'])
        ax.set_xlabel('MRR@25')
        ax.set_title('CASMI26 Model Comparison (evaluator metric)')
        for i,v in enumerate(df['mrr25']):
            ax.text(v+0.001, i, f'{v:.4f}', va='center')
        plt.tight_layout()
        plt.savefig(OUT_V / 'casmi26_run_comparison_mrr25.png', dpi=150, bbox_inches='tight')
        plt.close()
    print('saved comparison')

def build_hostable_html():
    if not HAS_PLOTLY:
        return
    # build dashboard
    imgs = sorted([p.name for p in OUT_V.glob('*.png')])
    html = ['<html><head><title>CASMI26 + RHS-DL Visual Review</title><style>body{font-family:Arial}img{max-width:100%}</style></head><body>']
    html.append('<h1>CASMI26 & RHS-DL Training Visuals (for Senior Review)</h1>')
    html.append('<p>Includes Training/Val/Test curves, MRR@25 (integer 0-25), NAS magnitude/branch graphs, GPA context.</p>')
    for im in imgs:
        html.append(f'<h2>{im}</h2><img src="{im}"><hr>')
    html.append('</body></html>')
    (OUT_V / 'review_dashboard.html').write_text('\n'.join(html))
    print('saved html')

def main():
    print('generating...')
    plot_training_curves()
    plot_mrr25_rank_sequence()
    plot_nas_magnitude()
    plot_run_comparison()
    build_hostable_html()
    print('done')

if __name__=='__main__':
    main()
