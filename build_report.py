"""Assemble report.md -> Assignment2_Report.pdf, filling every %%NUMBER%% from the result files.

  python build_report.py && pandoc report_full.md -o Assignment2_Report.pdf --pdf-engine=xelatex
"""
import json
import subprocess
from pathlib import Path

R = Path('results')


def j(path):
    return json.loads((R / path).read_text())


def main():
    body = ''.join(Path(f).read_text() + '\n' for f in
                   ['report_body_impl.md', 'report_body_cp.md', 'report_body_msp.md', 'report_body_tests.md'])
    text = Path('report.md').read_text().replace('%%BODY%%', body)

    a = j('validation/analytic.json')
    t1 = {r['gamma']: r['max_abs_err'] for r in a['chain_mdp']}
    c = j('validation/cartpole.json')
    e = c['epsilon_noise']
    phys = {(r['param'], r['value']): r['mean'] for r in c['physics']}
    sub = {
        'CP_TARGET': '500', 'CP_LR': 'lr $10^{-3}\\to10^{-4}$ (linear over 100k steps)', 'CP_BATCH': '64',
        'T1_0': f"{t1[0.0]:.1e}", 'T1_05': f"{t1[0.5]:.1e}", 'T1_095': f"{t1[0.95]:.1e}", 'T1_099': f"{t1[0.99]:.1e}",
        'T1_BUG': f"{a['chain_mdp_without_terminal_mask']['max_abs_err']:.2f}",
        'FORCE5': f"{phys[('force_mag', 5.0)]:.0f}", 'MASS05': f"{phys[('masspole', 0.5)]:.0f}",
        'G196': f"{phys[('gravity', 19.6)]:.0f}",
        'MIRROR': f"{c['mirror_symmetry_visited']['action_flip_rate_confident_states']:.2f}",
        'SWEEP_TABLE': subprocess.run(['python3', 'summarize.py'], capture_output=True, text=True).stdout,
    }
    for k, key in zip(['0', '005', '01', '025', '05', '1'], ['0.0', '0.05', '0.1', '0.25', '0.5', '1.0']):
        sub['E' + k] = f"{e[key]['mean']:.1f}"
        sub['F' + k] = f"{e[key]['fraction_500']:.2f}"
    msp = R / 'validation' / 'mspacman.json'
    if msp.exists():
        m = json.loads(msp.read_text())
        rows = [(k.split('_')[1], v) for k, v in m.items() if k.startswith('epsilon_')]
        sub['MSP_NOISE'] = ('| $\\varepsilon$ at test time | ' + ' | '.join(r[0] for r in rows) + ' |\n|---|'
                            + '---|' * len(rows) + '\n| MsPacman mean score (30 seeds) | '
                            + ' | '.join(f"{r[1]['mean']:.0f}" for r in rows) + ' |\n| MsPacman mean episode length | '
                            + ' | '.join(f"{r[1]['mean_length']:.0f}" for r in rows) + ' |\n')
    extra = Path('report_numbers.json')
    if extra.exists():  # numbers computed in the MsPacman analysis
        sub.update(json.loads(extra.read_text()))
    for k, v in sub.items():
        text = text.replace(f'%%{k}%%', str(v))
    missing = sorted(set(p.split('%%')[0] for p in text.split('%%')[1::2]))
    if missing:
        print('UNFILLED placeholders:', missing)
    Path('report_full.md').write_text(text)


if __name__ == '__main__':
    main()
