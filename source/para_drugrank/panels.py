"""Select a declared ordered drug panel before training or evaluation."""
import json
from pathlib import Path
import numpy as np
from .data import axis_digest

ROOT = Path(__file__).resolve().parents[2]
COMMON_PANEL_HASHES = {
    'PRISM': '4bee086a426e6ec40fc992c3d0bd7bf3d0aea8996bc72ac706f21b13618a46ab',
    'GDSC2': 'c1d0351b69a85d91daca3c1ff3895c628bc768b93f075606e7a146230fd32c88',
    'CTRPv2.1': 'aec36bcd7981da67208281420817e8192421d7178d3a38602d3db878373b4349',
}


def select_panel(data, panel='common'):
    if panel in ('full', 'custom'):
        return data
    if panel != 'common':
        raise ValueError('Unknown panel: ' + panel)
    cohort = str(data['cohort'].item())
    lookup = {str(name): i for i, name in enumerate(data['drug_names'])}
    local_panel = ROOT / 'configs/common_drugs.json'
    if local_panel.is_file():
        names = json.loads(local_panel.read_text(encoding='utf-8'))[cohort]
    else:
        # The benchmark uses alphabetical drug order. One GDSC2 compound was
        # excluded from the common comparison because its structure was unavailable.
        names = sorted(n for n in lookup if not (cohort == 'GDSC2' and n == 'iap5620'))
    if axis_digest(names) != COMMON_PANEL_HASHES.get(cohort):
        raise ValueError('Drug axis does not match the published common panel. Use --panel custom for a different panel, or provide the study panel list in configs/common_drugs.json for a wider input.')
    missing = set(names) - set(lookup)
    if missing:
        raise ValueError('Input is missing common-panel drugs: ' + ', '.join(sorted(missing)))
    columns = np.asarray([lookup[name] for name in names], dtype=int)
    output = dict(data)
    for key in ('source_response', 'target_response'):
        if key in data:
            output[key] = data[key][:, columns].copy()
    output['drug_names'] = np.asarray(names)
    return output
