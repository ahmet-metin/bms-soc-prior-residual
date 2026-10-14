"""Assertions about the public-facing layout, not about GPU model convergence."""
import json
from pathlib import Path
import unittest
from soc_repro.audit import DOMAINS, FRACTIONS, SEEDS, MODELS

ROOT=Path(__file__).resolve().parents[1]

class TestPublicArchive(unittest.TestCase):
    def test_source_index_and_coverage(self):
        rows=json.loads((ROOT/'docs'/'SOURCE_REGISTRY.json').read_text())
        self.assertEqual(len(rows),28)
        for name in ('benchmark_ensemble','inference','fixed_time_sensitivity','robustness','preprocessing'):
            self.assertTrue(any('/'+name+'/' in r['path'] for r in rows),name)
        self.assertTrue(all((ROOT/r['path']).is_file() for r in rows))

    def test_paper_reference_grid(self):
        import csv
        def read(p):
            with (ROOT/'references'/p).open(newline='') as f:
                return list(csv.DictReader(f))
        r7=read('R7_35_conditions_ARCHIVED.csv')
        eve=read('EVE_postanchor_14_conditions_ARCHIVED.csv')
        self.assertEqual(len(r7),35)
        self.assertEqual(len(eve),14)
        self.assertEqual(len(DOMAINS)*len(FRACTIONS)*len(SEEDS)*len(MODELS),350)
        self.assertEqual(len({(r['domain'],r['fraction']) for r in r7}),35)

    def test_no_model_weights_or_private_data_in_repo(self):
        bad=[]
        for p in ROOT.rglob('*'):
            if p.is_file() and p.suffix.lower() in {'.parquet','.pt','.pth'}:
                bad.append(str(p.relative_to(ROOT)))
        self.assertFalse(bad,bad)

if __name__=='__main__':unittest.main()
