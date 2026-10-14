"""Small offline smoke tests. Production GPUs and data are still required for training."""
import tempfile, unittest, pathlib, json
from unittest.mock import patch
import numpy as np
import pandas as pd
from soc_repro import r2c_engine as e
from soc_repro import audit

class TestResearchContracts(unittest.TestCase):
    def test_fixed_grid_size(self):
        self.assertEqual(len(e.ALL_DOMAINS)*len(e.FRACTIONS)*len(e.SEEDS)*len(e.MODELS_STD),350)
    def test_sparse_selector_determinism(self):
        df=pd.DataFrame({'vehicle_id':['v']*20,'segment_id':['s']*20,'time_value':list(range(20)),'target':list(range(20))})
        a=e.exact_sparse_subset(df,0.1,42)
        b=e.exact_sparse_subset(df,0.1,42)
        self.assertEqual(e.identity_sha256(a),e.identity_sha256(b))
        self.assertEqual(len(a),2)
    def test_progress_empty_root(self):
        with tempfile.TemporaryDirectory() as d:
            z=audit.scan_run_root(d)
            self.assertEqual(z['verified_models'],0)
            self.assertEqual(len(z['pending']),350)
            self.assertFalse(z['fresh_all_350_verified'])
    def test_progress_rejects_fake_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            out=pathlib.Path(d)/'VED'/'run_records';out.mkdir(parents=True)
            (out/'VED_fraction0.01_DirectPlusPrior_seed42.json').write_text(json.dumps({'domain':'VED'}))
            z=audit.scan_run_root(d)
            self.assertTrue(z['errors'])
    def test_no_original_checkpoint_source(self):
        from inspect import getsource
        source=getsource(e.train_one)
        self.assertIn('CKPT_DIR /',source)
        self.assertNotIn('R1A_info_matched_gru/checkpoints',source)


    def test_scheduler_skips_prior_40_runs_and_selects_next_fraction(self):
        existing={0.01,0.02,0.05,0.10}
        seen=[]
        def existing_receipt(domain,fraction,model,seed):
            return {'done':True} if float(fraction) in existing else None
        def train(domain,fraction,model,seed):
            seen.append((domain,fraction,model,seed))
            return {'resume_checkpoint':'/tmp/nonexistent_soc_checkpoint.pt',
                    'test_MAE':0.2,'best_epoch':1,'final_epoch':2}
        with patch.object(e,'read_completed',side_effect=existing_receipt), \
             patch.object(e,'train_one',side_effect=train), \
             patch.object(e,'save_receipt',side_effect=lambda item:item), \
             patch.object(e,'RUN_DOMAIN','VED',create=True), \
             patch.object(e,'OUT',pathlib.Path('/tmp/soc_testing'),create=True):
            n=e.run_domain(1)
        self.assertEqual(n,1)
        self.assertEqual(seen,[('VED',0.20,'Direct+Prior',42),('VED',0.20,'PriorResidual+Prior',42)])

    def test_cli_status_with_no_gpu(self):
        from soc_repro import cli
        with tempfile.TemporaryDirectory() as d:
            cli.main(['--root',d,'--status'])

if __name__=='__main__':unittest.main()
