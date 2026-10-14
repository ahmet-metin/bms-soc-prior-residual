"""Synthetic contracts; these do not assert the research models were retrained."""
import unittest,tempfile,importlib
from pathlib import Path
import numpy as np
import pandas as pd
from soc_repro.postprocess import normalize_predictions,summarize_r7,normalized_log_auc,DOMAINS,FRACTIONS,MODELS
from soc_repro.robustness import summarize_anchor_age,bootstrap_segment_delta
from soc_repro.reanchoring import scheduled_refresh_positions,apply_refresh_policy,summarize_periodic_refresh
from soc_repro.experiments import entries
from soc_repro.report import figure2,figure3,figure4


def sample_all(pooled_eve_segments=509):
    rows=[]; budget=[]
    for domain in DOMAINS:
        for frac in FRACTIONS:
            budget.append({'domain':domain,'fraction':frac,'effective_fit_fraction_mean':0.1+0.9*frac,
                           'actual_nominal_fraction_mean':frac})
            nsegments=(pooled_eve_segments+39 if domain=='EVE' else 2)
            for j in range(nsegments):
                # 39 extra EVE segments contain only the start anchor.
                times=[0] if domain=='EVE' and j>=pooled_eve_segments else [0,300,600,900]
                for t in times:
                    for model in MODELS:
                        rows.append({'domain':domain,'fraction':frac,'model':model,
                                     'vehicle_id':'v0','segment_id':f's{j}',
                                     'compound_segment':f'v0||s{j}',
                                     'time_value':t,'elapsed_s':t,'target':50.,
                                     'prediction':48. if model==MODELS[0] else 49.})
    return pd.DataFrame(rows),pd.DataFrame(budget)

class TestPredictionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pred,cls.budget=sample_all()

    def test_70_35_and_eve_509_primary(self):
        long,wide,auc=summarize_r7(self.pred,self.budget)
        self.assertEqual(len(long),70);self.assertEqual(len(wide),35);self.assertEqual(len(auc),5)
        self.assertEqual(set(wide.n_segments[wide.domain=='EVE']),{509})
        self.assertTrue((wide.PR_segment_balanced_MAE<wide.Direct_segment_balanced_MAE).all())
        self.assertTrue(np.allclose(auc['Direct_effective_AULEC_segment_balanced'],2.))
        self.assertTrue(np.allclose(auc['PR_effective_AULEC_segment_balanced'],1.))

    def test_reject_single_model_missing_matched_row(self):
        z=self.pred.drop(self.pred.index[0]).copy()
        with self.assertRaises(ValueError):normalize_predictions(z)

    def test_reject_seed_level_duplicates(self):
        z=pd.concat([self.pred,self.pred.iloc[[0]]],ignore_index=True)
        with self.assertRaises(ValueError):normalize_predictions(z)

    def test_strict_wrong_grid_fails(self):
        z=self.pred.loc[~((self.pred.domain=='VED')&(self.pred.fraction==0.01))]
        with self.assertRaises(ValueError):normalize_predictions(z)

    def test_eve_anchor_only_removed(self):
        post=normalize_predictions(self.pred)
        a=post.loc[post.domain=='EVE']
        self.assertEqual(a.compound_segment.nunique(),509)
        self.assertTrue((a.elapsed_s>0).all())

    def test_r6_age_bins_and_ci(self):
        _,s=summarize_anchor_age(self.pred,draws=80)
        self.assertTrue(set(s.anchor_age_bin.astype(str)).issubset({'0–5','5–10','10–20','20–40','≥40'}))
        self.assertTrue((s.Delta_PR_minus_Direct<0).all())
        self.assertEqual(len(s[['domain','fraction']].drop_duplicates()),10)
        self.assertEqual(s['interpretable_bin'].sum(),6)  # EVE has 509 supported segments in 3 nonempty bins × 2 fractions

    def test_refresh_policy_causal_and_cost(self):
        g=pd.DataFrame({'elapsed_s':[0,100,600,900,1200], 'time_value':[0,100,600,900,1200],
                        'target':[10]*5,'prediction':[5]*5})
        self.assertEqual(scheduled_refresh_positions(g,600),[2,4])
        z=apply_refresh_policy(g,600)
        self.assertEqual(z.refresh_event.sum(),2)
        self.assertEqual(z.prediction_refreshed.tolist(),[5.,5.,10.,10.,10.])
        # Never compare errors at a refreshed row as if it were an unseen target.
        self.assertEqual(z.loc[~z.refresh_event].prediction_refreshed.tolist(),[5.,5.,10.])

    def test_refresh_grouped(self):
        s,seg=summarize_periodic_refresh(self.pred)
        self.assertEqual(len(s),30)
        self.assertEqual(set(s.policy),{'5min','10min','20min'})
        self.assertTrue((s.additional_BMS_observations_per_hour>=0).all())

    def test_plot_files(self):
        _,wide,_=summarize_r7(self.pred,self.budget)
        _,age=summarize_anchor_age(self.pred,draws=20)
        refr,_=summarize_periodic_refresh(self.pred)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for fn,data,method in [('f2.pdf',wide,figure2),('f3.pdf',age,figure3),('f4.pdf',refr,figure4)]:
                path=root/fn;method(data,path)
                self.assertTrue(path.exists() and path.stat().st_size>100)

    def test_archive_aulec_example(self):
        x=np.array([.03,.06,.1,.2,.35,.7,1.])
        self.assertAlmostEqual(normalized_log_auc(x,[2]*7),2.)
        with self.assertRaises(ValueError):normalized_log_auc([0]*7,[2]*7)

    def test_bounded_bootstrap(self):
        point,lo,hi=bootstrap_segment_delta(np.full(50,-2.0),draws=333)
        self.assertAlmostEqual(point,-2.)
        self.assertAlmostEqual(lo,-2.)
        self.assertAlmostEqual(hi,-2.)

    def test_manuscript_registry(self):
        r=entries()
        self.assertEqual(len(r),17)
        self.assertEqual(len({z['component'] for z in r}),len(r))
        self.assertTrue(all(z['verification']!='independently_reexecuted' for z in r))

if __name__=='__main__':unittest.main()
