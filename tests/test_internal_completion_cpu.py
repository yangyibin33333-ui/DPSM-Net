"""CPU-only checks for scientific definitions and paired-export integrity."""
import copy,io,json,subprocess,sys,tempfile,unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import yaml
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'figures/internal_visualizations'
sys.path.insert(0,str(HERE))
from completion_common import (boundary_mask,boundary_display,topology_display,spatial_scores,
    validate_framework_configs,check_prediction_pair,PredictionBundle,binary_mask)
from select_geometry_cases import resized_geometry
from capture_internal import sha256
import render_completion as render
import completion_common as completion_module


def require_contiguous_skeletonize(image):
    """Emulate older skimage wrappers that pass strides to the Cython kernel."""
    if not image.flags.c_contiguous:
        raise ValueError('ndarray is not C-contiguous')
    if not image.flags.writeable:
        raise ValueError('buffer source array is read-only')
    return original_skeletonize(image)


original_skeletonize=completion_module.skeletonize


class DiagnosticsTest(unittest.TestCase):
    def test_readonly_masks_get_owned_writable_buffers(self):
        raw=np.zeros((12,18),bool); raw[2:10,5:9]=True
        raw.setflags(write=False)
        for value in (raw,raw[1:11,2:16],raw.T,raw[::-1,::-1]):
            prepared=binary_mask(value)
            self.assertTrue(prepared.flags.c_contiguous)
            self.assertTrue(prepared.flags.writeable)
            self.assertTrue(prepared.flags.owndata)
            np.testing.assert_array_equal(prepared,value)
            self.assertFalse(np.shares_memory(prepared,value))
            with patch.object(completion_module,'skeletonize',require_contiguous_skeletonize):
                self.assertAlmostEqual(spatial_scores(value,value)['clDice'],1)

    def test_strided_masks_match_contiguous_scores_and_display(self):
        gt=np.zeros((18,30),bool); gt[3:15,10:13]=True
        pred=gt.copy(); pred[8,13:16]=True
        views=(lambda a:a[1:17,4:24],lambda a:a.T,
               lambda a:np.asfortranarray(a),lambda a:a[::-1,::-1])
        with patch.object(completion_module,'skeletonize',require_contiguous_skeletonize):
            for view in views:
                g,p=view(gt),view(pred)
                self.assertFalse(g.flags.c_contiguous)
                with self.assertRaisesRegex(ValueError,'not C-contiguous'):
                    require_contiguous_skeletonize(g)
                gc,pc=np.ascontiguousarray(g),np.ascontiguousarray(p)
                expected=spatial_scores(pc,gc)
                np.testing.assert_array_equal(topology_display(p,g),topology_display(pc,gc))
                self.assertEqual(spatial_scores(p,g),expected)
                np.testing.assert_array_equal(g,gc)
                np.testing.assert_array_equal(p,pc)

    def test_nearest_coordinates(self):
        raw=np.arange(15).reshape(3,5)
        resized,crop=resized_geometry(raw,[7,8])
        np.testing.assert_array_equal(resized,raw[np.array([0,0,1,1,2])[:,None],np.array([0,0,1,1,2,3,3,4])[None,:]])
        self.assertEqual(crop,[1,0,5,8])

    def test_boundary_padding_matches_pool_definition(self):
        rng=np.random.default_rng(42)
        for a in (np.ones((3,4),bool),np.zeros((3,4),bool),rng.random((7,9))>.7):
            expected=np.zeros(a.shape,bool)
            for y in range(a.shape[0]):
                for x in range(a.shape[1]):
                    v=a[max(0,y-1):y+2,max(0,x-1):x+2]
                    expected[y,x]=v.max()!=v.min()
            np.testing.assert_array_equal(boundary_mask(a),expected)

    def test_cldice_uses_mask_coverage_not_skeleton_intersection(self):
        gt=np.zeros((9,12),bool); gt[3:6,2:10]=True
        pred=gt.copy(); pred[2,2:10]=True
        self.assertAlmostEqual(spatial_scores(pred,gt)['clDice'],1)
        a=topology_display(pred,gt)
        self.assertTrue(np.all(a[np.any(a!=.96,axis=-1)]==[0,.62,.45]))

    def test_component_count_gain_can_coexist_with_missed_geometry(self):
        gt=np.zeros((13,15),bool); gt[2:11,2]=True; gt[2:11,12]=True
        a=np.zeros_like(gt); a[2:11,2]=True
        b=a.copy(); b[6,12]=True
        sa,sb=spatial_scores(a,gt),spatial_scores(b,gt)
        self.assertEqual((sa['ComponentError'],sb['ComponentError']),(1,0))
        self.assertEqual(sb['GT_skeleton_missed'],8)
        self.assertLess(sb['clDice'],.8)

    def test_boundary_tolerance_and_display(self):
        gt=np.zeros((15,20),bool); gt[4:11,5:8]=True
        near=np.roll(gt,1,axis=1); far=np.roll(gt,7,axis=1)
        self.assertAlmostEqual(spatial_scores(near,gt)['BoundaryF1'],1)
        self.assertEqual(spatial_scores(far,gt)['BoundaryF1'],0)
        colors=boundary_display(far,gt)
        self.assertTrue(np.any(np.all(colors==[0,.45,.70],axis=-1)))
        self.assertTrue(np.any(np.all(colors==[.84,.37,0],axis=-1)))


class IntegrityTest(unittest.TestCase):
    def test_real_framework_configs_and_confounds(self):
        preset=json.loads((HERE/'verified_completion_seed42.json').read_text(encoding='utf-8'))
        cfg=[yaml.safe_load((ROOT/preset['runs'][s]['directory']/'config.yaml').read_text(encoding='utf-8')) for s in ('F2','F3')]
        validate_framework_configs(*cfg)
        for section,key in (('train','batch_size'),('model','base_channels'),('loss','lambda_tangent'),
                            ('auto_protocol_resolved','training_seed')):
            wrong=copy.deepcopy(cfg[1]); wrong[section][key]=999
            with self.assertRaises(ValueError): validate_framework_configs(cfg[0],wrong)

    def test_pair_rejects_seed_precision_and_secondary_loss(self):
        a={'seed':42,'weights':'model','precision':'FP32','prediction_threshold':.5,'source_sha256':{},
           'model_name':'DEMambaNet','model':{},'data':{},'train':{},'protocol_identity':{},'training_context':{},
           'loss':{'lambda_tangent':0,'lambda_bpcm':0}}
        b=copy.deepcopy(a); b['loss']['lambda_tangent']=.2
        check_prediction_pair(SimpleNamespace(manifest=a),SimpleNamespace(manifest=b),'tangent')
        for key,value in (('seed',43),('precision','FP16'),('train',{'batch_size':64})):
            bad=copy.deepcopy(b); bad[key]=value
            with self.assertRaises(ValueError): check_prediction_pair(SimpleNamespace(manifest=a),SimpleNamespace(manifest=bad),'tangent')
        bad=copy.deepcopy(b); bad['loss']['lambda_bpcm']=.025
        with self.assertRaises(ValueError): check_prediction_pair(SimpleNamespace(manifest=a),SimpleNamespace(manifest=bad),'tangent')

    def test_prediction_archive_rejects_changed_file_and_nan(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td); content=np.zeros((8,10),bool); content[1:7,2:8]=True
            arrays={'rgb':np.zeros((8,10,3),np.float32),'mask':np.zeros((8,10),bool),
                    'probability':np.full((8,10),.2,np.float32),'content':content,'crop_yxhw':np.array([1,2,6,6])}
            np.savez_compressed(path/'case.npz',**arrays)
            m={'schema':'demamba_prediction_v1','complete':True,'data':{'img_size':[8,10]},
               'samples':[{'stem':'case','file':'case.npz','sha256':sha256(path/'case.npz'),
                           'crop_yxhw':[1,2,6,6],'model_hw':[8,10]}]}
            def manifest(): (path/'manifest.json').write_text(json.dumps(m),encoding='utf-8')
            manifest(); bundle=PredictionBundle(path); bundle.load('case')
            with (path/'case.npz').open('ab') as f: f.write(b'changed')
            with self.assertRaises(ValueError): bundle.load('case')
            arrays['probability'][0,0]=np.nan
            np.savez_compressed(path/'case.npz',**arrays); m['samples'][0]['sha256']=sha256(path/'case.npz')
            manifest()
            with self.assertRaises(ValueError): PredictionBundle(path).load('case')


class RenderContractTest(unittest.TestCase):
    """Synthetic arrays exercise renderer APIs; no scientific artifacts are saved."""
    def test_plot_buffers_preserve_masked_values(self):
        raw=np.arange(120).reshape(10,12)[:,2:10]
        masked=np.ma.array(raw,mask=raw%3==0)
        prepared=render.plot_array(masked)
        self.assertTrue(prepared.data.flags.c_contiguous)
        self.assertTrue(prepared.mask.flags.c_contiguous)
        np.testing.assert_array_equal(prepared.data,masked.data)
        np.testing.assert_array_equal(prepared.mask,masked.mask)
        np.testing.assert_array_equal(render.crop(raw,[1,1,3,4]),raw[1:5,1:4])
        self.assertTrue(render.crop(raw,[1,1,3,4]).flags.c_contiguous)

    def test_runtime_rejects_outdated_helper(self):
        capture=io.StringIO()
        with redirect_stdout(capture),patch.object(completion_module,'CONTIGUITY_FIX_VERSION','missing'):
            with self.assertRaisesRegex(RuntimeError,'Synchronize both'):
                render.report_runtime()
        self.assertIn('Loaded diagnostics:',capture.getvalue())
        self.assertIn('diagnostics fix: missing',capture.getvalue())

    def test_cli_keeps_traceback_for_failed_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            result=subprocess.run([sys.executable,str(HERE/'render_completion.py'),
                '--stage','framework','--output-root',td],cwd=ROOT,text=True,capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Renderer fix: contiguous-v2; diagnostics fix: contiguous-v2',result.stdout)
        self.assertIn('Traceback (most recent call last)',result.stderr)
        self.assertIn('FileNotFoundError',result.stderr)

    def test_framework_and_direction_renderers_without_torch(self):
        selection=json.loads((HERE/'cases_four_directions_seed42.json').read_text(encoding='utf-8'))
        preset=json.loads((HERE/'verified_completion_seed42.json').read_text(encoding='utf-8'))
        configs={s:yaml.safe_load((ROOT/preset['runs'][s]['directory']/'config.yaml').read_text(encoding='utf-8')) for s in ('F2','F3')}
        common={'seed':42,'weights':'model','precision':'FP32','prediction_threshold':.5,'source_sha256':{},
                'model_name':'DEMambaNet','data':{},'train':{},'protocol_identity':{},'training_context':{}}
        samples={r['stem']:{**r,'model_hw':[352,640],'original_hw':[360,640]} for r in selection['cases']}
        gt=np.zeros((352,640),bool); gt[60:285,200:204]=True
        arr={'mask':gt,'rgb':np.full((352,640,3),.3),'probability':np.where(gt,.7,.1),
             'crop_yxhw':np.array([0,7,352,626]),'content':np.pad(np.ones((352,626),bool),((0,0),(7,7)))}
        for s in ('route_probability','route_target'): arr[s]=np.full((4,44,80),.25)
        arr['route_valid']=np.ones((44,80),bool)
        def prediction(path):
            label=path.name
            return SimpleNamespace(samples=samples,manifest={**common,'resolved_config':configs[label]},
                                   path=path,load=lambda stem:{k:v.copy() for k,v in arr.items()})
        def internal(path):
            return SimpleNamespace(samples=samples,manifest={'seed':42,'label':'D3','prediction_threshold':.5},
                                   path=path,load=lambda stem:{k:v.copy() for k,v in arr.items()})
        outputs=[]
        def inspect(fig,out,name,metadata,dpi):
            fig.canvas.draw()
            self.assertTrue(np.isfinite(np.asarray(fig.canvas.buffer_rgba())).all())
            for format in ('png','pdf'):
                buffer=io.BytesIO()
                fig.savefig(buffer,format=format,dpi=72)
                self.assertGreater(buffer.getbuffer().nbytes,100)
            outputs.append((name,metadata)); render.plt.close(fig)
        with tempfile.TemporaryDirectory() as td, patch.object(completion_module,'skeletonize',require_contiguous_skeletonize),\
                patch.object(render,'PredictionBundle',prediction),\
                patch.object(render,'Bundle',internal),patch.object(render,'bundle_evidence',lambda b:{'synthetic_test':True}),\
                patch.object(render,'export',inspect):
            out=Path(td)
            render.framework(out,out,selection,600,.25)
            render.routes(out,out,selection,600)
        self.assertEqual([s for s,_ in outputs],['fig04_framework_F2_F3','fig05_four_direction_heads'])
        self.assertEqual(len(outputs[0][1]['cases']),2)
        self.assertEqual(len(outputs[1][1]['cases']),4)


if __name__=='__main__': unittest.main()
