"""CPU tests: python tests/test_export_demamba_shared_cases.py. Never trains a model."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('exporter',REPO/'scripts/export_demamba_shared_cases.py')
tool=importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)


def rejects(function, fragment):
    try: function()
    except (ValueError,FileNotFoundError) as error:
        assert fragment.lower() in str(error).lower(),str(error)
    else: raise AssertionError('Invalid export was accepted')


def fixture(root):
    data=root/'dataset'
    for folder in ('test/images','test/masks'): (data/folder).mkdir(parents=True)
    rows=[]
    for i in range(6):
        stem=f'sample ({i})'
        rgb=np.zeros((16,32,3),np.uint8); rgb[:,:,0]=20+i*25
        gt=np.zeros((16,32),np.uint8); gt[2:14,8+i]=255
        ip=data/'test/images'/f'{stem}.bmp'; mp=data/'test/masks'/f'{stem}.png'
        Image.fromarray(rgb).save(ip); Image.fromarray(gt).save(mp)
        rows.append({'dataset':'CFD','split':'test','pair_id':stem,'original_stem':stem,'original_mask_stem':stem,
          'case_id':f'CFD_{i+1:02d}','original_size':[32,16],'roi':[4,2,20,12],
          'foreground_rule':{'mode':'binary','evaluation_threshold':127},
          'rgb_pixel_sha256':tool.shared.digest(rgb.tobytes()),'gt_binary_pixel_sha256':tool.shared.digest(gt.tobytes()),
          'image_file_sha256':tool.shared.file_hash(ip),'mask_file_sha256':tool.shared.file_hash(mp),
          'image_relative_path':ip.relative_to(data).as_posix(),'mask_relative_path':mp.relative_to(data).as_posix()})
    manifest={'schema_version':1,'status':'frozen_partial','selection_seed':42,
              'datasets':{'CFD':{'cases':rows,'split_source':{'kind':'synthetic fixture'}}},'pending_items':[]}
    manifest['manifest_sha256']=tool.hashed(manifest)
    tool.save_json(root/'shared_cases.json',manifest)
    roots={'root':str(data),'image_dir':'test/images','mask_dir':'test/masks'}
    return manifest,roots


def test_manifest_test_membership_and_pixel_identity():
    with tempfile.TemporaryDirectory(dir=REPO/'tmp') as tmp:
        root=Path(tmp); manifest,roots=fixture(root)
        assert tool.read_manifest(root/'shared_cases.json')==manifest
        data=tool.verify_cases('CFD',manifest['datasets']['CFD'],roots,{'data':{'threshold':127}})
        assert len(data)==6
        first=manifest['datasets']['CFD']['cases'][0]
        # Different lossless encoding is allowed, but never different decoded pixels.
        old=Path(roots['root'])/first['image_relative_path']; new=old.with_suffix('.png')
        Image.open(old).save(new); old.unlink()
        data=tool.verify_cases('CFD',manifest['datasets']['CFD'],roots,{'data':{'threshold':127}})
        assert data[0][3]['source_encoding_changed_pixels_identical']['image']
        bad=copy.deepcopy(roots); bad['image_dir']='val/images'
        rejects(lambda:tool.verify_cases('CFD',manifest['datasets']['CFD'],bad,{'data':{'threshold':127}}),'train/val')
        bad=copy.deepcopy(manifest); bad['datasets']['CFD']['cases'][0]['split']='val'
        bad['manifest_sha256']=tool.hashed({k:v for k,v in bad.items() if k!='manifest_sha256'})
        tool.save_json(root/'bad.json',bad)
        rejects(lambda:tool.read_manifest(root/'bad.json'),'Non-test')
        rgb=np.array(Image.open(new)); rgb[0,0]=255; Image.fromarray(rgb).save(new)
        rejects(lambda:tool.verify_cases('CFD',manifest['datasets']['CFD'],roots,{'data':{'threshold':127}}),'RGB identity')


def test_geometry_arrays_error_colors_completion_and_cpu_render():
    with tempfile.TemporaryDirectory(dir=REPO/'tmp') as tmp:
        root=Path(tmp); manifest,roots=fixture(root)
        verified=tool.verify_cases('CFD',manifest['datasets']['CFD'],roots,{'data':{'threshold':127}})
        directory=root/'out/CFD'; codes=tool.code_hashes()
        for row,rgb,gt,source in verified:
            geom=tool.geometry(row['original_size'],[48,64])
            assert geom['top']==8 and geom['resized_hw']==[32,64]
            canvas=tool.input_canvas(rgb,geom)
            assert not canvas[:8].any() and not canvas[-8:].any()
            prob=np.full((48,64),.5,np.float32)
            # Non-contiguous original-size probability must be accepted and stored contiguous.
            display=np.full((32,16),.25,np.float32).T; display[3:12,8]=.75
            assert not display.flags.c_contiguous
            identity={'manifest_sha256':manifest['manifest_sha256'],'case_sha256':tool.hashed(row),'code_hashes':codes}
            tool.write_case(directory/row['case_id'],row,rgb,gt,source,{'synthetic':True},identity,
                            np.zeros_like(prob),prob,display,np.zeros_like(prob,dtype=np.uint8),geom)
            assert tool.completed(directory/row['case_id'],tool.hashed(identity))
            assert np.load(directory/row['case_id']/'probability.npy').flags.c_contiguous
            assert not np.array(Image.open(directory/row['case_id']/'eval_prediction.png')).any(),'> threshold, not >='
        colors=tool.error_image(np.array([[255,255],[0,0]],np.uint8),np.array([[255,0],[255,0]],np.uint8))
        assert colors.tolist()==[[list(tool.COLORS['TP']),list(tool.COLORS['FP'])],[list(tool.COLORS['FN']),list(tool.COLORS['TN'])]]
        rejects(lambda:tool.checked_array(np.array([np.nan])),'Non-finite')
        rejects(lambda:tool.checked_array(np.array([1.01]),True),'out-of-range')
        try:
            import matplotlib
        except ImportError:
            print('SKIP: CPU render (matplotlib unavailable)')
        else:
            base=Path(tool.render_dataset(directory,manifest['datasets']['CFD'],manifest['manifest_sha256'],codes))
            assert base.with_suffix('.pdf').is_file() and base.with_suffix('.png').is_file()
            # Keep an explicitly synthetic preview for visual QA, never as a method prediction.
            import shutil
            target=REPO/'outputs/sota_export_synthetic_preview'; target.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(base.with_suffix('.png'),target/'synthetic_preview.png')
        rejects(lambda:tool.completed(directory/'CFD_01','wrong fingerprint'),'provenance mismatch')
        (directory/'CFD_01'/'prediction.png').unlink()
        rejects(lambda:tool.completed(directory/'CFD_01'),'corrupted')


def test_registered_six_final_run_files_and_log_provenance():
    try: import yaml
    except ImportError:
        print('SKIP: final-run config audit requires PyYAML'); return
    runs=tool.load(REPO/'configs/qualitative/demamba_final_runs.json')
    assert set(runs['datasets'])==set(tool.DATASETS)
    paths=set()
    for name,run in runs['datasets'].items():
        paths.add(run['checkpoint'])
        try: cfg,result=tool.audit_run(name,run,runs['protocol'])
        except FileNotFoundError as error:
            print('SKIP: unavailable registered run',name,error); continue
        assert result['weight_type'].startswith('EMA') and result['training_seed']==42
        assert result['status']=='files_config_log_passed_NOT_model_loaded'
        bad=copy.deepcopy(run); bad['img_size']=[512,512]
        rejects(lambda:tool.audit_run(name,bad,runs['protocol']),'Input size')
        bad=copy.deepcopy(run); bad['paper_metrics']['F1']=0.
        rejects(lambda:tool.audit_run(name,bad,runs['protocol']),'provenance metric')
        print('PASS: final config/log/weight-file audit',name,result['selected_update'])
    assert len(paths)==6,'One checkpoint cannot serve six datasets'


def test_checkpoint_metadata_wrong_run_and_score_rejected_without_torch():
    resolved={'final_checkpoint_type':'significant_best','final_checkpoint_update':123,
              'final_checkpoint_score':.8,'resume_identity':{'dataset_fingerprint':'known'}}
    checkpoint={'checkpoint_type':'significant_best','best_structural_score':.8,
                'auto_protocol_state':{'global_update':123,'significant_best_update':123,
                                       'resume_identity':{'dataset_fingerprint':'known'}}}
    assert tool.validate_checkpoint_metadata(checkpoint,{'auto_protocol_resolved':resolved})['checkpoint_metadata_verified']
    bad=copy.deepcopy(checkpoint); bad['auto_protocol_state']['global_update']=124
    rejects(lambda:tool.validate_checkpoint_metadata(bad,{'auto_protocol_resolved':resolved}),'update metadata')
    bad=copy.deepcopy(checkpoint); bad['best_structural_score']=.9
    rejects(lambda:tool.validate_checkpoint_metadata(bad,{'auto_protocol_resolved':resolved}),'score metadata')
    bad=copy.deepcopy(checkpoint); bad['auto_protocol_state']['resume_identity']['dataset_fingerprint']='other dataset'
    rejects(lambda:tool.validate_checkpoint_metadata(bad,{'auto_protocol_resolved':resolved}),'resume_identity')


if __name__=='__main__':
    (REPO/'tmp').mkdir(exist_ok=True)
    for test in (test_manifest_test_membership_and_pixel_identity,
                 test_geometry_arrays_error_colors_completion_and_cpu_render,
                 test_checkpoint_metadata_wrong_run_and_score_rejected_without_torch,
                 test_registered_six_final_run_files_and_log_provenance):
        test(); print('PASS:',test.__name__)
    print('CUDA strict checkpoint load/forward NOT executed by this CPU suite.')
