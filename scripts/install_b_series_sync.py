"""Validate and add only new B-series assets inside the current DE-MambaNet."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,zipfile,subprocess,sys
def digest(data):return hashlib.sha256(data).hexdigest()

def project_path(root,name):
    rel=PurePosixPath(name.replace('\\','/'))
    target=root.joinpath(*rel.parts).resolve()
    if rel.is_absolute() or '..' in rel.parts or ':' in rel.as_posix() or not target.is_relative_to(root):
        raise ValueError('Unsafe project path: '+name)
    return target
def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,default=Path('b_series_server_sync_20261023_v2.zip'));a=p.parse_args()
    root=Path.cwd().resolve()
    assert (root/'train.py').is_file() and all((root/n).is_dir() for n in ['models','configs','scripts','tests','runs']), '请先cd到服务器原DE-MambaNet根目录'
    with zipfile.ZipFile(a.archive) as z:
        meta=json.loads(z.read('sync_manifest.json'));problems=[];pending=[]
        for name,h in meta['protected_source_hashes'].items():
            target=project_path(root,name)
            actual=digest(target.read_bytes()) if target.is_file() else None
            if actual!=h:problems.append({'path':name.replace('\\','/'),'reason':'missing' if actual is None else 'sha256_mismatch','expected_sha256':h,'actual_sha256':actual})
        for name,h in meta['payload_hashes'].items():
            target=project_path(root,name)
            raw=z.read('payload/'+name);assert digest(raw)==h, name
            if target.exists():
                if not target.is_file() or digest(target.read_bytes())!=h:problems.append('已有目标文件不同，拒绝覆盖: '+name)
            else:pending.append((target,raw))
        if problems:
            print(json.dumps({'status':'stopped_before_writes','workspace':str(root),'issues':problems},ensure_ascii=False,indent=2))
            raise SystemExit('停止同步；未覆盖任何文件。请回传差异列表，不要关闭协议锁。')
        for target,raw in pending:
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as out:out.write(raw)
        git=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True)
        workspace_record=root/'docs/b_series_20261023/server_sync_workspace.json'
        if not workspace_record.exists():
            with workspace_record.open('x',encoding='utf-8') as f:
                json.dump({'workspace':str(root),'python':sys.version,'git_head':git.stdout.strip() if git.returncode==0 else None,'git_issue':git.stderr.strip() if git.returncode else None,'protected_source_hashes':meta['protected_source_hashes'],'package_id':meta['package_id']},f,ensure_ascii=False,indent=2)
        print(json.dumps({'status':'passed','workspace':str(root),'new_files':len(pending),'dataset_copied':False,'checkpoints_copied':False,'training_started':False},ensure_ascii=False))
if __name__=='__main__':main()
