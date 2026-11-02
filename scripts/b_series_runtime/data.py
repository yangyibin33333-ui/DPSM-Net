import hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from .preprocessing import _letterbox_image,_letterbox_target,_photometric_augment

MEAN=torch.tensor([.485,.456,.406]).view(3,1,1)
STD=torch.tensor([.229,.224,.225]).view(3,1,1)

def augment_pair(image,mask):
    if random.random()<.5: image,mask=image[:,::-1],mask[:,::-1]
    if random.random()<.5: image,mask=image[::-1],mask[::-1]
    turns=random.choice((0,2))
    if turns: image,mask=np.rot90(image,turns).copy(),np.rot90(mask,turns).copy()
    return np.ascontiguousarray(image),np.ascontiguousarray(mask)

def verify_train_val(root,manifest):
    root=Path(root);counts={}
    # Test contents are deliberately never opened, even for hashing.
    for split in ('train','val'):
        samples=manifest['splits'][split]['samples']
        for kind in ('image','mask'):
            folder=root/split/('images' if kind=='image' else 'masks')
            actual={p.relative_to(root).as_posix() for p in folder.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp')}
            assert actual=={s[kind] for s in samples},(split,kind,'file set differs')
        for sample in samples:
            for kind in ('image','mask'):
                assert hashlib.sha256((root/sample[kind]).read_bytes()).hexdigest()==sample[kind+'_sha256'],sample['id']
        counts[split]=len(samples)
    assert counts=={'train':368,'val':53}
    return counts

class MaskDataset(Dataset):
    def __init__(self,root,split,manifest,config):
        if split not in ('train','val'): raise ValueError('Only train/val authorized')
        self.root=Path(root);self.split=split;self.samples=manifest['splits'][split]['samples'];self.cfg=config
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        s=self.samples[i]
        with Image.open(self.root/s['image']) as im: image=np.asarray(im.convert('RGB'))
        with Image.open(self.root/s['mask']) as im: raw=np.asarray(im.convert('L'))
        mask=(raw>(.5 if raw.max()<=1 else 127)).astype(np.float32)
        if self.split=='train':
            image,mask=augment_pair(image,mask)
            image=_photometric_augment(image,self.cfg['data']['photometric_augment'])
        size=tuple(self.cfg['data']['input_hw'])
        image=_letterbox_image(image,size)
        mask=_letterbox_target(mask[None],size,'nearest')
        tensor=torch.from_numpy(image.transpose(2,0,1).copy()).float()/255.
        return {'image':(tensor-MEAN)/STD,'mask':torch.from_numpy(mask).float(),'id':s['id']}

def build_loader(root,split,manifest,config,generator=None):
    c=config['training'];kw={}
    if c['num_workers']>0: kw={'persistent_workers':True,'prefetch_factor':2}
    if split=='train' and generator is None:
        generator=torch.Generator().manual_seed(config['seed'])
    return DataLoader(MaskDataset(root,split,manifest,config),batch_size=c['micro_batch'],shuffle=split=='train',drop_last=split=='train',num_workers=c['num_workers'],pin_memory=True,generator=generator,**kw)
