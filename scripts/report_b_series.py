"""Refresh only this new B-series report; never changes models/protocol/checkpoints."""
from pathlib import Path
import argparse,csv,hashlib,json,time
import yaml
ROOT=Path(__file__).resolve().parents[1];TAG='b_series_20261023';DOC=ROOT/'docs'/TAG;CFG=ROOT/'configs/experiments'/TAG
NAMES=['B0','B1','B2','B2-Conv']
FIELDS=['model','status','seed','params','conv_macs','conv_flops','best_epoch','actual_updates','IoU','mIoU','Dice','F1','Precision','Recall','ODS','OIS','BoundaryF1','clDice','ComponentError','thin_miss_rate','thin_trigger_images','far_gt_fp_pixels','far_gt_fp_trigger_images','latency_ms','fps','checkpoint_sha256']
def readj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def text(name,s):
    p=DOC/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(s,encoding='utf-8');tmp.replace(p)
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def build():
    cfg=yaml.safe_load((CFG/'common.yaml').read_text(encoding='utf-8'));pre=readj(DOC/'preflight.json');pro=readj(DOC/'source_provenance.json');budget=readj(DOC/'adapter_budget.json')
    with (DOC/'complexity.csv').open(encoding='utf-8-sig') as f:comp={r['model']:r for r in csv.DictReader(f) if r['batch']=='1'}
    states=[];rows=[];finished=True;failed=False
    paused=(DOC/'local_training_paused.json').exists()
    for name in NAMES:
        out=ROOT/cfg['execution']['output_root']/name/'seed42';row={k:'' for k in FIELDS};row.update(model=name,status='not_started',seed=42,**{k:comp[name][k] for k in ['params','conv_macs','conv_flops']})
        last={}
        if (out/'train.jsonl').exists():
            lines=(out/'train.jsonl').read_text(encoding='utf-8').splitlines()
            if lines:
                try:last=json.loads(lines[-1])
                except json.JSONDecodeError:pass
            row.update(status='running',actual_updates=last.get('global_update',''))
        if (out/'best_validation.json').exists() and not(out/'failure.json').exists():
            b=readj(out/'best_validation.json');s=readj(out/'run_summary.json')
            row.update(status='completed',best_epoch=s['best_epoch'],actual_updates=s['actual_updates'],**b['metrics'],thin_miss_rate=b['thin_skeleton_pixel_weighted_miss_rate'],thin_trigger_images=b['thin_trigger_images'],far_gt_fp_pixels=b['far_gt_fp_pixels'],far_gt_fp_trigger_images=b['far_gt_fp_trigger_images'],latency_ms=b['latency_ms'],fps=b['fps'],checkpoint_sha256=b['checkpoint_sha256'])
        elif (out/'failure.json').exists():row['status']='failed';failed=True
        elif (out/'run_summary.json').exists():row['status']='training_finished_evaluation_pending'
        if paused and row['status']=='running':row['status']='paused_local'
        states.append({'model':name,'status':row['status'],'last_epoch':last.get('epoch'),'actual_updates':row['actual_updates']});rows.append(row)
        finished &= row['status']=='completed'
    # This table is re-derived from each run's immutable selected checkpoint evaluation.
    path=DOC/'b_series_results.csv';tmp=path.with_suffix('.csv.tmp')
    # The trainer owns the live CSV. Avoid a concurrent writer while it finishes a model.
    if not path.exists() or finished or failed or paused:
        with tmp.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
        tmp.replace(path)
    gpu=pre.get('gpu',[]);t=cfg['training']
    text('b_series_source_audit.md',f'''# B系列源码与资产审计

当前实际工作区：`{ROOT}`。所有新增源码、配置、测试、报告、训练输出均位于该项目内。

全局标识检索记录：global_source_search.txt。旧项目中的DEMambaNet及F系列通过复杂模型开关组合，不作为Clean CNN。完整Slim实现在项目外 `{list(pro['external_read_only_sources'])[0]}`，其实际class、Forward、配置及哈希已只读核实；external_network_snapshot.py保存其可核验参考，未覆盖或移动外部源码。

## 新增目录与实现

- models/b_series：独立Clean CNN；不改models/de_mambanet.py或models/__init__.py注册表。
- configs/experiments/{TAG}：统一协议、四模型声明、Train/Val冻结manifest和锁。
- scripts/b_series_runtime 与 b_series_study.py：Mask-only Loss、训练/验证/选模组件及正式入口。
- tests/test_b_series.py：四模型Forward/Backward、有限梯度、主Mask Loss与选模回归。
- docs/{TAG}：本组报告与预检；runs/pilot/{TAG}：仅本组新运行。

B0为普通Conv/BN/GELU Encoder–Decoder，单Mask；B1只改变DPER早期证据/半分辨率stage1/重建路径以及对应down1 stride；B2在B1的stage1之后加一次adapted RIFT-SDM。B2-Conv替换完全相同rsdm位置，仍同时流向reconstruct与down1。Decoder和Head逐张量初始化相同。没有LSSCF、Transformer、Mamba、DPSM或辅助Head。

新B0/B1/B2与只读外部Slim进行了相同seed的state_dict逐张量及实际Forward严格相等验证，结果见preflight.json。外部U-Net不实例化、不训练、不加入本组结果。

## 不伪造历史等价性或版本

本组恢复的是完整可验证的最新Slim，不是通过DEMambaNet开关推断。恢复后的普通Decoder仍偏重，不能把“干净CNN”写成“已经满足轻量化预算”。没有中途悄悄换Decoder以改善统计。

Git HEAD：{pro['git'] or '不可用'}。实际Git命令返回：`{pro.get('git_issue')}`。`.git`目录存在并不代表有效仓库；没有创建/改写Git历史。每组source_identity.json记录实际Git不可用原因与全部源码SHA256，不能把源码哈希冒充Git commit。

historical_sources与external_read_only_sources身份见source_provenance.json；统一锁覆盖实际训练源码与配置。禁止覆盖已有run，所有训练只从头初始化。原模型、旧配置、旧checkpoint、外部Slim均保持不变。
''')
    text('rift_sdm_comparison.md','''# adapted RIFT-SDM 对照

官方来源：[RIFT/rift.py](https://github.com/xauat-liushipeng/RIFT/blob/main/rift.py)，核对日期2026-10-23。仅作已有机制适配，不宣称原创。实际网络与调用位置另见model_structure.json、adapter_budget.json。

| 项目 | 官方StructuralDirectionalBlock默认 | 当前B2 |
|---|---|---|
| expand_ratio | 2 | 1（40→40） |
| kernel_size | 11 | 13 |
| dilation | max(1,k//4)，默认2 | 3 |
| local | 3×3 depthwise | 同类 |
| 方向卷积 | 横、纵depthwise及膨胀3×3 | 同类 |
| gate | 对local+direction门控 | 同类 |
| activation | SiLU | GELU |
| normalization | 两处GroupNorm | 两处GN8，C40 |
| DropPath | 支持，block默认0 | 无 |
| residual | 输入加投影分支 | 同类 |
| 输入/位置/次数 | 由RIFT encoder调用确定 | C40，半分辨率Stage1后，1次 |

官方整网另有多stage及非零DropPath调度，本组未复现整网。当前保留的是项目中已存在的适配类，算子AST与原models/blocks.py保持一致。官方main可变化，未取得可验证远程commit/file hash时不伪造，网络读取与本地源码身份情况记录在rift_official_source_identity.json；官方代码另通过web原始源码核实。

B2-Conv不是方向建模：GN40→1×1(40→23)→普通3×3(23→23)→GN23/GELU→1×1(23→40)→残差。所有Conv均groups=1、dilation=1；无StripConv、结构gate、Transformer或Mamba。实际budget见下述结构报告；接近的是Conv MACs，不把未计入GN/GELU的代价假称完整FLOPs相同。
''')
    structure_rows=[[n,comp[n]['params'],f"{int(comp[n]['conv_macs'])/1e9:.9f}",f"{int(comp[n]['conv_flops'])/1e9:.9f}"] for n in NAMES]
    text('b_series_architecture.md','''# B系列实际结构

新模型按核实后的Slim原样恢复：stage通道40/80/160/320；尺度H、H/2、H/4、H/8；每stage两层普通3×3，三处相同concat Decoder与40→1单Mask Head。逐模块结构见model_structure.json。

B1的DPER输入RGB+固定Sobel，full-detail16通道、半分辨率语义40通道；同一stage1改在H/2执行；重建24语义通道+16细节通道形成全分辨率skip。语义主路径继续进入stride1的down1，不从重建f1继续下采样。B0的down1为stride2，两者f2尺寸相同。

B2及B2-Conv均调用`stage1 → rsdm → {reconstruct,down1}`。输入输出均为[1,40,192,272]，实际调用一次。两者的上游、下游、Decoder、Head相同。

'''+table(['Model','Params','Conv MACs(G)','Conv FLOPs(G)'],structure_rows)+f'''

统计采用实际eval Forward、统一[1,3,384,544]，另实际执行Batch8且严格为8倍。1MAC=2FLOPs，含Sobel/分组卷积，不含归一化、激活、插值等完整算子代价。

模块实测：adapted RIFT-SDM的Conv权重6560、总参数6760、Conv FLOPs0.685178880G；普通控制Conv权重6601、总参数6727、Conv FLOPs0.689461248G。Conv MACs相差{100*budget['conv_macs_relative_gap']:.3f}%，总参数相差33个。理论历史增量0.68518G与实际吻合，没有通道/尺度偷换。

轻量化边界：原样恢复的Decoder仍为72.194458G，B0/B1主要瓶颈仍是宽通道拼接后的普通稠密卷积。DPER减少早期算量，不能据此声称整网达到未定义的部署预算。为了保持本次单因素消融，未额外调整Decoder。
''')
    pre_rows=[[r['model'],r['output_shape'],r['successful'],r['finite_gradients'],f"{r['peak_allocated_bytes']/2**30:.3f}"] for r in gpu]
    text('b_series_preflight.md',f'''# 统一预检与训练协议

状态：{pre['status']}。设备：{pre['identity']['environment']['gpu']}。真实FW/BW、Loss、参数更新、相同Decoder/Head、同位置替换、数据无交叉重复、Batch8线性和历史RSDM增量均已通过，详见preflight.json。未构造Test loader。

'''+table(['Model','输出Shape','更新成功','梯度有限','训练单步allocated GiB'],pre_rows)+f'''

统一协议来自实际configs/experiments/auto_protocol/C40_DPSM80_AUTO_PROTOCOL_V2.yaml及train.py、utils/auto_protocol.py，不从聊天填默认值。解析快照resolved_original_protocol.json，配置SHA256 {sha(CFG/'common.yaml')}。

DeepCrack368/53/106，RGB{cfg['data']['input_hw']}；seed42；batch{t['effective_batch']}，micro{t['micro_batch']}，累积{t['accumulation_steps']}；每轮{t['steps_per_epoch']}，最大{t['max_epochs']}轮/目标上限{t['target_updates']}更新；AdamW LR={t['lr']} WD={t['weight_decay']}，fused={t['fused']}，norm/bias不排除WD。BCE+原Dice系数各1，eps/reduction从losses/loss.py直接提取，移除所有结构Loss。

FP16 GradScaler={t['grad_scaler']}；Cosine按成功更新，eta_min={t['eta_min']}；EMA={t['ema']}，从第一个成功更新开始；验证/选模用EMA FP32。AMP跳步不推进EMA/Scheduler/global_update。

验证间隔{t['validation_interval_updates']}成功更新，在epoch边界执行；Structural Score=.5F1+.3clDice+.2BoundaryF1；significant-best阈值{t['early_stopping']['min_delta']}，另存absolute诊断best；Early Stop最少update{t['early_stopping']['min_update']}，patience={t['early_stopping']['patience_updates']}成功更新。best.pth等于significant-best，不按不同指标挑不同checkpoint。

原letterbox+ImageNet归一；同步水平/垂直翻转和0/180度旋转；原Photometric概率.5、brightness/contrast .12、gamma .9–1.1。四模型完全一致；mask正确处理0/1与0/255。所有原53张Val保留，包括此前疑似错位标注。

Train/Val逐文件和解码RGB交叉重复检查为0；Test只做数量元数据核对，未读取其图像/GT。历史Test曾参与开发，本组Val单seed结果不是独立泛化/统计稳定性证据。

原显存门槛源码使用allocated≤总显存×{t['max_memory_fraction']}，reserved同时记录但不能误用为同一门槛。四组均通过。4090D性能只在实际4090D满足条件时测；当前设备非4090D，Latency/FPS保持空值。

运行入口：`python -B scripts/b_series_study.py --mode all --execute`。已有选中输出目录会拒绝覆盖，不支持把重跑当作自动续训。发生任何核心错误时停止后续模型并写failure.json，不自动改配置。
''')
    state_table=table(['Model','状态','已记录Epoch','实际更新'],[[s['model'],s['status'],s['last_epoch'] or '',s['actual_updates']] for s in states])
    analysis=['# B系列实验状态与分析','',state_table,'','所有最终验证指标只从同一个significant-best EMA checkpoint重算，SHA256在CSV及各组best_validation.json中。未完成项保持空值，不将训练中数值当作成功结果。','',
      '1. **B1效率收益**：卷积算量相对B0减少8.973337G（约6.91%），参数增加3464。该计算收益已实测；当前未测4090D延迟，不能声称实际速度必然提升。',
      '6. **主要瓶颈**：B0/B1宽通道concat Decoder的三组普通卷积，共72.194458G；DPER只优化早期路径，未消除此瓶颈。']
    done={r['model']:r for r in rows if r['status']=='completed'}
    if set(['B1','B2'])<=set(done):
        a,b=done['B1'],done['B2'];delta=lambda k:float(b[k])-float(a[k])
        analysis += ['',f"2. **结构质量**：B2相对B1，BoundaryF1变化{100*delta('BoundaryF1'):+.3f}pp，clDice变化{100*delta('clDice'):+.3f}pp，ComponentError变化{delta('ComponentError'):+.3f}；应同时看三项，不仅看Recall。",f"3. **误检与召回**：Recall变化{100*delta('Recall'):+.3f}pp，Precision变化{100*delta('Precision'):+.3f}pp，远GT误检像素变化{delta('far_gt_fp_pixels'):+.0f}；“以误检换召回”的现象判断须依据这组共同变化，不能仅由单项上升下结论。"]
    else:analysis += ['','2–3. B1/B2尚未全部完成，不能判断结构改善或误检–Recall权衡。']
    if set(['B2','B2-Conv'])<=set(done):
        b,k=done['B2'],done['B2-Conv'];score=lambda r:.5*float(r['F1'])+.3*float(r['clDice'])+.2*float(r['BoundaryF1'])
        analysis += ['',f"4. **等预算控制**：B2相对B2-Conv的Structural Score差为{score(b)-score(k):+.6f}，F1差为{100*(float(b['F1'])-float(k['F1'])):+.3f}pp，BoundaryF1差为{100*(float(b['BoundaryF1'])-float(k['BoundaryF1'])):+.3f}pp，clDice差为{100*(float(b['clDice'])-float(k['clDice'])):+.3f}pp，Component Error差为{float(b['ComponentError'])-float(k['ComponentError']):+.3f}，远GT误检像素差为{float(b['far_gt_fp_pixels'])-float(k['far_gt_fp_pixels']):+.0f}。"]
        base=done.get('B1')
        supported=base is not None and score(b)>score(base) and score(b)>score(k) and all(float(b[x])>=float(base[x]) and float(b[x])>=float(k[x]) for x in ['F1','BoundaryF1','clDice']) and float(b['far_gt_fp_pixels'])<=max(float(base['far_gt_fp_pixels']),float(k['far_gt_fp_pixels']))
        if supported:analysis+=['5. **是否保留RSDM**：该单seed同时优于B1和等预算卷积的综合分数、区域/边界/中心线指标，且未出现更大的远GT误检总量，提供继续保留的描述性线索；尚不能宣称统计稳定或独立泛化优势。']
        else:analysis+=['5. **是否保留RSDM**：当前单seed尚不足以同时证明相对B1和等预算卷积的无明显代价结构优势；不应将现有结果写成已证实值得保留。若有部分指标改善，必须同时保留上列F1/误检/结构代价，而非择优汇报。']
        analysis+=['上述判断是冻结实验后的描述性证据解读，不用于改变模型或训练。没有自动推进新模块。']
    else:analysis += ['','4–5. B2与B2-Conv尚未全部完成，不能判断是否优于等预算卷积、是否值得保留RSDM。']
    analysis += ['', '细区域：GT骨架局部EDT半径≤2像素，按骨架像素总量加权漏检；触发规则≥20细点、≥10漏点、漏检率≥.15。远GT误检定义距离GT>3像素，另报告≥25像素且占预测前景≥.1的触发图像数。所有GT均来自当前Val，未用于生成训练权重或修改训练。', '', 'Git HEAD不可用时如实记录null及原因，不伪造commit；每组完整配置、源码SHA256、EMA/raw/optimizer/scheduler/scaler与日志均独立保存。']
    if paused:analysis+=['','本机训练已按用户请求暂停；本机部分run不作为完成实验。服务器使用同步包重新预检后从头训练四组，不自动续接本机checkpoint。尚无四组最终结果或研究结论。']
    elif failed:analysis+=['','出现运行错误，后续模型必须保持停止；详见对应failure.json。']
    elif finished:analysis+=['','四组均完成，正式实验结束。未运行Test、其他seed、独立数据集或B3；不自动设计或训练新模块。']
    else:analysis+=['','四组正式实验仍在进行。当前报告仅记录真实状态，尚未完成的研究结论留空。报告监督器会在本组完成或失败后停止。']
    text('b_series_analysis.md','\n'.join(analysis)+'\n')
    return {'states':states,'completed':finished,'failed':failed,'paused_local':paused}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while True:
        status=build();print(json.dumps(status),flush=True)
        if not args.watch or status['completed'] or status['failed'] or status['paused_local']:break
        time.sleep(15)
if __name__=='__main__':main()
