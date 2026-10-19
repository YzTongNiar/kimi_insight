import csv, glob, os, re, json, html
from collections import defaultdict

BASE = r"C:\Users\Jarvis\Desktop\kimi_insight\profiling"
OUT = r"C:\Users\Jarvis\Desktop\kimi_insight\reports\cannbot"
PHASES = {
    "P (prefill)": os.path.join(BASE, "kimik3_A3_prefill", "ASCEND_PROFILER_OUTPUT"),
    "D (decode)": glob.glob(os.path.join(BASE, "kimik3_A5_decode", "**", "ASCEND_PROFILER_OUTPUT"), recursive=True)[0],
}

ATTN_RE = re.compile(r"attn|attention|mla|flashmla|fusedinferattention|attnres|kda|causalconv|recurrentkda|gdn|latent", re.I)
MOE_RE = re.compile(r"moe|expert|groupedmatmul|gating|tokenpermute|tokunpermute|distribute|alltoall|dispatch|combine", re.I)
COMM_RE = re.compile(r"hcom|hccl|allgather|all_gather|reduce.?scatter|reducescatter|all.?to.?all|broadcast|send|recv|communication", re.I)

def fnum(x):
    try: return float(str(x).strip().replace('\t',''))
    except: return 0.0

def pct(x, total): return 100.0*x/total if total else 0.0

def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))

def classify(name, typ='', core=''):
    s = f"{name} {typ}"
    if COMM_RE.search(s) or core.upper() == 'COMMUNICATION': return 'communication'
    if MOE_RE.search(s): return 'moe'
    if ATTN_RE.search(s): return 'attention'
    return 'other'

def core_class(row):
    core=(row.get('Accelerator Core') or '').upper()
    name=row.get('Name') or ''
    typ=row.get('Type') or ''
    if COMM_RE.search(f'{name} {typ}') or core == 'COMMUNICATION': return 'communication'
    if core == 'AI_CPU' and COMM_RE.search(f'{name} {typ}'): return 'communication'
    return 'compute_memory'

def load_phase(label, outdir):
    kd=read_csv(os.path.join(outdir,'kernel_details.csv'))
    ops=read_csv(os.path.join(outdir,'op_statistic.csv'))
    st=read_csv(os.path.join(outdir,'step_trace_time.csv'))
    step=st[0] if st else {}
    stage=fnum(step.get('Stage'))
    pipe=defaultdict(float)
    comp=defaultdict(float)
    rows_by_comp=defaultdict(list)
    for r in kd:
        name=r.get('Name',''); typ=r.get('Type',''); core=r.get('Accelerator Core','')
        dur=fnum(r.get('Duration(us)'))
        cc=core_class(r)
        if cc=='communication':
            pipe['communication'] += dur
        else:
            # PipeUtilization counters. These are overlapping unit times, normalized below.
            pipe['compute'] += fnum(r.get('aic_mac_time(us)')) + fnum(r.get('aic_scalar_time(us)')) + fnum(r.get('aiv_vec_time(us)')) + fnum(r.get('aiv_scalar_time(us)'))
            pipe['memory'] += fnum(r.get('aic_mte1_time(us)')) + fnum(r.get('aic_mte2_time(us)')) + fnum(r.get('aiv_mte2_time(us)')) + fnum(r.get('aiv_mte3_time(us)')) + fnum(r.get('aic_fixpipe_time(us)'))
        c=classify(name,typ,core)
        comp[c] += dur
        rows_by_comp[c].append(r)
    # operator statistic time, classified by OP Type/name
    op_comp=defaultdict(float); op_rows=defaultdict(list)
    for r in ops:
        name=r.get('OP Type') or r.get('Name') or ''
        typ=r.get('OP Type','')
        val=fnum(r.get('Total Time(us)'))
        c=classify(name,typ,r.get('Core Type',''))
        op_comp[c]+=val; op_rows[c].append((name,val,int(fnum(r.get('Count')))))
    pipe_total=sum(pipe.values())
    # derive step trace direct wall shares
    trace={k:fnum(step.get(k)) for k in ['Computing','Communication','Communication(Not Overlapped)','Overlapped','Free','Stage','Preparing']}
    top_ops=[]
    for r in sorted(ops,key=lambda x:fnum(x.get('Total Time(us)')), reverse=True)[:30]:
        name=r.get('OP Type') or r.get('Name') or ''
        c=classify(name,r.get('OP Type',''),r.get('Core Type',''))
        top_ops.append({'name':name,'time_us':fnum(r.get('Total Time(us)')),'count':int(fnum(r.get('Count'))),'ratio':fnum(r.get('Ratio(%)')),'component':c,'core':r.get('Core Type','')})
    comp_top={}
    for c in ['attention','moe','communication','other']:
        comp_top[c]=sorted([x for x in top_ops if x['component']==c], key=lambda x:x['time_us'], reverse=True)[:10]
    return {'label':label,'outdir':outdir,'kernel_rows':len(kd),'op_rows':len(ops),'trace':trace,'pipe':dict(pipe),'pipe_total':pipe_total,'kernel_comp':dict(comp),'op_comp':dict(op_comp),'top_ops':top_ops,'comp_top':comp_top}

results=[load_phase(k,v) for k,v in PHASES.items()]
# Generate Markdown
md=[]
md.append('# Kimi K3 Prefill/Decode Profiling 负载分析报告')
md.append('')
md.append('> 数据来源：`C:\\Users\\Jarvis\\Desktop\\kimi_insight\\profiling`；按 `model-infer-profiling` 与 `model-infer-perf-breakdown` 的 kernel / step / operator 口径分析。')
md.append('')
md.append('## 口径与限制')
md.append('')
md.append('- P = `kimik3_A3_prefill`，D = `kimik3_A5_decode`。')
md.append('- 访存/计算/通信占比使用 `kernel_details.csv` 的 PipeUtilization 单元时间：计算 = MAC/Vector/Scalar，访存 = MTE1/MTE2/MTE3/FixPipe，通信 = COMMUNICATION/HCCL 核。由于流水单元可重叠，这一组比例表示**设备流水负载构成**，不是互斥墙钟时间。')
md.append('- P/D 的墙钟阶段占比使用 `step_trace_time.csv` 的 `Stage`、`Computing`、`Communication`、`Overlapped`。')
md.append('- Attention 包含 MLA/FlashMLA/Attention、KDA、CausalConv、RecurrentKda 等；MoE 包含 MoE 路由、Dispatch/Combine、Token Permute/Unpermute、GroupedMatmul/Expert 相关核。分类依据算子名与 Kimi K3 结构代码，未匹配项归为 Other。')
md.append('')
md.append('## 阶段总览')
md.append('')
md.append('| 阶段 | kernel 行数 | Stage (ms) | Computing / Stage | Communication / Stage | Overlapped / Stage | Free / Stage |')
md.append('|---|---:|---:|---:|---:|---:|---:|')
for x in results:
 t=x['trace']; stage=t['Stage']; md.append(f"| {x['label']} | {x['kernel_rows']:,} | {stage/1000:.3f} | {pct(t['Computing'],stage):.1f}% | {pct(t['Communication'],stage):.1f}% | {pct(t['Overlapped'],stage):.1f}% | {pct(t['Free'],stage):.1f}% |")
md.append('')
md.append('## 访存 / 计算 / 通信负载构成')
md.append('')
md.append('| 阶段 | 访存流水时间 (us) | 计算流水时间 (us) | 通信核时间 (us) | 访存 | 计算 | 通信 |')
md.append('|---|---:|---:|---:|---:|---:|---:|')
for x in results:
 p=x['pipe']; total=x['pipe_total']; md.append(f"| {x['label']} | {p.get('memory',0):,.0f} | {p.get('compute',0):,.0f} | {p.get('communication',0):,.0f} | {pct(p.get('memory',0),total):.1f}% | {pct(p.get('compute',0),total):.1f}% | {pct(p.get('communication',0),total):.1f}% |")
md.append('')
md.append('## Attention / MoE 负载与算子耗时')
md.append('')
md.append('| 阶段 | Attention kernel 时间 (ms) | MoE kernel 时间 (ms) | A/M kernel 中 Attention | A/M kernel 中 MoE | Attention operator 时间 (ms) | MoE operator 时间 (ms) | 占全部 operator Attention | 占全部 operator MoE | A/M operator 中 Attention | A/M operator 中 MoE |')
md.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for x in results:
 kc=x['kernel_comp']; oc=x['op_comp']; kt=kc.get('attention',0)+kc.get('moe',0); ot=oc.get('attention',0)+oc.get('moe',0); all_ot=sum(oc.values())
 md.append(f"| {x['label']} | {kc.get('attention',0)/1000:.3f} | {kc.get('moe',0)/1000:.3f} | {pct(kc.get('attention',0),kt):.1f}% | {pct(kc.get('moe',0),kt):.1f}% | {oc.get('attention',0)/1000:.3f} | {oc.get('moe',0)/1000:.3f} | {pct(oc.get('attention',0),all_ot):.1f}% | {pct(oc.get('moe',0),all_ot):.1f}% | {pct(oc.get('attention',0),ot):.1f}% | {pct(oc.get('moe',0),ot):.1f}% |")
md.append('')
md.append('## 主要算子')
md.append('')
for x in results:
 md.append(f"### {x['label']}")
 md.append('')
 md.append('| 算子 | 类型 | 核心 | 次数 | 总耗时 (ms) | 占 op_statistic |')
 md.append('|---|---|---|---:|---:|---:|')
 for o in x['top_ops'][:15]: md.append(f"| `{o['name']}` | {o['component']} | {o['core']} | {o['count']:,} | {o['time_us']/1000:.3f} | {o['ratio']:.2f}% |")
 md.append('')
md.append('## 瓶颈判断与优化建议')
md.append('')
for x in results:
 t=x['trace']; p=x['pipe']; kc=x['kernel_comp']; oc=x['op_comp']; kt=kc.get('attention',0)+kc.get('moe',0); ot=oc.get('attention',0)+oc.get('moe',0)
 md.append(f"### {x['label']}")
 md.append('')
 if t['Communication']>t['Computing']:
  md.append(f"- 阶段级瓶颈偏通信：通信墙钟占 Stage **{pct(t['Communication'],t['Stage']):.1f}%**，高于计算的 **{pct(t['Computing'],t['Stage']):.1f}%**；应优先检查 HCCL all-gather/reduce-scatter/all-to-all 的消息量、并行拓扑和通信计算重叠。")
 else:
  md.append(f"- 阶段级瓶颈偏计算：计算墙钟占 Stage **{pct(t['Computing'],t['Stage']):.1f}%**；应优先优化高耗时矩阵乘、量化/反量化及 Attention/MoE 主核。")
 if kc.get('moe',0)>kc.get('attention',0):
  md.append(f"- kernel 负载中 MoE 更重（{pct(kc.get('moe',0),kt):.1f}% vs Attention {pct(kc.get('attention',0),kt):.1f}%）。重点关注 MoE 路由、Dispatch/Combine、GroupedMatmul、Token Permute/Unpermute，以及其前后的量化核。")
 else:
  md.append(f"- kernel 负载中 Attention 更重（{pct(kc.get('attention',0),kt):.1f}% vs MoE {pct(kc.get('moe',0),kt):.1f}%）。重点关注 MLA/FlashMLA、KDA/RecurrentKda、Attention residual/prolog，以及 KV cache 访存。")
 if oc.get('moe',0)>oc.get('attention',0):
  md.append(f"- `op_statistic` 的 MoE 算子耗时占 Attention+MoE **{pct(oc.get('moe',0),ot):.1f}%**，与 kernel 结论一致，建议先做 MoE 通信和 GroupedMatmul 融合/量化路径优化。")
 else:
  md.append(f"- `op_statistic` 的 Attention 算子耗时占 Attention+MoE **{pct(oc.get('attention',0),ot):.1f}%**，建议先做 FlashMLA/Attention 主核、KV cache 访问和 KDA 相关算子优化。")
 md.append('')
# Add machine-readable data
with open(os.path.join(OUT,'analysis_data.json'),'w',encoding='utf-8') as fh: json.dump(results,fh,ensure_ascii=False,indent=2)
with open(os.path.join(OUT,'k3_profiling_report.md'),'w',encoding='utf-8') as fh: fh.write('\n'.join(md)+'\n')
# simple styled HTML from markdown-ish sections/tables
html_parts=['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Kimi K3 Profiling Report</title><style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;max-width:1400px;margin:32px auto;padding:0 24px;color:#1f2937}h1{color:#0f3d5e}h2{border-bottom:2px solid #dbeafe;padding-bottom:6px}h3{color:#075985}table{border-collapse:collapse;width:100%;margin:12px 0 24px;font-size:14px}th,td{border:1px solid #cbd5e1;padding:7px 9px;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#e0f2fe}tr:nth-child(even){background:#f8fafc}code{background:#f1f5f9;padding:2px 4px;border-radius:3px}.note{background:#fff7ed;border-left:4px solid #f97316;padding:10px 14px}</style></head><body>']
for line in md:
 if line.startswith('# '): html_parts.append('<h1>'+html.escape(line[2:])+'</h1>')
 elif line.startswith('## '): html_parts.append('<h2>'+html.escape(line[3:])+'</h2>')
 elif line.startswith('### '): html_parts.append('<h3>'+html.escape(line[4:])+'</h3>')
 elif line.startswith('> '): html_parts.append('<div class="note">'+html.escape(line[2:])+'</div>')
 elif line.startswith('- '): html_parts.append('<li>'+html.escape(line[2:])+'</li>')
 elif line.startswith('|'):
  cells=[c.strip() for c in line.strip('|').split('|')]
  if all(set(c)<=set('-: ') for c in cells): continue
  tag='th' if not any(ch.isdigit() for ch in ''.join(cells)) or (cells and cells[0] in ['阶段','算子']) else 'td'
  html_parts.append('<tr>'+''.join(f'<{tag}>'+html.escape(c)+'</{tag}>' for c in cells)+'</tr>')
 elif not line.strip():
  pass
 else:
  if html_parts and html_parts[-1].startswith('<tr>') and not line.startswith('|'): html_parts.append('</table>')
  html_parts.append('<p>'+html.escape(line)+'</p>')
# Wrap table rows in table tags by rebuilding simpler
body=[]; in_table=False
for part in html_parts[1:]:
 if part.startswith('<tr>') and not in_table: body.append('<table>'); in_table=True
 if not part.startswith('<tr>') and in_table: body.append('</table>'); in_table=False
 body.append(part)
if in_table: body.append('</table>')
body.append('</body></html>')
with open(os.path.join(OUT,'k3_profiling_report.html'),'w',encoding='utf-8') as fh: fh.write(''.join(html_parts[:1]+body))
print(json.dumps({'out':OUT,'reports':['k3_profiling_report.md','k3_profiling_report.html'],'phases':[(x['label'],x['kernel_rows']) for x in results]},ensure_ascii=False))
