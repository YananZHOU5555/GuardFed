"""Render saved descriptive statistics only."""
import json
from pathlib import Path
H=Path(__file__).resolve().parent
j=json.loads((H/'statistics.json').read_text());p=j['panels']['ten'];checks=json.loads((H/'verification.json').read_text())
M=['accuracy','aeod','aspd'];V=['raw','native','shared_calibration']

def triple(stats,signed=False):
    return ' / '.join((f"{stats[k]['mean']*(100 if k=='accuracy' else 1):+.4f}" if signed else f"{stats[k]['mean']*(100 if k=='accuracy' else 1):.4f}")+' ± '+f"{stats[k]['sample_sd']*(100 if k=='accuracy' else 1):.4f}" for k in M)
lines=['# 九方法三视图：校准相关差异的描述性解释','',
'共享校准后，GuardFed-AD2+ 的跨场景平均 ACC 仍高于7/8个基线，但 AEOD 更低的只有4/8、ASPD 更低的只有1/8。原生视图对应6/8、8/8、7/8。**当前数字不支持“共享校准后三指标全面领先”，也不能把原生公平性优势全部归因于聚合机制。**','',
'本报告只整理已接受的900个终轮模型记录，每条三视图来自同一checkpoint。每方法、每seed先等权平均 IID/non-IID × 五场景共10格，再跨seed计算均值±样本SD（ddof=1）；统计样本量是10、9或6，场景不是独立seed。均值差和差值SD均按相同seed配对计算。没有新增显著性检验、CI、score或最佳seed选择，没有指定正式主终点。','',
'原raw：全部argmax；原native：GuardFed使用训练root组阈值、八基线argmax；shared：九方法均使用既存的同一规则、各自模型训练root拟合组阈值。本次无重新拟合。GuardFed的100条native/shared完整view字典完全一致；八基线的raw/native三指标逐条完全一致。因此native→shared的相对排名变化来自比较基线也接受校准，不是GuardFed再次改善。','',
'## 原指标：10seed','',
'每格依次为 **ACC(%) / AEOD / ASPD**，均为均值±跨seed样本SD。AEOD实际为绝对TPR差，非完整equalized-odds差；两个gap保持0–1原尺度。','',
'| 方法 | raw | native | shared calibration |','|---|---|---|---|']
for m,views in p['method_statistics'].items():lines.append('| '+m+' | '+' | '.join(triple(views[v]) for v in V)+' |')
lines+=['','## 同seed校准变化：10seed','','下表为 **目标视图−raw**；ACC单位为百分点，正值更好；AEOD/ASPD为原尺度，负值更好。表内仍按ACC / AEOD / ASPD排列，展示均值±配对差值样本SD。全部负结果保留。','','| 方法 | raw→native | raw→shared |','|---|---|---|']
for m,views in p['view_changes_target_minus_raw'].items():lines.append('| '+m+' | '+triple(views['native'],True)+' | '+triple(views['shared_calibration'],True)+' |')
lines+=['','## GuardFed相对每个基线：10seed','','三指标统一为**正值表示GuardFed较好**：ACC=GuardFed−基线（百分点）；AEOD/ASPD=基线−GuardFed（原尺度）。每格为同seed差值的均值±样本SD；负号不能省略。','','| 对比基线 | raw | native | shared calibration |','|---|---|---|---|']
for m,views in p['GuardFed_advantage'].items():lines.append('| '+m+' | '+' | '.join(triple(views[v],True) for v in V)+' |')
lines+=['','## 哪些差异保留，哪些与后处理有关','','- GuardFed自身从raw到native/shared：ACC **−0.3191±0.0590个百分点**，AEOD **−0.026475±0.005444**，ASPD **−0.039979±0.005514**。这是固定终轮模型上应用既存阈值规则的实测差异，可描述其后处理贡献；不能推出训练聚合组件的因果贡献。',
'- 共享校准后，ACC仍高于除FLTrust外的全部基线；对FLTrust为 **−0.8445个百分点**。对FedAA-DDPG从native的 **−0.2036个百分点**转为shared的 **+0.1517个百分点**，来源是两模型校准代价不同，不是新增训练。',
'- shared下AEOD仍低于FedAvg、Median、FedAA-DDPG及FLTrust+FairGuard；相对FairFed、FLTrust、LASA、FairGuard则更高。其中对FLTrust的优势从native **+0.036181**变成shared **−0.000429**。这些小差异仅为描述性均值差，不代表显著差异。',
'- shared下ASPD仅低于FLTrust，对其余七个基线均更高。例如对Median从native优势 **+0.031756**变成shared劣势 **−0.009598**；对FairGuard劣势从 **−0.004272**扩大为 **−0.030080**。',
'- raw下已存在的部分准确率/AEOD优势与最终训练模型有关，但模型同时包含完整训练流程、配置选择、候选选择与其他干预；不能由此隔离纯聚合的因果作用。公平性小也可能伴随性能退化，不能只按gap排名。','',
'## 固定9seed与6seed敏感性面板','','完整逐方法均值/SD及配对差均保存在statistics.json；这里汇报正向平均差涉及的基线数量（分母8），依次ACC / AEOD / ASPD。这些是方向计数，不是seed胜率或显著性。','','| 固定seed面板 | raw | native | shared calibration |','|---|---|---|---|']
for name,panel in j['panels'].items():
    labels={'ten':'91001–91010，n=10','nonselection_nine':'91002–91010，n=9','matching_six':'91005–91010，n=6'}
    counts=panel['GuardFed_positive_mean_advantage_baseline_count']
    lines.append('| '+labels[name]+' | '+' | '.join(' / '.join(str(counts[v][k])+'/8' for k in M) for v in V)+' |')
lines+=['','10/9/6面板均保留“shared ACC强、ASPD不全面占优”的方向；AEOD正向对比在9seed为5/8，在10/6seed为4/8，不能据此挑选更有利面板。91001曾参与选择；9/6排除它仍是已暴露validation，不能改称未触碰test或独立确认集。','',
'## 身份、限制与复算','','- 输入records SHA：`'+j['source_records_sha256']+'`；原ROOT_REVIEW、source seal、别名与种子定义均由INPUTS.json绑定。本次不重复88归档审计，只依赖原已接受900记录并重新计算描述统计。',
'- 原训练环境886条cu128、14条cu130；推理434 CPU、466 GPU，driver/runtime按原逐ID来源保留，没有统一环境或跨设备训练等价主张。FairFed/FairGuard/组合等适配身份沿用原表，不能将适配结果冒称完整原论文实现。',
'- raw/native各29条、shared28条常量预测仍在统计中。公平性零差不等于有效预测；FairGuard原CPU失败记录保持无效，使用的是原链已明确接受的GPU诊断记录。本报告未重新采纳任何旧失败。',
'- 独立NumPy从原900记录重建数组，与纯statistics均值/样本SD核对 **'+str(checks['scalar_checks'])+'标量**，最大绝对差 `'+str(checks['max_abs_difference'])+'`；另核原native配对汇总162标量。没有修改科学容差。',
'- 本报告不改变原预测、阈值、统计选择、主终点或表格。低gap/均值优势不能替代机制消融、配对差异解释和最终冻结评价。','',
'复算（项目根目录，仅写本目录）：','','```powershell','python -B tmp/celeba_nine_method_view_attribution_20261009/recompute.py','python -B tmp/celeba_nine_method_view_attribution_20261009/render_report.py','```','','机器可读：statistics.json包含10/9/6全部视图、校准差、GuardFed对八基线的配对统计；per_seed_ten_scene_means.json保留每seed的10场景来源ID；paired_per_seed.json保留全部有符号差值；verification.json记录独立复算。']
text='\n'.join(lines)+'\n'
p=H/'REPORT.md'
if p.exists():assert p.read_text(encoding='utf-8')==text,'Refuse altered report'
else:p.write_text(text,encoding='utf-8')
print(str(p))
