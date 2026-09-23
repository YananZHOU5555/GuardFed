from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUT_DIR = Path(__file__).resolve().parent
DOCX_PATH = OUT_DIR / "GuardFed_AD2plus_algorithm_explanation_for_advisor.docx"
MD_PATH = OUT_DIR / "GuardFed_AD2plus_algorithm_explanation_for_advisor.md"


TITLE = "GuardFed-AD2+ 算法说明"
SUBTITLE = "自适应双目标聚合：同时防御性能攻击与公平性攻击"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in [("top", top), ("start", start), ("bottom", bottom), ("end", end)]:
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_cell_width(cell, width_dxa: int) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.first_child_found_in("w:tcW")
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width_dxa))
    tc_w.set(qn("w:type"), "dxa")


def set_row_cant_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def set_font(run, size=None, bold=None, italic=None, color=None, font="Microsoft YaHei") -> None:
    run.font.name = font
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)


def add_paragraph(doc, text="", style=None, bold_prefix: str | None = None):
    p = doc.add_paragraph(style=style)
    if bold_prefix and text.startswith(bold_prefix):
        r1 = p.add_run(bold_prefix)
        set_font(r1, bold=True)
        r2 = p.add_run(text[len(bold_prefix):])
        set_font(r2)
    else:
        r = p.add_run(text)
        set_font(r)
    return p


def add_bullet(doc, text: str):
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(text)
    set_font(r)
    return p


def add_numbered(doc, text: str):
    p = doc.add_paragraph(style="List Number")
    r = p.add_run(text)
    set_font(r)
    return p


def add_equation(doc, eq: str, caption: str | None = None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(eq)
    set_font(r, size=10.5, font="Cambria Math")
    if caption:
        c = doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.paragraph_format.space_after = Pt(8)
        cr = c.add_run(caption)
        set_font(cr, size=9, italic=True, color="666666")


def add_note(doc, title: str, body: str, fill="EEF4FF"):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.allow_autofit = False
    set_row_cant_split(table.rows[0])
    cell = table.cell(0, 0)
    set_cell_shading(cell, fill)
    set_cell_margins(cell, top=110, bottom=110, start=160, end=160)
    set_cell_width(cell, 9120)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(title)
    set_font(r, bold=True, color="17324D")
    p2 = cell.add_paragraph()
    p2.paragraph_format.space_after = Pt(0)
    r2 = p2.add_run(body)
    set_font(r2, size=10)


def add_table(doc, headers, rows, widths):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.allow_autofit = False
    table.style = "Table Grid"
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        set_cell_width(cell, widths[i])
        set_cell_shading(cell, "F2F4F7")
        set_cell_margins(cell)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(header)
        set_font(r, bold=True, size=9.5, color="17324D")
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cell = cells[i]
            set_cell_width(cell, widths[i])
            set_cell_margins(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(str(value))
            set_font(r, size=9)
    return table


def configure_document(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.10

    for style_name, size, color, before, after in [
        ("Heading 1", 16, "2E74B5", 16, 8),
        ("Heading 2", 13, "2E74B5", 12, 6),
        ("Heading 3", 12, "1F4D78", 8, 4),
    ]:
        style = styles[style_name]
        style.font.name = "Microsoft YaHei"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)
        style.font.bold = True
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)


def build_docx() -> None:
    doc = Document()
    configure_document(doc)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(3)
    r = title.add_run(TITLE)
    set_font(r, size=22, bold=True, color="17324D")

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(12)
    sr = subtitle.add_run(SUBTITLE)
    set_font(sr, size=12, color="555555")

    doc.add_heading("1. 一句话概括", level=1)
    add_paragraph(
        doc,
        "GuardFed-AD2+ 是一个自适应双目标聚合算法。它不直接平均客户端更新，而是在每一轮用少量干净 server/root data 检查每个客户端更新是否同时满足“提升性能、不过度破坏公平性、方向不异常、幅度不过大”这几个条件，然后根据综合得分分配聚合权重。",
    )
    add_bullet(doc, "如果一个客户端更新让 clean accuracy 变高，而且 AEOD/ASPD 没有明显变坏，它会获得更高权重。")
    add_bullet(doc, "如果一个客户端更新看起来像性能攻击，例如方向和 clean root update 相反，或者离大多数客户端很远，它会被降权。")
    add_bullet(doc, "如果一个客户端更新导致敏感群体之间的 TPR 或 positive prediction rate 差距变大，它也会被降权。")

    doc.add_heading("2. 为什么需要 AD2+", level=1)
    add_paragraph(
        doc,
        "传统鲁棒聚合方法通常重点防御性能攻击，例如模型投毒、反向更新或离群更新。这类方法可以保护 accuracy，但不一定能保护 fairness。另一方面，公平性算法通常关注不同敏感群体之间的预测差异，但面对 FOE 或 DFA 这类性能攻击时，accuracy 可能明显下降。",
    )
    add_paragraph(
        doc,
        "AD2+ 的目标是同时处理这两类风险：既不能让性能攻击破坏模型可用性，也不能让公平性攻击让某个敏感群体受到系统性伤害。",
    )
    add_note(
        doc,
        "直观例子",
        "假设某一轮有两个客户端更新。客户端 A 让 accuracy 从 83% 提到 84%，但 AEOD 从 0.02 变成 0.12；客户端 B 让 accuracy 从 83% 提到 83.7%，AEOD 只从 0.02 变成 0.03。普通 FedAvg 可能更偏向 A，因为它看起来性能更好；AD2+ 会识别 A 的公平性风险，并更倾向于 B。",
    )

    doc.add_heading("3. 符号定义", level=1)
    add_table(
        doc,
        ["符号", "含义", "解释"],
        [
            ("w^t", "第 t 轮全局模型", "服务器当前持有的模型参数"),
            ("Δ_i^t", "客户端 i 上传的更新", "Δ_i^t = w_i^{t+1} - w^t"),
            ("D_r", "clean server/root data", "服务器保留的一小部分干净数据，含标签和敏感属性"),
            ("a", "敏感属性", "Adult 中可为 sex，COMPAS 中可为 race"),
            ("Δ_r^t", "clean root update", "只用 root data 得到的可信更新方向"),
            ("U_i^t", "utility 分数", "客户端更新在 root data 上的 clean accuracy"),
            ("R_i^t", "fairness risk", "AEOD 和 ASPD 组成的公平风险"),
            ("C_i^t", "centrality", "客户端更新是否靠近正常更新中心"),
            ("A_i^t", "alignment", "客户端更新是否和 clean root update 方向一致"),
            ("s_i^t", "AD2+ 总分", "决定客户端聚合权重的综合得分"),
        ],
        [850, 2150, 6360],
    )

    doc.add_heading("4. 第一步：评估 clean utility", level=1)
    add_paragraph(doc, "服务器先把客户端更新临时作用到当前模型上：")
    add_equation(doc, "w_i^t = w^t + Δ_i^t")
    add_paragraph(doc, "然后在 clean root data 上计算 accuracy：")
    add_equation(doc, "U_i^t = Acc(w_i^t; D_r) = (1 / |D_r|) Σ 1[ŷ_{w_i^t}(x) = y]", "(1) Root utility")
    add_paragraph(
        doc,
        "这里的 U_i^t 不是客户端自己报告的训练准确率，而是服务器用干净 root data 独立评估得到的。因此攻击者不能轻易伪造这个分数。",
    )
    add_note(
        doc,
        "小例子：utility 怎么影响权重",
        "如果客户端 A 的更新在 root data 上得到 84.0% accuracy，而客户端 B 得到 81.0%，在其它风险相近时，A 的 utility 项更高，AD2+ 会给 A 更高的聚合权重。",
    )

    doc.add_heading("5. 第二步：评估 fairness risk", level=1)
    add_paragraph(doc, "AD2+ 同时考虑两类公平性指标。第一类是 AEOD，即两个敏感群体的 TPR 差异：")
    add_equation(doc, "AEOD(w; D_r) = | TPR_{a=0}(w) - TPR_{a=1}(w) |", "(2) Equal opportunity difference")
    add_equation(doc, "TPR_{a=g}(w) = Pr(ŷ_w = 1 | y = 1, a = g)")
    add_paragraph(doc, "第二类是 ASPD，即两个敏感群体被预测为 positive 的概率差异：")
    add_equation(doc, "ASPD(w; D_r) = | Pr(ŷ_w = 1 | a = 0) - Pr(ŷ_w = 1 | a = 1) |", "(3) Statistical parity difference")
    add_paragraph(doc, "当前 AD2+ 使用 act_fairness_metric = aeod_aspd，因此公平风险可以写作：")
    add_equation(doc, "R_i^t = 0.5 · AEOD(w_i^t; D_r) + 0.5 · ASPD(w_i^t; D_r)", "(4) Fairness risk")
    add_note(
        doc,
        "小例子：为什么不能只看 accuracy",
        "假设一个更新让 root accuracy 达到 84%，但 AEOD=0.15、ASPD=0.12；另一个更新让 root accuracy 为 83.5%，但 AEOD=0.02、ASPD=0.03。AD2+ 不会简单选择第一个更新，因为它可能显著伤害某个敏感群体。",
    )

    doc.add_heading("6. 第三步：公平预算与 violation", level=1)
    add_paragraph(doc, "AD2+ 设置一个公平预算 B。只要公平风险没有超过预算，就不额外施加 violation 惩罚；超过预算的部分才会被进一步惩罚：")
    add_equation(doc, "V_i^t = max(0, R_i^t - B)", "(5) Fairness violation")
    add_paragraph(doc, "当前表格中的 AD2+ 配置使用 B = 0.12。这个预算可以理解为“服务器允许的最大公平风险容忍度”。")
    add_note(
        doc,
        "小例子：violation 怎么计算",
        "若某客户端更新的 fairness risk 为 R_i^t=0.08，且 B=0.12，则 V_i^t=0，不额外惩罚。若另一个更新 R_i^t=0.18，则 V_i^t=0.06，表示它超过预算 0.06，需要被额外降权。",
    )

    doc.add_heading("7. 第四步：鲁棒中心性 centrality", level=1)
    add_paragraph(doc, "性能攻击常常表现为更新方向或幅度异常。AD2+ 因此计算每个客户端更新相对于本轮更新中心的距离。令鲁棒中心为：")
    add_equation(doc, "Δ_med^t = median{Δ_1^t, ..., Δ_m^t}")
    add_paragraph(doc, "客户端 i 的离群距离为：")
    add_equation(doc, "d_i^t = || Δ_i^t - Δ_med^t ||_2")
    add_paragraph(doc, "中心性分数可以写成：")
    add_equation(doc, "C_i^t = - d_i^t / (median_j d_j^t + ε)", "(6) Update centrality")
    add_note(
        doc,
        "小例子：识别离群更新",
        "如果 20 个客户端中有 16 个更新方向相近，而 4 个恶意客户端上传反向或极端更新，那么这 4 个更新通常离 median center 更远，C_i^t 更低，从而在 AD2+ 得分中被降权。",
    )

    doc.add_heading("8. 第五步：与 clean root update 的方向一致性", level=1)
    add_paragraph(doc, "服务器还计算每个客户端更新与 clean root update 的 cosine alignment：")
    add_equation(
        doc,
        "A_i^t = cos(Δ_i^t, Δ_r^t) = <Δ_i^t, Δ_r^t> / (||Δ_i^t||_2 ||Δ_r^t||_2 + ε)",
        "(7) Root alignment",
    )
    add_paragraph(doc, "如果一个更新和 clean root update 方向一致，A_i^t 较高；如果方向相反或近似正交，A_i^t 较低。")
    add_note(
        doc,
        "小例子：FOE 攻击",
        "FOE 类攻击会让恶意客户端更新偏离正常优化方向。即使它在某些局部指标上看起来不差，只要它和 clean root update 方向明显不一致，alignment 项就会降低其总分。",
    )

    doc.add_heading("9. 第六步：AD2+ 综合得分", level=1)
    add_paragraph(doc, "AD2+ 将上述信号合成为一个统一客户端得分：")
    add_equation(
        doc,
        "s_i^t = α U_i^t + β C_i^t + γ A_i^t - λ_r R_i^t - λ_v V_i^t",
        "(8) AD2+ score",
    )
    add_paragraph(doc, "这个公式是 AD2+ 的核心。前面三项是奖励项，后面两项是惩罚项：")
    add_bullet(doc, "α U_i^t：奖励 clean root utility 高的更新。")
    add_bullet(doc, "β C_i^t：奖励靠近正常更新中心的更新。")
    add_bullet(doc, "γ A_i^t：奖励和 clean root update 方向一致的更新。")
    add_bullet(doc, "λ_r R_i^t：惩罚公平风险高的更新。")
    add_bullet(doc, "λ_v V_i^t：额外惩罚超过公平预算的更新。")
    add_table(
        doc,
        ["参数", "当前表格配置", "含义"],
        [
            ("B", "0.12", "公平风险预算"),
            ("α", "3.0", "utility 权重"),
            ("β", "0.2", "centrality 权重"),
            ("γ", "COMPAS=1.0, Adult=1.5", "root alignment 权重"),
            ("λ_r", "0.1", "fairness risk 惩罚权重"),
            ("λ_v", "0.02", "violation 惩罚权重"),
            ("τ", "0.8", "softmax temperature"),
            ("c", "5.0", "score clipping 范围"),
            ("norm", "root", "聚合更新范数缩放到 clean root update 尺度"),
        ],
        [1250, 2450, 5660],
    )

    doc.add_heading("10. 第七步：softmax 权重与聚合", level=1)
    add_paragraph(doc, "AD2+ 不一定硬删除客户端，而是用 softmax 将得分转为连续聚合权重。先裁剪得分：")
    add_equation(doc, "s̃_i^t = clip(s_i^t, -c, c)")
    add_paragraph(doc, "然后计算权重：")
    add_equation(doc, "p_i^t = exp(s̃_i^t / τ) / Σ_{j∈K_t} exp(s̃_j^t / τ)", "(9) Softmax aggregation weight")
    add_paragraph(doc, "当前配置 keep=1，因此所有客户端都可以进入候选集合 K_t，但权重不同。最终加权聚合为：")
    add_equation(doc, "Δ_AD2+^t = Σ_{i∈K_t} p_i^t Δ_i^t", "(10) Weighted aggregation")
    add_note(
        doc,
        "小例子：soft weighting 比硬过滤更稳定",
        "假设某客户端只是轻微可疑，而不是明显恶意。硬过滤可能直接丢弃它，导致可用训练信息损失；AD2+ 可以降低它的权重但不完全删除，因此聚合更平滑。",
    )

    doc.add_heading("11. 第八步：root-norm scaling", level=1)
    add_paragraph(doc, "为了避免聚合更新幅度过大，AD2+ 使用 clean root update 的范数进行缩放：")
    add_equation(
        doc,
        "Δ̂_AD2+^t = Δ_AD2+^t · min(1, ||Δ_r^t||_2 / (||Δ_AD2+^t||_2 + ε))",
        "(11) Root-norm scaling",
    )
    add_paragraph(doc, "最后更新全局模型：")
    add_equation(doc, "w^{t+1} = w^t + Δ̂_AD2+^t", "(12) Global update")
    add_note(
        doc,
        "小例子：限制异常大更新",
        "如果恶意客户端让聚合更新范数变得远大于 clean root update，root-norm scaling 会把整体更新缩回到可信尺度，从而减少模型漂移。",
    )

    doc.add_heading("12. 为什么 AD2+ 是 adaptive", level=1)
    add_paragraph(doc, "AD2+ 的 adaptive 不是指每个超参数都自动调参，而是指每一轮的聚合行为会根据当前训练状态重新计算。")
    add_bullet(doc, "每轮重新计算每个客户端的 root utility。")
    add_bullet(doc, "每轮重新计算 AEOD/ASPD 和 fairness violation。")
    add_bullet(doc, "每轮重新计算客户端更新的 centrality 和 alignment。")
    add_bullet(doc, "每轮重新计算 softmax 聚合权重。")
    add_bullet(doc, "每轮根据 clean root update 重新进行 norm scaling。")
    add_paragraph(
        doc,
        "因此，即使 B、α、β、γ、λ_r、λ_v 是固定超参数，AD2+ 的实际客户端权重和全局更新仍然会随攻击强度、数据分布和训练阶段动态变化。",
    )

    doc.add_heading("13. 算法伪代码", level=1)
    pseudo = [
        "Input: global model w^t, client updates {Δ_i^t}, clean root data D_r, clean root update Δ_r^t",
        "For each client i, construct candidate model w_i^t = w^t + Δ_i^t.",
        "Evaluate clean utility U_i^t on D_r.",
        "Compute AEOD and ASPD on D_r, then compute fairness risk R_i^t.",
        "Compute violation V_i^t = max(0, R_i^t - B).",
        "Compute update centrality C_i^t relative to the robust update center.",
        "Compute root alignment A_i^t = cos(Δ_i^t, Δ_r^t).",
        "Compute AD2+ score s_i^t = αU_i^t + βC_i^t + γA_i^t - λ_rR_i^t - λ_vV_i^t.",
        "Convert scores to softmax weights p_i^t.",
        "Aggregate updates and apply root-norm scaling.",
        "Output updated global model w^{t+1}.",
    ]
    for item in pseudo:
        add_numbered(doc, item)

    doc.add_heading("14. 可以放进论文的中文表述", level=1)
    add_paragraph(
        doc,
        "GuardFed-AD2+ 是一种面向性能攻击与公平性攻击的自适应双目标聚合机制。与传统鲁棒聚合方法主要依赖更新几何异常检测不同，GuardFed-AD2+ 在服务器端引入少量干净 root data，对每个客户端更新同时评估其 clean utility 和 fairness risk。具体而言，服务器将每个客户端更新临时作用到当前全局模型上，并在 root data 上计算 accuracy、AEOD 和 ASPD。随后，算法根据公平预算构造 violation term，用于额外惩罚超过可接受公平风险的更新。与此同时，GuardFed-AD2+ 还计算客户端更新相对于本轮更新中心的 centrality，以及其与 clean root update 的 cosine alignment，从而识别方向异常或离群的恶意更新。最终，utility、fairness risk、fairness violation、centrality 和 alignment 被整合为统一的 AD2+ score，并通过 temperature-controlled softmax 转化为聚合权重。聚合后的全局更新进一步通过 clean root update 的范数进行缩放，以限制恶意更新造成的模型漂移。由于上述所有信号均在每一轮根据当前客户端更新和 root-data 反馈重新计算，GuardFed-AD2+ 能够动态调整客户端权重，在保持模型性能的同时抑制公平性退化。",
    )

    doc.add_heading("15. 最后需要注意的边界", level=1)
    add_bullet(doc, "Root data 只用于服务器端评估和校准，不等于把敏感属性作为模型输入特征。")
    add_bullet(doc, "AD2+ 的公平预算 B 是超参数；adaptive 体现在每一轮客户端得分、权重和更新尺度会动态变化。")
    add_bullet(doc, "若需要更强的理论表达，可以把 AD2+ score 解释为 utility-robustness-fairness constrained aggregation 的拉格朗日型近似。")

    footer = doc.sections[0].footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = footer.add_run("GuardFed-AD2+ advisor note")
    set_font(fr, size=8, color="777777")

    doc.save(DOCX_PATH)


def build_markdown() -> None:
    content = """# __TITLE__

**副标题：** __SUBTITLE__

## 1. 一句话概括

GuardFed-AD2+ 是一个自适应双目标聚合算法。它不直接平均客户端更新，而是在每一轮用少量干净 server/root data 检查每个客户端更新是否同时满足“提升性能、不过度破坏公平性、方向不异常、幅度不过大”这几个条件，然后根据综合得分分配聚合权重。

- 如果一个客户端更新让 clean accuracy 变高，而且 AEOD/ASPD 没有明显变坏，它会获得更高权重。
- 如果一个客户端更新看起来像性能攻击，例如方向和 clean root update 相反，或者离大多数客户端很远，它会被降权。
- 如果一个客户端更新导致敏感群体之间的 TPR 或 positive prediction rate 差距变大，它也会被降权。

## 2. 为什么需要 AD2+

传统鲁棒聚合方法通常重点防御性能攻击，例如模型投毒、反向更新或离群更新。这类方法可以保护 accuracy，但不一定能保护 fairness。另一方面，公平性算法通常关注不同敏感群体之间的预测差异，但面对 FOE 或 DFA 这类性能攻击时，accuracy 可能明显下降。

AD2+ 的目标是同时处理这两类风险：既不能让性能攻击破坏模型可用性，也不能让公平性攻击让某个敏感群体受到系统性伤害。

**例子：** 假设某一轮有两个客户端更新。客户端 A 让 accuracy 从 83% 提到 84%，但 AEOD 从 0.02 变成 0.12；客户端 B 让 accuracy 从 83% 提到 83.7%，AEOD 只从 0.02 变成 0.03。普通 FedAvg 可能更偏向 A，因为它看起来性能更好；AD2+ 会识别 A 的公平性风险，并更倾向于 B。

## 3. 符号定义

| 符号 | 含义 | 解释 |
|---|---|---|
| w^t | 第 t 轮全局模型 | 服务器当前持有的模型参数 |
| Δ_i^t | 客户端 i 上传的更新 | Δ_i^t = w_i^{{t+1}} - w^t |
| D_r | clean server/root data | 服务器保留的一小部分干净数据，含标签和敏感属性 |
| a | 敏感属性 | Adult 中可为 sex，COMPAS 中可为 race |
| Δ_r^t | clean root update | 只用 root data 得到的可信更新方向 |
| U_i^t | utility 分数 | 客户端更新在 root data 上的 clean accuracy |
| R_i^t | fairness risk | AEOD 和 ASPD 组成的公平风险 |
| C_i^t | centrality | 客户端更新是否靠近正常更新中心 |
| A_i^t | alignment | 客户端更新是否和 clean root update 方向一致 |
| s_i^t | AD2+ 总分 | 决定客户端聚合权重的综合得分 |

## 4. Clean utility

临时候选模型：

```text
w_i^t = w^t + Δ_i^t
```

Root utility：

```text
U_i^t = Acc(w_i^t; D_r) = (1 / |D_r|) Σ 1[ŷ_{w_i^t}(x) = y]
```

这里的 U_i^t 不是客户端自己报告的训练准确率，而是服务器用干净 root data 独立评估得到的。

## 5. Fairness risk

AEOD：

```text
AEOD(w; D_r) = | TPR_{a=0}(w) - TPR_{a=1}(w) |
TPR_{a=g}(w) = Pr(ŷ_w = 1 | y = 1, a = g)
```

ASPD：

```text
ASPD(w; D_r) = | Pr(ŷ_w = 1 | a = 0) - Pr(ŷ_w = 1 | a = 1) |
```

当前 AD2+ 同时考虑 AEOD 和 ASPD：

```text
R_i^t = 0.5 · AEOD(w_i^t; D_r) + 0.5 · ASPD(w_i^t; D_r)
```

## 6. 公平预算与 violation

```text
V_i^t = max(0, R_i^t - B)
```

当前表格中的 AD2+ 配置使用 B = 0.12。若 R_i^t=0.08，则 V_i^t=0；若 R_i^t=0.18，则 V_i^t=0.06。

## 7. Robust centrality

```text
Δ_med^t = median{{Δ_1^t, ..., Δ_m^t}}
d_i^t = || Δ_i^t - Δ_med^t ||_2
C_i^t = - d_i^t / (median_j d_j^t + ε)
```

越离群的更新，C_i^t 越低。

## 8. Root alignment

```text
A_i^t = cos(Δ_i^t, Δ_r^t)
      = <Δ_i^t, Δ_r^t> / (||Δ_i^t||_2 ||Δ_r^t||_2 + ε)
```

如果一个更新和 clean root update 方向一致，A_i^t 较高；如果方向相反或近似正交，A_i^t 较低。

## 9. AD2+ score

```text
s_i^t = α U_i^t + β C_i^t + γ A_i^t - λ_r R_i^t - λ_v V_i^t
```

当前表格配置：

| 参数 | 当前配置 | 含义 |
|---|---:|---|
| B | 0.12 | 公平风险预算 |
| α | 3.0 | utility 权重 |
| β | 0.2 | centrality 权重 |
| γ | COMPAS=1.0, Adult=1.5 | root alignment 权重 |
| λ_r | 0.1 | fairness risk 惩罚权重 |
| λ_v | 0.02 | violation 惩罚权重 |
| τ | 0.8 | softmax temperature |
| c | 5.0 | score clipping 范围 |
| norm | root | 聚合更新范数缩放到 clean root update 尺度 |

## 10. Softmax aggregation

```text
s̃_i^t = clip(s_i^t, -c, c)
p_i^t = exp(s̃_i^t / τ) / Σ_{j∈K_t} exp(s̃_j^t / τ)
Δ_AD2+^t = Σ_{i∈K_t} p_i^t Δ_i^t
```

当前配置 keep=1，因此所有客户端都可以进入候选集合，但权重不同。

## 11. Root-norm scaling

```text
Δ̂_AD2+^t = Δ_AD2+^t · min(1, ||Δ_r^t||_2 / (||Δ_AD2+^t||_2 + ε))
w^{t+1} = w^t + Δ̂_AD2+^t
```

如果恶意客户端让聚合更新范数变得远大于 clean root update，这一步会把整体更新缩回到可信尺度。

## 12. 为什么 AD2+ 是 adaptive

AD2+ 的 adaptive 不是指每个超参数都自动调参，而是指每一轮的聚合行为会根据当前训练状态重新计算。

- 每轮重新计算每个客户端的 root utility。
- 每轮重新计算 AEOD/ASPD 和 fairness violation。
- 每轮重新计算客户端更新的 centrality 和 alignment。
- 每轮重新计算 softmax 聚合权重。
- 每轮根据 clean root update 重新进行 norm scaling。

因此，即使 B、α、β、γ、λ_r、λ_v 是固定超参数，AD2+ 的实际客户端权重和全局更新仍然会随攻击强度、数据分布和训练阶段动态变化。

## 13. 论文可用中文表述

GuardFed-AD2+ 是一种面向性能攻击与公平性攻击的自适应双目标聚合机制。与传统鲁棒聚合方法主要依赖更新几何异常检测不同，GuardFed-AD2+ 在服务器端引入少量干净 root data，对每个客户端更新同时评估其 clean utility 和 fairness risk。具体而言，服务器将每个客户端更新临时作用到当前全局模型上，并在 root data 上计算 accuracy、AEOD 和 ASPD。随后，算法根据公平预算构造 violation term，用于额外惩罚超过可接受公平风险的更新。与此同时，GuardFed-AD2+ 还计算客户端更新相对于本轮更新中心的 centrality，以及其与 clean root update 的 cosine alignment，从而识别方向异常或离群的恶意更新。最终，utility、fairness risk、fairness violation、centrality 和 alignment 被整合为统一的 AD2+ score，并通过 temperature-controlled softmax 转化为聚合权重。聚合后的全局更新进一步通过 clean root update 的范数进行缩放，以限制恶意更新造成的模型漂移。由于上述所有信号均在每一轮根据当前客户端更新和 root-data 反馈重新计算，GuardFed-AD2+ 能够动态调整客户端权重，在保持模型性能的同时抑制公平性退化。

## 14. 边界说明

- Root data 只用于服务器端评估和校准，不等于把敏感属性作为模型输入特征。
- AD2+ 的公平预算 B 是超参数；adaptive 体现在每一轮客户端得分、权重和更新尺度会动态变化。
- 若需要更强的理论表达，可以把 AD2+ score 解释为 utility-robustness-fairness constrained aggregation 的拉格朗日型近似。
"""
    MD_PATH.write_text(
        content.replace("__TITLE__", TITLE).replace("__SUBTITLE__", SUBTITLE),
        encoding="utf-8",
    )


if __name__ == "__main__":
    build_docx()
    build_markdown()
    print(DOCX_PATH)
    print(MD_PATH)
