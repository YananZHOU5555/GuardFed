# New Baseline References And GuardFed-AD2 Explanation

## New Baseline References

| Method | Concise cite for table | Full reference / status | Link | Why it is relevant |
|---|---|---|---|---|
| FLGMM | Zhu et al., Inf. Fusion'26 | Han-Tao Zhu, Wen-Po Huang, Zhong-Liang Zhang. "Statistical test-based adversarial client detection in federated learning under poisoning attacks." Information Fusion, Vol. 126, Part A, 103569, Feb. 2026. | https://doi.org/10.1016/j.inffus.2025.103569 | Statistical-process-control/GMM based adversarial-client detection. It is a recent robust FL defense against multiple poisoning attacks. |
| FLAURA | Xiao, Sci. Rep.'26 | Yang Xiao. "Adaptive trust evaluation and representation-based robust aggregation against poisoning attacks in federated learning." Scientific Reports, 2026. Published 28 Apr. 2026. | https://doi.org/10.1038/s41598-026-50985-2 | Representation-space robust aggregation using PLR, geometric median, MMD/knee-point trust boundary, hard filtering, and soft weighting. |
| LayerGuard | Wang et al., OpenReview'25 | Kaiqi Wang, Jiangang Shu, Qingfeng Tan, Bo Hu. "LayerGuard: Poisoning-Resilient Federated Learning via Layer-Wise Similarity Analysis." OpenReview submission to NeurIPS 2025, modified 21 Apr. 2026. | https://openreview.net/forum?id=InyYuWLWHD | Layer-wise poisoning defense. Useful because it is designed for advanced/stealthy model poisoning and severe non-IID conditions. |
| SmartFL | Dong et al., Inf. Fusion'26 | Qihao Dong, Yansong Gao, Chunyi Zhou, Shengyuan Yang, Boyu Kuang, Anmin Fu. "SmartFL: Simple majority rule based Byzantine-robust federated learning." Information Fusion, Vol. 126, 103555, Feb. 2026. | https://doi.org/10.1016/j.inffus.2025.103555 | Recent majority-rule based Byzantine defense that does not require a trusted root dataset. Strong comparison for utility robustness. |
| FLTG | Wen et al., arXiv/BlockSys'25 | Yanhua Wen, Lu Ai, Gang Liu, Chuang Li, Jianhao Wei. "FLTG: Byzantine-Robust Federated Learning via Angle-Based Defense and Non-IID-Aware Weighting." arXiv:2505.12851, BlockSys 2025. | https://doi.org/10.48550/arXiv.2505.12851 | Angle-based filtering with server-side clean reference and non-IID-aware weighting. It is close to our setting because it also uses a trusted reference signal. |
| FedDNA | Garg et al., JISA'26 | Aditya Garg, Naman Bansal, Sumit Yadav, Nisha Kandhoul, Sanjay K. Dhurandher, Isaac Woungang. "FedDNA: Behavioural based approach for byzantine defense in federated learning via model fingerprinting and adaptive thresholding." Journal of Information Security and Applications, Vol. 97, 104358, Mar. 2026. | https://doi.org/10.1016/j.jisa.2025.104358 | Behavioral/model-fingerprint defense with MAD adaptive thresholding. Useful recent baseline for adaptive Byzantine filtering. |
| LASA | Xu et al., WACV'25 | Jiahao Xu, Zikai Zhang, Rui Hu. "Achieving Byzantine-Resilient Federated Learning via Layer-Adaptive Sparsified Model Aggregation." WACV 2025. | https://openaccess.thecvf.com/content/WACV2025/papers/Xu_Achieving_Byzantine-Resilient_Federated_Learning_via_Layer-Adaptive_Sparsified_Model_Aggregation_WACV_2025_paper.pdf | Layer-adaptive sparsified aggregation. Strong recent model-poisoning defense under IID and non-IID settings. |

## GuardFed-AD2: What The New Algorithm Is

## GuardFed-AD2 中文详细解释

### 1. 为什么需要 GuardFed-AD2

GuardFed-AD2 的出发点是：在联邦学习里，攻击者不一定只破坏一个目标。

传统鲁棒聚合方法主要关注性能攻击。例如 FOE、model poisoning、Byzantine update 这类攻击会让全局模型 accuracy 下降。因此 Median、FLTrust、FLGMM、SmartFL、FedDNA、LASA 等方法通常会检查客户端更新是否“方向异常”“范数异常”“偏离多数客户端”。

公平防御方法主要关注公平攻击。例如敏感属性翻转会让模型在不同 sensitive group 上产生不同的 TPR 或正预测率。因此 FairFed、FairGuard 这类方法通常会检查 AEOD、ASPD 或 group fairness gap。

但是 Dual-Facet Attack 的危险在于它同时利用两条攻击路径：

- 性能路径：让模型预测能力下降，即 ACC 降低。
- 公平路径：让模型对不同敏感群体产生不公平，即 AEOD/ASPD 增大。

更进一步，Sp-DFA 里不同恶意客户端可以分工：一部分客户端负责性能攻击，另一部分客户端负责公平攻击。这样，单独看 accuracy 的防御可能漏掉公平攻击，单独看 fairness 的防御可能漏掉性能攻击。

所以 GuardFed-AD2 的核心原则是：

> 一个客户端更新只有同时满足“对模型学习有帮助”和“不会破坏群体公平”，才应该被聚合。

这就是 AD2 里的 "Adaptive Dual-Defense" 含义：自适应地同时防御 utility attack 和 fairness attack。

### 2. GuardFed-AD2 每一轮到底做什么

假设第 \(t\) 轮开始时，server 有当前全局模型 \(\theta_t\)。每个客户端本地训练后上传一个模型更新：

\[
u_i^t = \theta_i^t - \theta_t.
\]

GuardFed-AD2 不会直接平均这些更新，而是先用 server 端的 clean root data 检查每个更新是否可信。

当前实验中，root data 是 5% clean server data：

- 1% 真实 clean server data；
- 4% 通过 Gaussian Copula 生成的 clean synthetic data。

这个 root data 的作用不是大规模训练模型，而是作为 server 端的“参考尺子”。所有客户端更新都用同一把尺子检查，所以比较是公平和一致的。

每一轮包含六步。

### 3. Step 1：构造 clean reference update

Server 先在 root data 上从当前模型 \(\theta_t\) 出发训练一个很小的参考更新：

\[
u_0^t.
\]

这个 \(u_0^t\) 可以理解为：

> 如果一个更新是干净、正常、并且对当前任务有帮助，它大致应该朝哪个方向走。

因为 root data 是 clean 的，并且可以按 sensitive group 做 balance 或 reweight，所以 \(u_0^t\) 同时包含两个信号：

- 正常学习方向；
- 公平敏感的参考方向。

这一步很重要。没有 reference update 时，server 只能看客户端之间谁像谁；但在 non-IID 场景下，正常客户端本来就可能彼此差异很大。GuardFed-AD2 用 clean reference update 给每个客户端一个统一的校准基准，减少 non-IID 造成的误判。

### 4. Step 2：把每个客户端更新临时加到模型上

对每个客户端 \(i\)，server 构造一个 candidate model：

\[
\theta_{i,t}^{cand} = \theta_t + u_i^t.
\]

然后在 root data 上评估这个 candidate model。

也就是说，server 问的是：

> 如果我这一轮只接受这个客户端的更新，它会让模型变好还是变坏？它会让模型更公平还是更不公平？

对每个客户端，server 记录四类信息：

- clean accuracy 或 validation loss；
- AEOD；
- ASPD；
- update direction，也就是 \(u_i^t\) 和 clean reference update \(u_0^t\) 的 cosine similarity；
- update norm，也就是 \(\|u_i^t\|_2\) 是否异常大或异常小。

这些信息分别对应不同攻击迹象。

| 检查项 | 正常客户端表现 | 恶意客户端可能表现 |
|---|---|---|
| Cosine direction | 和 clean reference 方向接近 | 方向相反或明显偏离 |
| Update norm | 大小接近正常更新 | 被 FOE 放大、缩小或扰动 |
| Clean ACC/loss | 不破坏 root data 性能 | ACC 下降或 loss 上升 |
| AEOD/ASPD | 群体差异不明显增加 | 公平差距增加 |

### 5. Step 3：计算 Utility Reliability

Utility reliability 记作 \(R_i^t\)。它回答：

> 这个客户端更新是否有助于模型正常学习？

它由三部分组成。

第一部分是方向一致性：

\[
\operatorname{ReLU}(\cos(u_i^t, u_0^t)).
\]

如果客户端更新和 clean reference update 方向一致，cosine 为正，说明它大致朝着正常学习方向走。如果方向相反，ReLU 后变成 0，说明这个更新非常可疑。

第二部分是范数正常性：

\[
\exp\left(
-\frac{
|\|u_i^t\|_2 - \operatorname{median}(\|u^t\|_2)|
}{
\tau_n \operatorname{MAD}(\|u^t\|_2)+\epsilon
}
\right).
\]

这里不用平均值和标准差，而用 median 和 MAD，是因为它们对恶意客户端更鲁棒。FOE 这类攻击经常会改变更新大小，范数异常的客户端会被压低分数。

第三部分是 clean performance：

\[
\sigma(\Delta Acc_i^t/\tau_a).
\]

如果 candidate model 在 root data 上 accuracy 变好或没有明显变差，这一项较高；如果它让 accuracy 下降，这一项较低。

所以 \(R_i^t\) 可以理解成：

> 方向对、大小正常、accuracy 不坏，utility reliability 才高。

### 6. Step 4：计算 Fairness Reliability

Fairness reliability 记作 \(F_i^t\)。它回答：

> 这个客户端更新是否会破坏敏感群体公平？

Server 在 root data 上计算 candidate model 的 AEOD 和 ASPD。

- AEOD 衡量两个 sensitive groups 的 TPR 差异。
- ASPD 衡量两个 sensitive groups 的正预测率差异。

如果 AEOD/ASPD 越低，说明 fairness 越好。但是这里有一个非常重要的 gate：

\[
Acc_i^t \geq \Gamma_{acc}.
\]

原因是：如果一个模型几乎没有训练好，或者预测几乎全是一个类别，那么 AEOD/ASPD 可能会接近 0。但这种 0 不是公平，而是模型失效。

例如：

- 一个模型把所有样本都预测成 0；
- 它对两个 group 的 TPR 可能都很低，差值接近 0；
- AEOD 看起来很好；
- 但 accuracy 很差，模型没有实际学习能力。

所以 GuardFed-AD2 明确规定：

> 只有当 candidate model 的 ACC 超过最低学习阈值时，AEOD/ASPD 才有资格参与 fairness ranking 或 fairness trust。

这正好回应你强调的点：那些因为性能攻击导致几乎没有训练、从而 AEOD/ASPD 归 0 的结果，不能算最好。

Fairness reliability 可以写成：

\[
F_i^t =
\exp\left(
-
\frac{
z(AEOD_i^t) + z(ASPD_i^t)
}{
\tau_f
}
\right),
\]

其中 \(z(\cdot)\) 是基于 median/MAD 的 robust normalization。AEOD/ASPD 比当前 round 大多数客户端更异常时，\(F_i^t\) 会下降。

### 7. Step 5：自适应决定更重视 utility 还是 fairness

如果固定写成：

\[
Score = \alpha R + (1-\alpha)F,
\]

审稿人容易认为这是手调超参数，是启发式拼分。

GuardFed-AD2 的改法是让 \(\alpha_t\) 每一轮自动计算。直觉是：

- 如果这一轮客户端在 accuracy、方向、范数上的分歧很大，说明性能攻击风险更强，就提高 utility 权重。
- 如果这一轮客户端在 AEOD/ASPD 上的分歧很大，说明公平攻击风险更强，就提高 fairness 权重。

可以写成：

\[
\alpha_t =
\frac{
\text{utility-risk dispersion}
}{
\text{utility-risk dispersion}+\text{fairness-risk dispersion}+\epsilon
}.
\]

更具体地：

\[
\alpha_t =
\frac{
\operatorname{MAD}(\Delta Acc^t) + \operatorname{MAD}(1-\cos(u_i^t,u_0^t))
}{
\operatorname{MAD}(\Delta Acc^t) + \operatorname{MAD}(1-\cos(u_i^t,u_0^t)) +
\operatorname{MAD}(AEOD^t) + \operatorname{MAD}(ASPD^t) + \epsilon
}.
\]

这样 \(\alpha_t\) 不是人工指定，而是由当前 round 的风险结构决定。

### 8. Step 6：用乘法式 trust score 融合两个目标

GuardFed-AD2 最终 trust score 是：

\[
T_i^t=(R_i^t)^{\alpha_t}(F_i^t)^{1-\alpha_t}.
\]

等价地也可以写成 log-linear 形式：

\[
T_i^t =
\exp\left(
\alpha_t \log(R_i^t+\epsilon)
+
(1-\alpha_t)\log(F_i^t+\epsilon)
\right).
\]

为什么不用加法？

因为加法允许“补偿”。例如：

- 一个客户端 \(R_i^t\) 很高，但 \(F_i^t\) 很低，说明它 accuracy 好但严重不公平；
- 如果用加法，它仍可能得到中等甚至较高分数；
- 但用乘法，只要 \(F_i^t\) 很低，最终 \(T_i^t\) 就会明显降低。

反过来也一样：

- 一个客户端 fairness 看起来很好，但 accuracy 很差；
- 它的 \(R_i^t\) 很低；
- 最终 trust 也会被压低。

所以这个 trust score 表达的是：

> 一个更新必须同时通过 utility 和 fairness 两个检查，才能获得高信任。

### 9. Step 7：筛选、裁剪、加权聚合

计算完每个客户端的 \(T_i^t\) 后，server 不直接平均所有客户端，而是用 robust threshold 选择可信集合：

\[
S_t =
\{i: T_i^t \geq \operatorname{median}(T^t) - \kappa \operatorname{MAD}(T^t)\}.
\]

这还是使用 median/MAD，而不是 mean/std，因为 malicious clients 会污染普通均值统计。

然后对保留的更新做 norm clipping：

\[
\operatorname{clip}(u_i^t).
\]

这一步是防止少数客户端即使通过筛选，也通过过大 update norm 影响全局模型。

最后用 trust-based weight 聚合：

\[
w_i^t =
\frac{n_i (T_i^t)^\gamma}{\sum_{j\in S_t} n_j (T_j^t)^\gamma},
\qquad
\theta_{t+1}
=
\theta_t + \sum_{i\in S_t} w_i^t \operatorname{clip}(u_i^t).
\]

这里 \(n_i\) 是客户端样本数，\(T_i^t\) 是信任分数，\(\gamma\) 控制 trust weighting 的强度。

直观上：

- 数据量大且可信的客户端权重大；
- 数据量大但不可信的客户端不会因为样本多而主导聚合；
- 可信但样本很小的客户端也不会被无限放大。

### 10. 为什么它能防 S-DFA 和 Sp-DFA

S-DFA 中，每个恶意客户端同时执行性能攻击和公平攻击。因此这些客户端通常会出现：

- update direction 偏离 clean reference；
- update norm 异常；
- candidate model accuracy 下降；
- AEOD/ASPD 上升。

GuardFed-AD2 会同时从 \(R_i^t\) 和 \(F_i^t\) 两侧压低它们的 trust。

Sp-DFA 中，恶意客户端分工：

- 一部分客户端只做性能攻击；
- 一部分客户端只做公平攻击。

这更难，因为只看单一指标会漏掉一部分攻击者。

GuardFed-AD2 的优势在于：

- 性能攻击客户端会被 \(R_i^t\) 抓住；
- 公平攻击客户端会被 \(F_i^t\) 抓住；
- 最终 \(T_i^t\) 是乘法式融合，所以任一侧失败都会导致低 trust。

因此，Sp-DFA 里的两类恶意客户端虽然攻击方式不同，但都会在对应的 reliability check 中暴露。

### 11. 可以放进论文里的中文表述

GuardFed-AD2 的设计动机是防御联邦学习中的双面攻击风险。现有鲁棒聚合方法通常关注模型性能退化，而公平防御方法主要关注敏感群体之间的预测差异。当攻击者同时破坏模型效用和群体公平，或者在不同恶意客户端之间拆分这两种攻击目标时，单一视角的防御机制容易失效。为此，GuardFed-AD2 将客户端可信度分解为效用可靠性和公平可靠性两个部分，并利用少量 clean server data 对每个上传更新进行统一评估。

在每一轮训练中，server 首先基于 clean root data 构造参考更新，用于表示当前模型的正常学习方向。随后，server 将每个客户端更新临时应用到全局模型上，并在同一 root data 上评估其 clean accuracy、AEOD、ASPD、更新方向和更新范数。效用可靠性用于衡量客户端更新是否与干净参考方向一致、范数是否正常、以及是否保持验证性能；公平可靠性用于衡量该更新是否维持较低的群体差异，并通过最低 accuracy gate 避免将未训练模型产生的虚假低公平差异误判为公平。

不同于固定权重的启发式加权，GuardFed-AD2 根据当前 round 中效用风险和公平风险的 robust dispersion 自动计算自适应权重。当性能攻击迹象更明显时，聚合规则自动更重视效用可靠性；当公平漂移更明显时，则自动更重视公平可靠性。最终 trust score 采用乘法式 log-linear 融合，使得客户端更新必须同时满足效用和公平要求才能获得高信任度。Server 随后基于 median/MAD 阈值筛选可信客户端，对保留更新进行范数裁剪，并按照 trust-normalized weights 聚合。该设计能够同时识别 S-DFA 中的同步双面攻击者和 Sp-DFA 中分工执行性能攻击与公平攻击的恶意客户端。

## GuardFed-AD2: Motivation And Plain-language Logic

### Motivation

The main weakness of previous defenses is that they usually watch only one side of the problem.

- Robust FL methods, such as Median, FLTrust, FLGMM, SmartFL, FedDNA, or LASA, mainly ask whether a client update damages model utility.
- Fairness-aware methods, such as FairFed or FairGuard, mainly ask whether the model becomes unfair across sensitive groups.
- Dual-facet attacks exploit this separation: an attacker can damage accuracy, damage fairness, or split these two goals across different malicious clients.

GuardFed-AD2 is designed around a simple principle:

> A client update should be trusted only when it is both useful for clean learning and safe for group fairness.

This is why the method is not a "best-method selector" and not a manually tuned ensemble. It is a single aggregation rule that assigns each client update a trust score from two evidence sources: utility evidence and fairness evidence.

### Intuitive idea

In every communication round, the server keeps a small clean reference set. In our current setting, this is 5 percent clean server data: 1 percent real clean data plus 4 percent Gaussian-Copula synthetic clean data. The server uses this small reference set as a calibration instrument.

For each uploaded client update, GuardFed-AD2 asks four questions:

1. Does this update move in the same direction as a clean reference update?
2. Is the update size normal, or is it abnormally scaled like a poisoning update?
3. Does the candidate model still perform well on clean validation data?
4. Does the candidate model keep AEOD and ASPD low without collapsing accuracy?

Only updates that pass these questions receive high trust. Updates that look fair only because the model is not learning are not rewarded.

### Easy-to-understand round logic

At round \(t\), the server receives client updates \(u_1^t,\ldots,u_N^t\). GuardFed-AD2 then performs the following steps.

Step 1: Build a clean reference direction.

The server uses the root data to compute a small clean update \(u_0^t\). This update represents the direction a benign and fairness-aware model should move.

Step 2: Test each client update on the same root data.

For each client \(i\), the server temporarily applies its update and evaluates the candidate model \(\theta_t + u_i^t\). This gives three kinds of evidence:

- utility evidence: clean accuracy or validation loss;
- fairness evidence: AEOD and ASPD on the sensitive groups;
- geometric evidence: cosine direction and update norm compared with the clean reference update.

Step 3: Convert the evidence into two reliability scores.

GuardFed-AD2 computes:

- utility reliability \(R_i^t\): high if the update has a clean direction, normal magnitude, and good validation performance;
- fairness reliability \(F_i^t\): high if the update does not increase group disparity and the model still passes the minimum accuracy gate.

Step 4: Adaptively decide which side is more urgent in this round.

Instead of fixing a manual coefficient, GuardFed-AD2 checks the current round statistics. If client updates show strong accuracy/direction abnormality, the method gives more weight to utility defense. If the round shows strong fairness drift, it gives more weight to fairness defense. This produces an adaptive coefficient \(\alpha_t\).

Step 5: Fuse the two scores with a product-style trust score.

GuardFed-AD2 combines \(R_i^t\) and \(F_i^t\) using a log-linear product:

\[
T_i^t=(R_i^t)^{\alpha_t}(F_i^t)^{1-\alpha_t}.
\]

This is important. A simple weighted sum can hide a failure: a client with very bad fairness can still get a high score if its utility score is high. The product-style score does not allow this compensation. If either utility reliability or fairness reliability is very low, the final trust score is also strongly reduced.

Step 6: Filter and aggregate.

The server keeps clients whose trust scores are above a robust median/MAD threshold. The retained updates are clipped to avoid abnormal scaling and then averaged with trust-based weights. Therefore, highly trusted clients contribute more, suspicious clients contribute little or nothing.

### One-sentence explanation

GuardFed-AD2 is an adaptive dual-objective aggregation rule that uses a small clean server reference set to test whether each client update is simultaneously useful for accuracy and safe for fairness, then aggregates only the updates that pass both checks.

### Why this addresses the "heuristic score" criticism

The original concern is that the score may look like a hand-designed mixture of unrelated terms. GuardFed-AD2 makes the design more principled in three ways.

First, the two score components have clear meanings. \(R_i^t\) estimates utility reliability, while \(F_i^t\) estimates fairness reliability. They are not arbitrary indicators; they correspond exactly to the two attack surfaces in dual-facet attacks.

Second, the balance coefficient is adaptive. The method does not manually fix the utility/fairness trade-off. It estimates which type of risk is more variable and more urgent in the current round using robust dispersion statistics.

Third, the final trust score uses multiplicative reliability rather than additive compensation. This matches the defense goal: an update is acceptable only if it is both utility-safe and fairness-safe.

### Paper-ready concise version

GuardFed-AD2 is motivated by the observation that dual-facet attacks create two coupled but distinct risks: utility corruption and fairness degradation. Existing robust aggregation rules mainly detect abnormal model updates, while fairness-aware aggregation mainly mitigates group disparity; neither side alone is sufficient when adversaries coordinate across both objectives. GuardFed-AD2 therefore evaluates every client update through a small clean server reference set and decomposes trust into two reliabilities: utility reliability and fairness reliability. Utility reliability measures whether the update aligns with a clean reference direction, has a normal magnitude, and preserves clean validation performance. Fairness reliability measures whether the candidate model maintains low AEOD and ASPD while still satisfying a minimum accuracy gate, preventing degenerate low-accuracy models from being considered fair.

The two reliabilities are fused by a round-adaptive log-linear product. The adaptive coefficient is computed from robust dispersion statistics of utility risk and fairness risk in the current round, so the method automatically emphasizes the risk that is more pronounced. The multiplicative form prevents one objective from compensating for failure on the other: an update with high accuracy but severe fairness degradation, or low fairness disparity caused by failed learning, receives low trust. The server then filters updates using a median/MAD threshold, clips abnormal update magnitudes, and aggregates the remaining updates with trust-normalized weights. In this way, GuardFed-AD2 provides a single adaptive dual-objective defense against both synchronous and split dual-facet attacks.

The revised method should not be described as "choosing the best method from a candidate pool." GuardFed-AD2 is a single server-side aggregation rule. Its core idea is to defend two attack surfaces at the same time:

1. Utility attack surface: malicious updates that move the global model away from the clean convergence direction.
2. Fairness attack surface: malicious updates that preserve apparent utility but increase group disparity, especially through sensitive-attribute manipulation.

GuardFed-AD2 uses a small clean server set as the root data. In our current reproduction setting this is 5 percent server data, implemented as 1 percent real clean data plus 4 percent Gaussian-Copula synthetic clean data. The root data is not used to retrain the full model; it is used to estimate a clean reference update and to evaluate each uploaded client update under identical utility/fairness tests.

### Round-level workflow

At round \(t\), the server has global model \(\theta_t\). Client \(i\) uploads update

\[
u_i^t = \theta_i^t - \theta_t.
\]

The server computes a clean reference update \(u_0^t\) on the root data. The reference update is fairness-aware because the root data is group-balanced or reweighted by sensitive group and label. For every client update, the server evaluates the provisional model

\[
\theta_{i,t}^{cand} = \theta_t + u_i^t
\]

on the same root data and obtains:

- validation accuracy or validation loss;
- AEOD and ASPD;
- sensitive-group prediction behavior;
- update direction and norm relative to the clean reference update.

### Utility reliability score

The utility score measures whether a client update is aligned with clean learning:

\[
R_i^t =
\operatorname{ReLU}\left(\cos(u_i^t, u_0^t)\right)
\cdot
\exp\left(-\frac{| \|u_i^t\|_2 - \operatorname{med}_j\|u_j^t\|_2 |}{\tau_n \operatorname{MAD}_j\|u_j^t\|_2 + \epsilon}\right)
\cdot
\sigma\left(\frac{\Delta Acc_i^t}{\tau_a}\right).
\]

This score has three roles:

- the cosine term rejects updates pointing against the trusted reference direction;
- the robust norm term suppresses scaled updates such as FOE-style model poisoning;
- the accuracy/loss term prevents updates with good-looking fairness but broken learning from being considered high quality.

### Fairness reliability score

The fairness score measures whether the update harms group parity on the clean root data:

\[
F_i^t =
\exp\left(
-
\frac{
z(AEOD_i^t) + z(ASPD_i^t)
}{\tau_f}
\right),
\]

where \(z(\cdot)\) is a robust round-wise standardization based on median and MAD. Lower AEOD/ASPD gives higher \(F_i^t\), but only if the candidate model passes the minimum learning gate:

\[
Acc_i^t \geq \Gamma_{acc}.
\]

This gate is important for the reviewer concern and for our tables: if a method is nearly untrained, AEOD/ASPD can collapse to zero simply because the model predicts almost one class. GuardFed-AD2 treats such fairness values as invalid rather than as genuinely fair.

### Adaptive dual-objective fusion

Instead of using a fixed heuristic sum, GuardFed-AD2 computes a round-specific objective weight:

\[
\alpha_t =
\frac{
\operatorname{MAD}(\Delta Acc^t) + \operatorname{MAD}(1-\cos(u_i^t,u_0^t))
}{
\operatorname{MAD}(\Delta Acc^t) + \operatorname{MAD}(1-\cos(u_i^t,u_0^t)) +
\operatorname{MAD}(AEOD^t) + \operatorname{MAD}(ASPD^t) + \epsilon
}.
\]

Then it combines utility and fairness through a log-linear reliability product:

\[
T_i^t =
\exp\left(
\alpha_t \log(R_i^t+\epsilon)
+
(1-\alpha_t)\log(F_i^t+\epsilon)
\right).
\]

This formulation is stronger than a simple \(a + (1-a)\) weighted sum. A sum can hide one failed objective behind a high score on the other objective. The log-linear product behaves like an adaptive multi-objective reliability model: if either utility reliability or fairness reliability is close to zero, the final trust score is strongly penalized. At the same time, \(\alpha_t\) is not manually fixed. It moves toward utility defense when the round shows strong performance-poisoning dispersion, and toward fairness defense when the round shows strong fairness-drift dispersion.

### Robust selection and aggregation

GuardFed-AD2 uses a robust threshold:

\[
S_t =
\{i: T_i^t \geq \operatorname{median}(T^t) - \kappa \operatorname{MAD}(T^t)\}.
\]

The retained updates are clipped relative to the root reference norm and aggregated by normalized trust weights:

\[
w_i^t =
\frac{n_i (T_i^t)^\gamma}{\sum_{j\in S_t} n_j (T_j^t)^\gamma},
\qquad
\theta_{t+1}
=
\theta_t + \sum_{i\in S_t} w_i^t \operatorname{clip}(u_i^t).
\]

Optionally, temporal smoothing can be added:

\[
\bar{T}_i^t = \rho \bar{T}_i^{t-1} + (1-\rho)T_i^t,
\]

which reduces round-to-round noise and makes the defense more stable under non-IID partitions.

## Paper-ready Method Positioning

GuardFed-AD2 can be positioned as:

> an adaptive dual-objective robust aggregation rule that jointly estimates utility reliability and fairness reliability from a small clean server reference set. Unlike fixed-score defenses that manually combine heterogeneous indicators, GuardFed-AD2 derives a round-specific trade-off coefficient from robust dispersion statistics of utility and fairness risks. The final client trust is computed by a log-linear reliability product, which prevents a client from compensating severe utility degradation with superficial fairness, or vice versa. This design directly targets dual-facet attacks where adversaries may either synchronously corrupt both objectives or split performance and fairness attacks across different malicious clients.

The key claims we can safely make from the current implementation and tables are:

- It is not a candidate-pool selector.
- It is a single aggregation algorithm with utility and fairness sensors.
- It explicitly prevents degenerate fairness values from being ranked as best when ACC indicates failed learning.
- It is better aligned with the reviewer concern because the fusion weight is data-adaptive rather than manually fixed.
- It gives a more defensible explanation for why GuardFed-AD2 is strong under S-DFA and Sp-DFA: both attack faces must pass the same root-data reliability test.
