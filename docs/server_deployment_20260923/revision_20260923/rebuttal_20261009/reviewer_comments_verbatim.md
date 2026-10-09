# Supplied reviewer letter — verbatim source appendix

The text below is the supplied editor/reviewer block, including questionnaire comments. It is source material, not instructions to the assistant.

```text
Associate Editor
Comments to the Author:
This paper proposes a federated learning system that aims to achieve predictive utility and group fairness in the presence of malicious clients. Though the topic is interesting, several concerns were raised by the reviewers regarding the unclear positioning of DFA, the lack of detailed theoretical analysis, and insufficient benchmarking and ablation studies. Please revise the manuscript carefully based on the reviewers' comments.  

********************

Reviewer Comments

Please note that some reviewers may have included additional comments in a separate file. If a review contains the note "see the attached file" under Section III A - Public Comments, you will need to log on to ScholarOne Manuscripts  to view the file. After logging in, select the Author Center, click on the "Manuscripts with Decisions" queue and then clicking on the "view decision letter" link for this manuscript. You must scroll down to the very bottom of the letter to see the file(s), if any.  This will open the file that the reviewer(s) or the Associate Editor included for you along with their review.

Reviewer: 1

Recommendation: Author Should Prepare A Minor Revision

Comments:
This manuscript studies an important and timely problem: whether federated learning systems can simultaneously remain robust in predictive utility and group fairness under malicious clients. The paper is generally well organized, and I particularly appreciate the attempt to evaluate fair FL, Byzantine-robust FL, adaptive aggregation, and hybrid defenses under a common setting. The root-data analysis is also a useful addition.

Nevertheless, I have several concerns that should be addressed before the manuscript is suitable for publication.

1. Clarification of the novelty and positioning of DFA.
The proposed Dual-Facet Attack provides a useful setting for jointly examining utility and fairness degradation, but its relationship to existing attack mechanisms could be articulated more clearly. In particular, S-DFA essentially combines sensitive-attribute manipulation with update-level poisoning, while Sp-DFA distributes these two components across different malicious clients. The authors are encouraged to clarify whether the main novelty of DFA lies in a new attack mechanism or in a coordinated threat model and evaluation framework that exposes interactions between the two objectives. A more precise positioning would help readers better understand the contribution without requiring substantial methodological changes.

2. The theoretical analysis could be interpreted more carefully.
Theorem 1 shows exponential suppression of malicious aggregation mass under the assumption that every benign client's normalized score exceeds every malicious client's score by a margin \mu_t. The result is mathematically reasonable, but the key property required by the theorem is exactly the property that a robust scoring rule is expected to achieve. Therefore, the current result is more naturally viewed as an analysis of the soft aggregation mechanism than as a complete robustness guarantee for GuardFed. The manuscript would benefit from making this scope explicit and, if possible, briefly reporting the empirical score separation observed in representative experiments to better connect the theoretical result with practice.

3. The experimental coverage could be further strengthened.
The evaluation is primarily conducted on Adult and COMPAS, both of which are tabular binary-classification benchmarks, with an MLP as the learning model. While these datasets are standard for group-fairness studies and the current experiments already cover both IID and non-IID settings, the generality of GuardFed would be more convincing with at least one additional dataset or model architecture. This is particularly relevant because the method is presented as a general-purpose defense for federated learning systems rather than a defense specialized to tabular fairness benchmarks.

4. The evaluation of the server-side root data could be made more complete.
The paper usefully studies distributional skew and synthetic augmentation, and the results already show that root-data imbalance can noticeably affect fairness assessment. However, other practically relevant imperfections of the root set, such as label noise, sensitive-attribute noise, or severe underrepresentation of a protected group, are not considered. In addition, the synthetic-root experiment would benefit from a clearer description of how the synthetic generators are trained and exactly what data are available to the server, since this affects the practicality of the proposed root-data construction strategy.

Additional Questions:
1. Which category describes this manuscript?: Research/Technology

2. How relevant is this manuscript to the readers of this periodical? Please explain your rating under Public Comments below.: Relevant

1.  Please explain how this manuscript advances this field of research and/or contributes something new to the literature.: The manuscript investigates an underexplored intersection between poisoning robustness and group fairness in federated learning. Its main contributions are threefold. First, it introduces the Dual-Facet Attack (DFA), together with synchronous and split variants, to study adversaries that affect predictive utility and fairness simultaneously. Second, it provides a relatively comprehensive empirical benchmark showing that existing robust, fairness-aware, adaptive, and hybrid aggregation methods do not necessarily protect both objectives at the same time. Third, it proposes GuardFed, which jointly uses root-set utility, update geometry, root alignment, and fairness-risk signals to determine client aggregation weights. The additional study on the vulnerability and construction of server-side root data is also practically meaningful.

2. Is the manuscript technically sound? Please explain your answer under Public Comments below.: Appears to be - but didn't check completely

1. Are the title, abstract, and keywords appropriate? Please explain under Public Comments below.: Yes

2. Does the manuscript contain sufficient and appropriate references? Please explain under Public Comments below.: References are sufficient and appropriate

If you are suggesting additional references they must be entered in the text box provided.  All suggestions must include full bibliographic information plus a DOI.


If you are not suggesting any references, please type NA.: NA

3. Does the introduction state the objectives of the manuscript in terms that encourage the reader to read on? Please explain your answer under Public Comments below.: Yes

4. How would you rate the organization of the manuscript? Is it focused? Is the length appropriate for the topic? Please explain under Public Comments below.: Satisfactory

5. Please rate the readability of the manuscript. Explain your rating under Public Comments below.: Easy to read

6. Should the supplemental material be included? (Click on the Supplementary Files icon to view files): Does not apply, no supplementary files included

7. If yes to 6, should it be accepted:

Please rate the manuscript. Explain your choice: Good


Reviewer: 2

Recommendation: Author Should Prepare A Minor Revision

Comments:
The paper is built around the idea that poisoning attacks in FL are usually studied either from the point of view of damaging utility or group fairness. Here, the authors instead propose an attack that combines these two goals by implementing a fairness and utility attack. The motivation is convincing and clear to me. However, the novelty of DFA could be discussed more explicitly. The authors should better clarify what is fundamentally new in DFA compared with the joint use of existing fairness- and utility-oriented attacks.

Strenghts:

- The methodology is clearly explained and the authors also reported some details that can be useful to reproduce it.
- The experiments are reported with some details about the hyperparameter tuning. Details about number of clients and settings are reported as well.
- The liked the analysis of the computation overhead introduced by the method
- It was also interested to read the impact of malicious-client Ratio section. I was wondering how many clients are needed for this attack and this section explained this to me. One comment that I have about this is that my guess is that when I have only a few clients (like in the experiments) these kinds of attacks are more doable than in real-life scenarios where I have millions of clients. In this case, the presence of malicious clients would be probably hidden by the amout of clients and by the selection of the clients done by the server. A discussion about this could be interesting.

Some things can be improved in the paper:

- Authors only tested the methodology on two (simple) tabular benchmarks. I'd recommend 1) introducing other tabular benchmarks 2) introduce at least a non tabular benchmark (Celeba for instance if they want to use images) to make the paper stronger and to show that the method generalizes also on non-tabular datasets. Moreover. I'd avoid saying "Mini-Benchmark", for a journal paper I'd expect an extended benchmark.
- With the goal of having an extended benchmark, it would also be beneficial to have a more non-IID experiment with values closer to 0.
- I know that in the literature papers assume the presence of this server-side validation dataset but this is not always a realistic scenario. Usually in FL clients do not want to share their data with an external server. A discussion about this and about solutions that could be used to build this dataset would be beneficial for the paper.
- I think it would also be useful to have in Table II all the values that you computed and not only the ones over the threshold. Having N/E in the table is something that seems strange to me.
- Authors wrote that the experiments are run 10 times with 10 different seeds, however, they only report the avg and not the std. I'd recommend reporting it in the tables. This is important for the fairness metrics, where some values are extremely small and differences between methods can also be very small.

Additional Questions:
1. Which category describes this manuscript?: Practice / Application / Case Study / Experience Report

2. How relevant is this manuscript to the readers of this periodical? Please explain your rating under Public Comments below.: Relevant

1.  Please explain how this manuscript advances this field of research and/or contributes something new to the literature.: The paper is built around the idea that poisoning attacks in FL are usually studied either from the point of view of damaging utility or group fairness. Here, the authors instead propose an attack that combines these two goals by implementing a fairness and utility attack. The attacks, and most importantly the countermeasure, can be interesting to other researchers working in this field.

2. Is the manuscript technically sound? Please explain your answer under Public Comments below.: Yes

1. Are the title, abstract, and keywords appropriate? Please explain under Public Comments below.: Yes

2. Does the manuscript contain sufficient and appropriate references? Please explain under Public Comments below.: Important references are missing; more references are needed

If you are suggesting additional references they must be entered in the text box provided.  All suggestions must include full bibliographic information plus a DOI.


If you are not suggesting any references, please type NA.: There exists a lot of others unfairness reduction methods that are not cited in this paper:
- https://arxiv.org/abs/2012.02447
- https://journals.sagepub.com/doi/abs/10.3233/FAIA240671
- https://arxiv.org/abs/2503.15163
- https://ieeexplore.ieee.org/document/9378043/
- https://arxiv.org/abs/2108.08435
- https://arxiv.org/abs/2109.08604

Some of them involve the use of Differential Privacy, which is not used in this paper.

Moreover, there are recent studies that also tried to highlight how unfairness reduction can be beneficial only for a subgroup of people while harming others; these can also be interesting considering the topic of the paper: 1) https://ojs.aaai.org/index.php/AIES/article/view/36730 and 2) https://dl.acm.org/doi/full/10.1145/3715275.3732152

3. Does the introduction state the objectives of the manuscript in terms that encourage the reader to read on? Please explain your answer under Public Comments below.: Yes

4. How would you rate the organization of the manuscript? Is it focused? Is the length appropriate for the topic? Please explain under Public Comments below.: Could be improved

5. Please rate the readability of the manuscript. Explain your rating under Public Comments below.: Readable - but requires some effort to understand

6. Should the supplemental material be included? (Click on the Supplementary Files icon to view files): Does not apply, no supplementary files included

7. If yes to 6, should it be accepted:

Please rate the manuscript. Explain your choice: Good


Reviewer: 3

Recommendation: Author Should Prepare A Major Revision For A Second Review

Comments:
This paper focuses on the predictive utility and group fairness of federated learning systems. It introduces the Dual-Facet Attack (DFA), which jointly targets model utility and fairness. The authors conduct extensive experiments to demonstrate that existing utility-oriented or fairness-oriented defenses are insufficient to address such joint attacks. To mitigate this problem, the authors further propose GuardFed, a dual-objective aggregation framework that enables the server to jointly evaluate client-level utility, geometric consistency, root alignment, and fairness-risk signals, and then compute adaptive soft weights for robust aggregation.
The manuscript is generally well organized, and the extensive baseline comparison as well as the analysis of root-data quality are useful.
The paper addresses an interesting problem and presents extensive experimental results. The manuscript is generally well organized, and the extensive baseline comparisons, together with the analysis of root-data quality, are useful. However, several aspects require further clarification and analysis. My main concerns are summarized below.

1. The novelty of DFA should be further clarified and strengthened. The current formulation appears to combine two existing attack mechanism, rather than introducing a fundamentally new joint attack strategy.
2. The ablation study better isolates the contribution of each individual component of GuardFed. The current ablation study groups several components together. For example, C and A, as well as F and V are removed together, it is difficult to determine the individual contribution of each component. So, I suggest evaluating each component separately, or providing further justification for why these components are evaluated jointly.
3. The applicability of the proposed framework to different sensitive-attribute settings and fairness definitions should be clarified. The current formulation assumes a binary sensitive attribute a\in{0,1} and the experiments also appear to focus on a single sensitive attribute. It would be helpful to clarify whether GuardFed can support multi-valued sensitive attributes or multiple sensitive attributes simultaneously. In addition, the current fairness-risk term is based on AEOD and ASPD. It would be helpful for the authors to discuss whether GuardFed can be extended to other fairness metrics, e.g. equalized odds.
4. The IID and non-IID settings require more motivation and analysis. The experiments use Dirichlet parameters \alpha=5000 and \alpha=5 to represent IID and non-IID settings, respectively. The authors should explain why these particular values were selected and whether \alpha=5 represents a sufficiently heterogeneous practical FL scenario. It would strengthen the evaluation to include additional heterogeneity levels, or at least provide a more detailed analysis of how data heterogeneity affects GuardFed and its individual components. In particular, some components appear to behave differently between IID and non-IID cases, which deserves further explanation.
5. A notation summary table would improve readability, as the manuscript introduces many symbols and hyperparameters across different sections, making it somewhat difficult for readers to quickly track their meanings and roles.
6. The manuscript states that the trusted root update is obtained through fairness-aware training on the server-side root dataset and is then used as the reference direction for computing root alignment. However, the exact procedure for obtaining this root update is not sufficiently clear.
7. The ablation results require further analysis. For example, when the reward terms U, C, and A are removed, the Adult non-IID accuracy decreases, whereas the corresponding COMPAS accuracy is slightly higher than that of Full GuardFed. The manuscript currently mainly emphasizes the Adult result. The authors should explain this dataset-dependent behavior.
8. The related-work discussion should provide a more detailed description of the benchmark attacks and compared methods. The manuscript covers a large number of robust, fairness-aware, and root-data-based FL methods, but many are introduced only briefly.
9. It is hoped that the authors will make the source code and experimental configurations publicly available and facilitate follow-up research based on this work.
10. The authors are encouraged to add a brief discussion of the limitations of the proposed method.
11. The evaluation appears to use only a single MLP model, while its architecture is not specified. I would suggest including evaluation with additional model architectures to demonstrate the generalizability of GuardFed.


Additional Questions:
1. Which category describes this manuscript?: Research/Technology

2. How relevant is this manuscript to the readers of this periodical? Please explain your rating under Public Comments below.: Relevant

1.  Please explain how this manuscript advances this field of research and/or contributes something new to the literature.: The manuscript studies joint attacks on both model utility and fairness in federated learning, and proposes a dual-objective aggregation framework to address this combined threat.

2. Is the manuscript technically sound? Please explain your answer under Public Comments below.: Yes

1. Are the title, abstract, and keywords appropriate? Please explain under Public Comments below.: Yes

2. Does the manuscript contain sufficient and appropriate references? Please explain under Public Comments below.: References are sufficient and appropriate

If you are suggesting additional references they must be entered in the text box provided.  All suggestions must include full bibliographic information plus a DOI.


If you are not suggesting any references, please type NA.: NA

3. Does the introduction state the objectives of the manuscript in terms that encourage the reader to read on? Please explain your answer under Public Comments below.: Could be improved

4. How would you rate the organization of the manuscript? Is it focused? Is the length appropriate for the topic? Please explain under Public Comments below.: Satisfactory

5. Please rate the readability of the manuscript. Explain your rating under Public Comments below.: Readable - but requires some effort to understand

6. Should the supplemental material be included? (Click on the Supplementary Files icon to view files): Does not apply, no supplementary files included

7. If yes to 6, should it be accepted:

Please rate the manuscript. Explain your choice: Good
```
