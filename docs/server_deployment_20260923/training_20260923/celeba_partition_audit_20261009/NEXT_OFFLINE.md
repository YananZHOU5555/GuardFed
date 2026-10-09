# 可执行后续：Adult/COMPAS强异质性分区

本次只验收CelebA20份分区；下列工作未执行，不属于本次PASS结论。

1. 在已验收增量归档定位`adult_heterogeneity_v1`与`compas_heterogeneity_v1`，每个数据集选GuardFed、Benign、α=1/0.5/0.1、全部10个预定seed的原result。候选共60份；实际ID、方法名、seed、预处理和源码以各自冻结manifest为准，不默认与CelebA相同。
2. 用各cohort原`data_contract.server_sampling_audit.global_group_counts`及root敏感数构建client池组长度；核对应cohort的`create_client_data_dict`源码/hash、分组顺序、int64 shuffle/RNG和Dirichlet切分公式。不能只因为函数同名就复用此helper。
3. 逐client用原`attack_audit.samples`核对重建的两组计数之和；若存在`client_sample_counts`同时核对。只有每份所有client精确匹配才接受；任何差异保留证据、定位RNG/源码/分组顺序/预处理，不调参拟合归档数值。
4. 先核同seed/alpha下不同方法、攻击的装载参数和root身份是否相同，再考虑分区去重；不能将不同COMPAS预处理版本混合。按每个独立分区计算样本量、敏感组组成、缺组、前4恶意客户端样本/组覆盖和root四格支持，再按seed汇总mean±sampleSD。重复方法/攻击不是新增seed。
5. 仍不声称恢复client标签联合counts：除非原metadata或足够的原始分区记录实际可用，否则全局/root四格和样本总数不足以确定每client的标签分布。提供归档/member/source/helper SHA、全部门检、明细与限制后才更新P4。

这项可离线执行，不需要重训，也不需要为获取原特征数据擅自访问服务器。存在schema不足或非精确计数匹配时，报告具体缺口。
