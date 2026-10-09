# C 三场景实际表：待根独立采用

IID Benign、F Flip、FedSA三完整场景，共Full30与minus_C30条已接受终轮记录，raw/native/shared三视图，10/9/6共享seed面板和配对差全部展示。S-DFA6条已接受C结果另存并明确排除均值，Full仅引用。

原两场景40条JSON记录片段、324统计标量及162展示单元保持一致。独立math.fsum/sampleSD复算486标量，最大差1.4210854715202004e-14；243展示单元、540receipt指标、1440计数字段检查通过。三个新增来源归档5+3+8全部按原strict/offserver/receipt身份连接；没有推理、校准重拟合或训练。

FedSA十seed新增结果（minus_C−Full）：原生/共享校准ΔACC=+0.279358pp、ΔAEOD=−0.000235940、ΔASPD=+0.000988233；raw分别+0.235063pp、−0.001320380、+0.001415191。去掉C后准确率/AEOD均值略好、ASPD略差，保留这一负面机制证据，不作C不可或缺或显著性主张。native/shared在本表60条记录的全部指标及计数相同，不能推广为其他cohort或设备等价。

ACC越高越好，AEOD/ASPD越低越好；AEOD为绝对TPR差，非完整equalized odds。native保留各自后处理，shared使用原共同root-only规则。Full重放CPU2/CUDA28，C为CPU30，训练均记录cu128，保留driver/runtime逐条来源与此前选择、validation/test历史暴露；9/6面板不是未触碰确认。主终点待作者，其他七个C场景未完成。此处结果仍待根独立表审阅，不修改canonical或论文。
