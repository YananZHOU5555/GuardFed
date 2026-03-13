#!/usr/bin/env python3
# scripts/test_complete.py - 测试完整脚本（只运行少量实验）

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 修改配置为快速测试
NUM_ROUNDS = 3  # 只用3轮测试
TEST_MODE = True

# 导入完整脚本的所有内容
exec(open('D:\\GitHub\\GuardFed-main\\scripts\\reproduce_table4_complete.py').read())
