#!/usr/bin/env python3
"""GuardFed 项目设置测试脚本"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """测试导入"""
    print("测试导入...")
    try:
        from src.data_loader import DatasetLoader
        print("✅ DatasetLoader 导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_structure():
    """测试结构"""
    print("\n测试项目结构...")
    dirs = ['data', 'src', 'scripts', 'src/algorithms', 'src/models']
    success = True
    for d in dirs:
        if os.path.isdir(d):
            print(f"✅ {d}/ 存在")
        else:
            print(f"❌ {d}/ 不存在")
            success = False
    return success

def main():
    print("="*60)
    print("GuardFed 项目设置测试")
    print("="*60)
    
    results = []
    results.append(test_structure())
    results.append(test_imports())
    
    print("\n" + "="*60)
    if all(results):
        print("✅ 所有测试通过！")
    else:
        print("❌ 部分测试失败")
    print("="*60)
    
    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main())
