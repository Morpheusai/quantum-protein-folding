#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AWS Braket设备检查脚本

功能：
- 列出所有可用的AWS Braket量子设备
- 显示设备类型（模拟器/QPU）
- 显示设备状态和基本信息
"""

from qiskit_braket_provider import BraketProvider

def check_aws_devices():
    """检查并列出所有可用的AWS Braket设备"""
    print("=" * 80)
    print("AWS Braket 可用量子设备列表")
    print("=" * 80)
    
    try:
        provider = BraketProvider()
        
        # 获取所有设备
        all_backends = provider.backends()
        print(f"\n总共有 {len(all_backends)} 个设备可用\n")
        
        # 分类显示设备
        print("【模拟器设备】")
        print("-" * 80)
        simulators = provider.backends(types=["SIMULATOR"])
        if simulators:
            for backend in simulators:
                print(f"  设备名称: {backend.name}")
                print(f"  设备类型: 模拟器")
                print(f"  量子比特数: {backend.num_qubits}")
                print(f"  设备描述: {backend.description}")
                print("-" * 80)
        else:
            print("  无可用模拟器设备")
        
        print("\n【量子处理器设备 (QPU)】")
        print("-" * 80)
        qpus = provider.backends(types=["QPU"])
        if qpus:
            for backend in qpus:
                print(f"  设备名称: {backend.name}")
                print(f"  设备类型: 量子处理器 (QPU)")
                print(f"  量子比特数: {backend.num_qubits}")
                print(f"  设备描述: {backend.description}")
                print("-" * 80)
        else:
            print("  无可用量子处理器设备")
        
        print("\n【所有设备详细信息】")
        print("=" * 80)
        for i, backend in enumerate(all_backends, 1):
            print(f"\n{i}. {backend.name}")
            print(f"   类型: {'模拟器' if 'SIMULATOR' in str(backend) else '量子处理器'}")
            print(f"   量子比特数: {backend.num_qubits}")
            print(f"   支持的门操作: {len(backend.operations)}")
            print(f"   最大电路数: {backend.max_circuits}")
            
            # 尝试获取更多信息
            try:
                if hasattr(backend, 'queue_depth'):
                    print(f"   队列深度: {backend.queue_depth()}")
            except:
                pass
        
        print("\n" + "=" * 80)
        print("检查完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"错误: 无法连接到AWS Braket服务")
        print(f"详细信息: {e}")
        print("\n请确保：")
        print("1. 已安装 qiskit-braket-provider: pip install qiskit-braket-provider")
        print("2. 已配置AWS凭据（环境变量或AWS配置文件）")
        print("3. 网络连接正常")

if __name__ == "__main__":
    check_aws_devices()