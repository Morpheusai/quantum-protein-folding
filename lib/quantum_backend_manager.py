#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量子后端管理器模块

功能：
- 管理不同量子后端的配置和初始化
- 支持本地模拟器和云量子设备
- 提供统一的后端接口

主要类：
- QuantumBackendManager: 量子后端管理器类，提供静态方法进行后端配置

主要方法：
- setup_backend(): 设置量子后端（本地/AWS/IBM）
- get_available_backends(): 获取可用的后端列表

支持的后端：
本地后端：
- 'local': 本地模拟器（优先使用AerSimulator）
- 'local_aer': 强制使用AerSimulator

AWS后端：
- 'aws_sv1': AWS SV1模拟器
- 'aws_garnet': AWS Garnet量子芯片
- 'aws_ionq': AWS IonQ量子设备
- 'aws_forte': AWS IonQ Forte量子设备

IBM后端：
- 'ibm': IBM量子设备
- 'ibm_simulator': IBM模拟器

使用示例：
    >>> from lib import QuantumBackendManager
    >>> backend_info = QuantumBackendManager.setup_backend('local', shots=100)
    >>> backend = backend_info['backend']
    >>> backend_info = QuantumBackendManager.setup_backend('aws_sv1', aws_region='us-east-1')

依赖：
- qiskit_aer: Qiskit Aer模拟器
- qiskit_ibm_runtime: IBM Quantum运行时
- braket.aws: AWS Braket SDK（用于AWS后端）
"""

import os
import sys

# 尝试导入AerSimulator（可选）
try:
    from qiskit_aer import AerSimulator
    has_aer = True
except ImportError:
    has_aer = False

# 导入BasicSimulator（必选，Qiskit 2.x中自带）
from qiskit.providers.basic_provider import BasicSimulator

# 检查是否安装了qiskit_ibm_runtime
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    has_ibm = True
except ImportError:
    has_ibm = False


class QuantumBackendManager:
    """量子后端管理器类"""
    
    @staticmethod
    def setup_backend(backend_name, aws_region=None, shots=100, use_estimator=True):
        """
        设置量子后端
        
        Args:
            backend_name (str): 后端名称，支持以下选项：
                - 'local': 本地模拟器（使用AerSimulator）
                - 'local_aer': 强制使用AerSimulator
                - 'aws_sv1': AWS SV1模拟器
                - 'aws_garnet': AWS Garnet量子芯片
                - 'aws_ionq': AWS IonQ量子设备
                - 'aws_forte': AWS IonQ Forte量子设备
                - 'ibm': IBM量子设备
                - 'ibm_simulator': IBM模拟器
            aws_region (str, optional): AWS区域设置（仅用于AWS后端）
            shots (int, optional): 量子采样次数，默认为100
            use_estimator (bool, optional): 是否使用Estimator模式，默认为True
            
        Returns:
            dict: 包含后端和配置信息的字典，包含以下键：
                - 'backend': 量子后端对象
                - 'type': 后端类型（'local', 'aws', 'ibm'）
                - 'name': 后端名称
                - 'estimator': Estimator实例（如果use_estimator=True）
                
        Raises:
            ValueError: 如果不支持指定的后端名称
            
        Example:
            >>> backend_info = QuantumBackendManager.setup_backend('local', shots=1000)
            >>> print(backend_info['name'])
            AerSimulator
        """
        backend_info = {}
        
        if backend_name.lower() == 'local' or backend_name.lower() == 'local_aer':
            # 本地Aer模拟器
            try:
                if has_aer:
                    backend = AerSimulator()
                    backend.set_options(shots=shots)
                    backend_info['backend'] = backend
                    backend_info['type'] = 'local'
                    backend_info['name'] = 'AerSimulator'
                else:
                    # 没有安装qiskit_aer，使用BasicSimulator
                    backend = BasicSimulator()
                    backend_info['backend'] = backend
                    backend_info['type'] = 'local'
                    backend_info['name'] = 'BasicSimulator'
            except Exception as e:
                print(f"警告: 无法加载AerSimulator，使用基础模拟器: {e}")
                # 备用模拟器
                backend = BasicSimulator()
                backend_info['backend'] = backend
                backend_info['type'] = 'local'
                backend_info['name'] = 'BasicSimulator'
                
        elif backend_name.lower().startswith('aws'):
            # AWS Braket后端
            backend_info = QuantumBackendManager._setup_aws_backend(backend_name, aws_region, shots)
            
        elif backend_name.lower().startswith('ibm'):
            # IBM Quantum后端
            backend_info = QuantumBackendManager._setup_ibm_backend(backend_name, shots)
            
        else:
            raise ValueError(f"不支持的量子后端: {backend_name}")
        
        # 如果需要estimator，创建estimator实例
        if use_estimator and 'backend' in backend_info:
            try:
                from qiskit.primitives import StatevectorEstimator
                backend_info['estimator'] = StatevectorEstimator()
            except Exception as e:
                print(f"警告: 无法创建Estimator: {e}")
                backend_info['estimator'] = None
        
        return backend_info
    
    @staticmethod
    def _setup_aws_backend(backend_name, region, shots):
        """设置AWS Braket后端"""
        backend_info = {}
        
        try:
            from qiskit_braket_provider import BraketProvider
            
            # 设置AWS区域
            if region:
                os.environ['AWS_DEFAULT_REGION'] = region
            
            provider = BraketProvider()
            
            # 根据后端名称选择设备
            if backend_name.lower() == 'aws_sv1':
                backend = provider.get_backend('SV1')
                print(f"✓ AWS SV1 已连接")
            elif backend_name.lower() == 'aws_garnet':
                backend = provider.get_backend('Garnet')
                print(f"✓ AWS Garnet 已连接")
            elif backend_name.lower() == 'aws_ionq':
                backend = provider.get_backend('IonQ Device')
                print(f"✓ AWS IonQ 已连接")
            elif backend_name.lower() == 'aws_forte':
                backend = provider.get_backend('Forte 1')
                print(f"✓ AWS IonQ Forte 已连接")
            else:
                raise ValueError(f"不支持的AWS后端: {backend_name}")
            
            backend_info['backend'] = backend
            backend_info['type'] = 'aws'
            backend_info['name'] = backend_name
            backend_info['shots'] = shots
            
        except ImportError as e:
            print(f"警告: Qiskit Braket Provider未安装，使用本地模拟器替代: {e}")
            # 使用本地模拟器作为替代
            if has_aer:
                backend = AerSimulator()
                backend.set_options(shots=shots)
                backend_info['backend'] = backend
                backend_info['type'] = 'local'
                backend_info['name'] = 'AerSimulator (AWS替代)'
            else:
                backend = BasicSimulator()
                backend_info['backend'] = backend
                backend_info['type'] = 'local'
                backend_info['name'] = 'BasicSimulator (AWS替代)'
            
        except Exception as e:
            print(f"警告: AWS后端设置失败，使用本地模拟器替代: {e}")
            # 使用本地模拟器作为替代
            if has_aer:
                backend = AerSimulator()
                backend.set_options(shots=shots)
                backend_info['backend'] = backend
                backend_info['type'] = 'local'
                backend_info['name'] = 'AerSimulator (AWS替代)'
            else:
                backend = BasicSimulator()
                backend_info['backend'] = backend
                backend_info['type'] = 'local'
                backend_info['name'] = 'BasicSimulator (AWS替代)'
            
        return backend_info
    
    @staticmethod
    def _setup_ibm_backend(backend_name, shots):
        """设置IBM Quantum后端"""
        backend_info = {}
        
        try:
            if not has_ibm:
                print("警告: IBM Quantum Runtime未安装，使用本地模拟器替代")
                # 使用本地模拟器作为替代
                if has_aer:
                    backend = AerSimulator()
                    backend.set_options(shots=shots)
                    backend_info['backend'] = backend
                    backend_info['type'] = 'local'
                    backend_info['name'] = 'AerSimulator (IBM替代)'
                else:
                    backend = BasicSimulator()
                    backend_info['backend'] = backend
                    backend_info['type'] = 'local'
                    backend_info['name'] = 'BasicSimulator (IBM替代)'
                return backend_info
            
            # 检查IBM Quantum凭据
            if not os.environ.get('QISKIT_IBM_TOKEN'):
                print("警告: IBM Quantum凭据未设置，使用本地模拟器替代")
                # 使用本地模拟器作为替代
                if has_aer:
                    backend = AerSimulator()
                    backend.set_options(shots=shots)
                    backend_info['backend'] = backend
                    backend_info['type'] = 'local'
                    backend_info['name'] = 'AerSimulator (IBM替代)'
                else:
                    backend = BasicSimulator()
                    backend_info['backend'] = backend
                    backend_info['type'] = 'local'
                    backend_info['name'] = 'BasicSimulator (IBM替代)'
                return backend_info
            
            service = QiskitRuntimeService()
            
            if backend_name.lower() == 'ibm_simulator':
                backend = service.least_busy(simulator=True, operational=True)
            else:
                backend = service.least_busy(simulator=False, operational=True)
            
            backend_info['backend'] = backend
            backend_info['type'] = 'ibm'
            backend_info['name'] = backend.name
            backend_info['shots'] = shots
            
        except Exception as e:
            print(f"警告: IBM Quantum后端设置失败，使用本地模拟器替代: {e}")
            # 使用本地模拟器作为替代
            if has_aer:
                backend = AerSimulator()
                backend.set_options(shots=shots)
                backend_info['backend'] = backend
                backend_info['type'] = 'local'
                backend_info['name'] = 'AerSimulator'
            else:
                backend = BasicSimulator()
                backend_info['backend'] = backend
                backend_info['type'] = 'local'
                backend_info['name'] = 'BasicSimulator'
            
        return backend_info