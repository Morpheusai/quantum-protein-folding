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
try:
    # Qiskit 2.x版本
    from qiskit_aer import Aer
except ImportError:
    try:
        # Qiskit 1.x版本
        from qiskit.providers.aer import Aer
    except ImportError:
        # 旧版本Qiskit
        from qiskit import Aer
from qiskit_ibm_runtime import QiskitRuntimeService


class QuantumBackendManager:
    """量子后端管理器类"""
    
    @staticmethod
    def setup_backend(backend_name, aws_region=None, shots=100, use_estimator=True):
        """
        设置量子后端
        
        Args:
            backend_name (str): 后端名称，支持以下选项：
                - 'local': 本地模拟器（优先使用AerSimulator）
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
        
        if backend_name.lower() == 'local':
            # 本地Aer模拟器
            try:
                backend = Aer.get_backend('aer_simulator')
                backend.set_options(shots=shots)
                backend_info['backend'] = backend
                backend_info['type'] = 'local'
                backend_info['name'] = 'AerSimulator'
            except Exception as e:
                print(f"警告: 无法加载AerSimulator，使用基础模拟器: {e}")
                # 备用模拟器
                from qiskit import BasicAer
                backend = BasicAer.get_backend('qasm_simulator')
                backend_info['backend'] = backend
                backend_info['type'] = 'local'
                backend_info['name'] = 'BasicAer'
                
        elif backend_name.lower() == 'local_aer':
            # 强制使用Aer模拟器
            backend = Aer.get_backend('aer_simulator')
            backend.set_options(shots=shots)
            backend_info['backend'] = backend
            backend_info['type'] = 'local'
            backend_info['name'] = 'AerSimulator'
            
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
            except ImportError:
                try:
                    from qiskit.primitives import Estimator
                    backend_info['estimator'] = Estimator(backend=backend_info['backend'])
                except Exception as e:
                    print(f"警告: 无法创建Estimator: {e}")
                    backend_info['estimator'] = None
        
        return backend_info
    
    @staticmethod
    def _setup_aws_backend(backend_name, region, shots):
        """设置AWS Braket后端"""
        backend_info = {}
        
        try:
            from braket.aws import AwsDevice
            from braket.circuits import Circuit
            
            # 根据后端名称选择设备
            if backend_name.lower() == 'aws_sv1':
                device_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
            elif backend_name.lower() == 'aws_garnet':
                device_arn = "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-3"
            elif backend_name.lower() == 'aws_ionq':
                device_arn = "arn:aws:braket:us-east-1::device/qpu/ionq/ionQdevice"
            elif backend_name.lower() == 'aws_forte':
                device_arn = "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1"
            else:
                raise ValueError(f"不支持的AWS后端: {backend_name}")
            
            device = AwsDevice(device_arn)
            backend_info['backend'] = device
            backend_info['type'] = 'aws'
            backend_info['name'] = backend_name
            backend_info['shots'] = shots
            
        except ImportError:
            print("警告: AWS Braket SDK未安装，使用本地模拟器替代")
            try:
                from qiskit_aer import Aer
            except ImportError:
                try:
                    from qiskit.providers.aer import Aer
                except ImportError:
                    from qiskit import Aer
            backend = Aer.get_backend('aer_simulator')
            backend.set_options(shots=shots)
            backend_info['backend'] = backend
            backend_info['type'] = 'local'
            backend_info['name'] = 'AerSimulator (AWS替代)'
            
        return backend_info
    
    @staticmethod
    def _setup_ibm_backend(backend_name, shots):
        """设置IBM Quantum后端"""
        backend_info = {}
        
        try:
            # 检查IBM Quantum凭据
            if not os.environ.get('QISKIT_IBM_TOKEN'):
                print("警告: IBM Quantum凭据未设置，使用本地模拟器替代")
                try:
                    from qiskit_aer import Aer
                except ImportError:
                    try:
                        from qiskit.providers.aer import Aer
                    except ImportError:
                        from qiskit import Aer
                backend = Aer.get_backend('aer_simulator')
                backend.set_options(shots=shots)
                backend_info['backend'] = backend
                backend_info['type'] = 'local'
                backend_info['name'] = 'AerSimulator (IBM替代)'
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
            try:
                from qiskit_aer import Aer
            except ImportError:
                try:
                    from qiskit.providers.aer import Aer
                except ImportError:
                    from qiskit import Aer
            backend = Aer.get_backend('aer_simulator')
            backend.set_options(shots=shots)
            backend_info['backend'] = backend
            backend_info['type'] = 'local'
            backend_info['name'] = 'AerSimulator'
            
        return backend_info