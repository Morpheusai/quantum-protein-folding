#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量子后端管理器模块

管理不同量子后端的配置和初始化，支持本地模拟器和云量子设备。

主要类：
- QuantumBackendManager: 量子后端管理器类

主要方法：
- setup_backend(): 设置量子后端（local/aws/ibm）

使用示例：
    >>> from lib import QuantumBackendManager
    >>> backend_info = QuantumBackendManager.setup_backend('local', shots=100)
    >>> backend = backend_info['backend']

依赖：
- qiskit_aer: Qiskit Aer模拟器（可选）
- qiskit_ibm_runtime: IBM Quantum运行时（可选）
- qiskit_braket_provider: AWS Braket Provider（可选）
"""

import os
import sys
import traceback

try:
    from qiskit_aer import AerSimulator
    has_aer = True
except ImportError:
    has_aer = False

from qiskit.providers.basic_provider import BasicSimulator

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    has_ibm = True
except ImportError:
    has_ibm = False


class QuantumBackendManager:
    """量子后端管理器类"""
    
    # AWS设备配置字典
    AWS_BACKENDS = {
        'aws_sv1': {'name': 'SV1', 'display': 'AWS SV1'},
        'aws_garnet': {'name': 'Garnet', 'display': 'AWS Garnet'},
        'aws_ionq': {'name': 'IonQ Device', 'display': 'AWS IonQ'},
        'aws_forte': {'name': 'Forte 1', 'display': 'AWS IonQ Forte'},
        'aws_aria': {'name': 'Aria 1', 'display': 'AWS IonQ Aria 1'},
        'aws_tn1': {'name': 'TN1', 'display': 'AWS TN1'},
        'aws_dm1': {'name': 'dm1', 'display': 'AWS dm1'},
        'aws_ankaa': {'name': 'Ankaa-3', 'display': 'AWS Rigetti Ankaa-3'},
        'aws_emerald': {'name': 'Emerald', 'display': 'AWS IQM Emerald'},
        'aws_ibex': {'name': 'IBEX Q1', 'display': 'AWS AQT IBEX Q1'},
    }
    
    # 可用的AWS设备列表（用于错误提示）
    AVAILABLE_AWS_BACKENDS = [
        'aws_sv1 (模拟器，34量子比特)',
        'aws_tn1 (模拟器，50量子比特)',
        'aws_dm1 (模拟器，17量子比特)',
        'aws_forte (IonQ Forte 1，36量子比特)',
        'aws_aria (IonQ Aria 1，25量子比特)',
        'aws_ankaa (Rigetti Ankaa-3，82量子比特)',
        'aws_emerald (IQM Emerald，54量子比特)',
        'aws_ibex (AQT IBEX Q1，1量子比特)',
        'local (本地模拟器)',
    ]
    
    @staticmethod
    def _get_local_backend(shots, suffix=''):
        """获取本地模拟器作为回退方案"""
        backend_info = {}
        if has_aer:
            backend = AerSimulator()
            backend.set_options(shots=shots)
            backend_info['backend'] = backend
            backend_info['type'] = 'local'
            backend_info['name'] = f'AerSimulator{suffix}'
        else:
            backend = BasicSimulator()
            backend_info['backend'] = backend
            backend_info['type'] = 'local'
            backend_info['name'] = f'BasicSimulator{suffix}'
        return backend_info
    
    @staticmethod
    def _print_error(title, reason, traceback_str=None, fallback=None):
        """统一格式化输出错误信息"""
        print("=" * 80)
        print(f"错误: {title}")
        print(f"原因: {reason}")
        if traceback_str:
            print(f"\n完整错误堆栈:")
            print(traceback_str)
        if fallback:
            print("=" * 80)
            print(f"回退方案: {fallback}")
        print("=" * 80)
    
    @staticmethod
    def _print_ionq_warning():
        """打印IonQ设备警告信息"""
        print(f"⚠ 注意: IonQ设备使用特殊的量子门集合 (gpi, gpi2, ms)")
        print(f"  当前代码使用标准量子门，可能需要额外的电路转换")
    
    @staticmethod
    def _print_available_aws_backends():
        """打印可用的AWS设备列表"""
        print(f"建议: 请使用其他可用的AWS设备，如:")
        for backend in QuantumBackendManager.AVAILABLE_AWS_BACKENDS:
            print(f"  - {backend}")
        print(f"\n可以使用以下命令查看所有可用设备:")
        print(f"  python aws_check.py")
    
    @staticmethod
    def _check_device_status(backend):
        """检查AWS设备状态"""
        try:
            device = backend._device
            status = device.status if hasattr(device, 'status') else None
            if status and status.value == 'OFFLINE':
                print(f"警告: {backend.name} 设备当前处于OFFLINE状态")
                QuantumBackendManager._print_available_aws_backends()
                raise ValueError(f"设备 {backend.name} 当前不可用，状态: {status.value}")
        except Exception as e:
            pass
    
    @staticmethod
    def setup_backend(backend_name, aws_region=None, shots=100, use_estimator=True):
        """
        设置量子后端

        Args:
            backend_name (str): 后端名称，支持：
                - local/local_aer: 本地模拟器
                - aws_*: AWS Braket后端（如aws_sv1, aws_aria等）
                - ibm/ibm_simulator: IBM Quantum后端
            aws_region (str, optional): AWS区域设置
            shots (int, optional): 量子采样次数，默认100
            use_estimator (bool, optional): 是否使用Estimator模式，默认True

        Returns:
            dict: 包含后端信息的字典，键包括：
                - 'backend': 量子后端对象
                - 'type': 后端类型（'local', 'aws', 'ibm'）
                - 'name': 后端名称
                - 'estimator': Estimator实例（如果use_estimator=True）

        Raises:
            ValueError: 不支持的后端名称

        Example:
            >>> backend_info = QuantumBackendManager.setup_backend('local', shots=1000)
            >>> print(backend_info['name'])
            AerSimulator
        """
        backend_info = {}
        
        if backend_name.lower() in ('local', 'local_aer'):
            backend_info = QuantumBackendManager._setup_local_backend(backend_name, shots)
        elif backend_name.lower().startswith('aws'):
            backend_info = QuantumBackendManager._setup_aws_backend(backend_name, aws_region, shots)
        elif backend_name.lower().startswith('ibm'):
            backend_info = QuantumBackendManager._setup_ibm_backend(backend_name, shots)
        else:
            raise ValueError(f"不支持的量子后端: {backend_name}")
        
        if use_estimator and 'backend' in backend_info:
            try:
                from qiskit.primitives import StatevectorEstimator
                backend_info['estimator'] = StatevectorEstimator()
            except Exception as e:
                print(f"警告: 无法创建Estimator: {e}")
                backend_info['estimator'] = None
        
        return backend_info
    
    @staticmethod
    def _setup_local_backend(backend_name, shots):
        """设置本地模拟器"""
        backend_info = {}
        try:
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
        except Exception as e:
            print(f"警告: 无法加载AerSimulator，使用基础模拟器: {e}")
            backend = BasicSimulator()
            backend_info['backend'] = backend
            backend_info['type'] = 'local'
            backend_info['name'] = 'BasicSimulator'
        return backend_info
    
    @staticmethod
    def _setup_aws_backend(backend_name, region, shots):
        """设置AWS Braket后端"""
        backend_info = {}
        
        try:
            from qiskit_braket_provider import BraketProvider
            
            if region:
                os.environ['AWS_DEFAULT_REGION'] = region
            
            provider = BraketProvider()
            
            backend_config = QuantumBackendManager.AWS_BACKENDS.get(backend_name.lower())
            if not backend_config:
                raise ValueError(f"不支持的AWS后端: {backend_name}")
            
            backend = provider.get_backend(backend_config['name'])
            print(f"✓ {backend_config['display']} 已连接")
            
            QuantumBackendManager._check_device_status(backend)
            
            backend_info['backend'] = backend
            backend_info['type'] = 'aws'
            backend_info['name'] = backend_name
            backend_info['shots'] = shots
            
        except ImportError as e:
            QuantumBackendManager._print_error(
                "无法加载AWS Braket Provider",
                str(e),
                traceback.format_exc(),
                "使用本地模拟器替代"
            )
            backend_info = QuantumBackendManager._get_local_backend(shots, ' (AWS替代)')
            
        except Exception as e:
            QuantumBackendManager._print_error(
                "AWS后端设置失败",
                str(e),
                traceback.format_exc(),
                "使用本地模拟器替代"
            )
            backend_info = QuantumBackendManager._get_local_backend(shots, ' (AWS替代)')
            
        return backend_info
    
    @staticmethod
    def _setup_ibm_backend(backend_name, shots):
        """设置IBM Quantum后端"""
        backend_info = {}
        
        try:
            if not has_ibm:
                QuantumBackendManager._print_error(
                    "IBM Quantum Runtime未安装",
                    "qiskit_ibm_runtime包未找到",
                    traceback.format_exc(),
                    "使用本地模拟器替代"
                )
                return QuantumBackendManager._get_local_backend(shots, ' (IBM替代)')
            
            if not os.environ.get('QISKIT_IBM_TOKEN'):
                print("=" * 80)
                print(f"错误: IBM Quantum凭据未设置")
                print(f"原因: 环境变量 QISKIT_IBM_TOKEN 未设置")
                print(f"\n请设置IBM Quantum API Token:")
                print(f"  export QISKIT_IBM_TOKEN='your_api_token_here'")
                print(f"或在代码中设置:")
                print(f"  os.environ['QISKIT_IBM_TOKEN'] = 'your_api_token_here'")
                print("=" * 80)
                print(f"回退方案: 使用本地模拟器替代")
                print("=" * 80)
                return QuantumBackendManager._get_local_backend(shots, ' (IBM替代)')
            
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
            QuantumBackendManager._print_error(
                "IBM Quantum后端设置失败",
                str(e),
                traceback.format_exc(),
                "使用本地模拟器替代"
            )
            backend_info = QuantumBackendManager._get_local_backend(shots)
            
        return backend_info