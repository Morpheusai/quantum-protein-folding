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


def create_noise_model(p_single=0.01, p_double=0.05, p_meas=0.03):
    """从IBM量子硬件获取真实噪声模型（带缓存）
    
    Args:
        p_single: 单比特门错误率（当无法连接IBM服务或加载缓存时使用）
        p_double: 双比特门错误率（当无法连接IBM服务或加载缓存时使用）
        p_meas: 测量错误率（当无法连接IBM服务或加载缓存时使用）
    
    Returns:
        NoiseModel: Qiskit噪声模型对象
    """
    import pickle
    
    noise_model_file = "ibm_fez_noise.pkl"
    
    # 1. 尝试从本地文件加载噪声模型
    if os.path.exists(noise_model_file):
        try:
            print("正在从本地缓存加载IBM量子硬件噪声模型...")
            with open(noise_model_file, "rb") as f:
                noise_model = pickle.load(f)
            print("✓ 成功加载本地缓存的噪声模型")
            return noise_model
        except Exception as e:
            print(f"⚠ 加载本地噪声模型失败: {e}")
            print("  - 将尝试从IBM量子硬件获取新的噪声模型")
    
    # 2. 尝试从IBM量子硬件获取噪声模型
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService 
        from qiskit_aer.noise import NoiseModel 
        
        print("正在从IBM量子硬件获取真实噪声模型...")
        service = QiskitRuntimeService() 
        backend = service.backend("ibm_fez") 
        
        noise_model = NoiseModel.from_backend(backend)
        print("✓ 成功获取IBM量子硬件噪声模型")
        print(f"  - 后端名称: {backend.name}")
        print(f"  - 噪声模型包含的门: {noise_model.basis_gates}")
        
        # 保存噪声模型到本地文件
        try:
            with open(noise_model_file, "wb") as f:
                pickle.dump(noise_model, f)
            print(f"✓ 噪声模型已保存到本地文件: {noise_model_file}")
        except Exception as e:
            print(f"⚠ 保存噪声模型到本地文件失败: {e}")
            print("  - 后续运行将需要重新从IBM量子硬件获取噪声模型")
        
        return noise_model
    except Exception as e:
        print(f"⚠ 无法从IBM量子硬件获取噪声模型: {e}")
        print("  - 将使用默认噪声模型作为替代")
        # 当无法连接IBM服务时，使用默认噪声模型
        from qiskit_aer.noise import NoiseModel, pauli_error
        
        noise_model = NoiseModel()
        
        # 1. 量子比特翻转错误（单比特门错误）
        error_single = pauli_error([('X', p_single), ('I', 1 - p_single)])
        
        # 2. 双比特门错误
        error_double = pauli_error([('XX', p_double), ('II', 1 - p_double)])
        
        # 3. 添加噪声到门操作
        noise_model.add_all_qubit_quantum_error(error_single, ['u1', 'u2', 'u3'])
        noise_model.add_all_qubit_quantum_error(error_double, ['cx'])
        
        # 4. 添加测量错误
        error_meas = pauli_error([('X', p_meas), ('I', 1 - p_meas)])
        noise_model.add_all_qubit_quantum_error(error_meas, ['measure'])
        
        return noise_model


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
    def setup_backend(backend_name, aws_region=None, shots=100, use_estimator=True, use_noise=False, noise_single=0.01, noise_double=0.05, noise_meas=0.03, **kwargs):
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
            use_noise (bool, optional): 是否使用噪声模型（仅本地模拟器支持），默认False
            noise_single (float, optional): 单比特门错误率，默认0.01
            noise_double (float, optional): 双比特门错误率，默认0.05
            noise_meas (float, optional): 测量错误率，默认0.03
            resilience_level (int, optional): 误差抑制等级 (0-3)，默认1

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
            backend_info = QuantumBackendManager._setup_local_backend(backend_name, shots, use_noise, noise_single, noise_double, noise_meas)
        elif backend_name.lower().startswith('aws'):
            backend_info = QuantumBackendManager._setup_aws_backend(backend_name, aws_region, shots)
        elif backend_name.lower().startswith('ibm'):
            # 传递 resilience_level 给 IBM 后端设置
            resilience_level = kwargs.get('resilience_level', 1)
            backend_info = QuantumBackendManager._setup_ibm_backend(backend_name, shots, resilience_level)
        else:
            raise ValueError(f"不支持的量子后端: {backend_name}")
        
        if use_estimator:
            try:
                # 尝试使用 resilience_level 初始化 Estimator (针对支持它的后端/Primitive)
                # 目前 qiskit.primitives.StatevectorEstimator 不接受 resilience_level，
                # 但如果是 IBM Runtime 的 EstimatorV2 则需要。
                # 这里为了兼容性，我们主要针对 IBM Runtime 做特殊处理，通用 Estimator 保持默认。
                
                if backend_info.get('type') == 'ibm':
                     # IBM Runtime 的 Estimator 会在 _setup_ibm_backend 中处理，或者需要在这里重新封装
                     # 由于 _setup_ibm_backend 主要是获取 backend 对象，我们在这里尝试创建 Estimator
                     # 注意：如果是使用 primitives V2，通常是在此时创建
                     pass
                
                # 对于本地模拟，继续使用 StatevectorEstimator 作为基础
                if 'estimator' not in backend_info: # 如果后端设置没创建 estimator
                    from qiskit.primitives import StatevectorEstimator
                    backend_info['estimator'] = StatevectorEstimator()

            except Exception as e:
                print(f"警告: 无法创建Estimator: {e}")
                backend_info['estimator'] = None
        
        # 尝试创建 SamplerV2 (用于现代化迁移)
        try:
            if backend_info.get('type') == 'local':
                if use_noise:
                    # 使用 BackendSamplerV2 以支持噪声模型
                    from qiskit.primitives import BackendSamplerV2
                    backend_info['sampler_v2'] = BackendSamplerV2(backend=backend_info['backend'])
                    backend_info['sampler_v2'].options.default_shots = shots
                    print(f"✓ 已配置本地 SamplerV2 (BackendSamplerV2, 支持噪声模型)")
                else:
                    from qiskit.primitives import StatevectorSampler
                    backend_info['sampler_v2'] = StatevectorSampler()
                    print(f"✓ 已配置本地 SamplerV2 (StatevectorSampler)")
            elif backend_info.get('type') == 'ibm':
                from qiskit_ibm_runtime import SamplerV2 as IBMSampler
                sampler = IBMSampler(mode=backend_info['backend'])
                backend_info['sampler_v2'] = sampler
                print(f"✓ 已配置 IBM SamplerV2 (支持现代化 Error Mitigation)")
            else:
                backend_info['sampler_v2'] = None
        except Exception as e:
            print(f"提示: 无法初始化 SamplerV2 ({e})，将使用传统接口")
            backend_info['sampler_v2'] = None
        
        return backend_info
    
    @staticmethod
    def _setup_local_backend(backend_name, shots, use_noise=False, noise_single=0.01, noise_double=0.05, noise_meas=0.03):
        """设置本地模拟器
        
        Args:
            backend_name: 后端名称
            shots: 采样次数
            use_noise: 是否使用噪声模型
            noise_single: 单比特门错误率
            noise_double: 双比特门错误率
            noise_meas: 测量错误率
        """
        backend_info = {}
        try:
            if has_aer:
                # 配置模拟器，启用噪声模型（如果需要）
                simulator_options = {}
                if use_noise:
                    noise_model = create_noise_model(p_single=noise_single, p_double=noise_double, p_meas=noise_meas)
                    simulator_options['noise_model'] = noise_model
                    print(f"✓ 已启用噪声模型 (单比特: {noise_single}, 双比特: {noise_double}, 测量: {noise_meas})")
                
                backend = AerSimulator(**simulator_options)
                backend.set_options(shots=shots)
                backend_info['backend'] = backend
                backend_info['type'] = 'local'
                backend_info['name'] = 'AerSimulator'
                backend_info['use_noise'] = use_noise
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
            
            print(f"✓ {backend_config['display']} 已连接")
            
            # 增加重试机制检查设备状态
            max_retries = 3
            import time
            for attempt in range(max_retries):
                try:
                    QuantumBackendManager._check_device_status(backend)
                    break
                except ValueError as ve:
                    if attempt < max_retries - 1:
                        print(f"⚠ 设备检查失败 (尝试 {attempt+1}/{max_retries}): {ve}")
                        print("  正在等待 5 秒后重试...")
                        time.sleep(5)
                    else:
                        raise ve
                except Exception as e:
                    print(f"⚠ 设备检查遇到未知错误: {e}")
                    # 非状态错误通常不重试或根据情况处理
                    break
            
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
    def _setup_ibm_backend(backend_name, shots, resilience_level=1):
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

            # 尝试为 IBM 后端创建带误差抑制的 Estimator
            try:
                from qiskit_ibm_runtime import EstimatorV2 as IBMEstimator
                
                # EstimatorV2 配置
                estimator = IBMEstimator(mode=backend)
                estimator.options.resilience_level = resilience_level
                backend_info['estimator'] = estimator
                print(f"✓ IBM EstimatorV2 已配置 (误差抑制等级: {resilience_level})")
            except ImportError:
                 print("⚠ qiskit_ibm_runtime.EstimatorV2 不可用，将使用标准 Estimator")
            except Exception as e:
                 print(f"⚠ 配置 IBM Estimator 失败: {e}")

            
        except Exception as e:
            QuantumBackendManager._print_error(
                "IBM Quantum后端设置失败",
                str(e),
                traceback.format_exc(),
                "使用本地模拟器替代"
            )
            backend_info = QuantumBackendManager._get_local_backend(shots)
            
        return backend_info