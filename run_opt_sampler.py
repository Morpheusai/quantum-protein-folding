#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
蛋白质折叠量子算法 - 采样器模式 (Sampler Mode)

功能特点：
1. 支持本地模拟器和AWS Braket后端
2. 使用CVaR (Conditional Value at Risk) 优化策略
3. 基于采样的能量计算方法
4. 支持多轮独立实验及结果汇总分析
5. 支持IBM量子硬件噪声模型模拟

噪声模型：
1. 支持从IBM量子硬件获取真实噪声模型（带本地缓存）
2. 支持自定义噪声参数（单比特门、双比特门、测量错误率）
3. 仅本地模拟器（local, local_aer）支持噪声模型
"""

import argparse
import os
import sys
import warnings
import datetime
import json
import csv
import numpy as np
import copy
import time
# 环境配置
current_dir = os.path.dirname(os.path.abspath(__file__))
# 强制使用 UTF-8 编码输出，适配 Windows 终端
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # 兼容旧版本 Python
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

if os.name == 'nt':
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, os.path.join(current_dir, 'src'))

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RealAmplitudes
from qiskit_braket_provider import BraketProvider
from scipy.optimize import minimize
from qiskit_algorithms.optimizers import SPSA
import traceback
from lib.job_metadata_logger import JobMetadataLogger

# 创建元数据记录器实例
metadata_logger = JobMetadataLogger("protein_folding_jobs_detailed.csv")

# 引入 Job 记录器
from lib.quantum_optimizer import JobRecorder



import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ====================
# 时间追踪类
# ====================


class QuantumTimeTracker:
    """量子计算时间追踪器 - 区分量子/经典计算时间"""

    def __init__(self):
        self.submit_time = None
        self.result_received = None
        self.iteration_start = None
        self.iteration_end = None
        self.quantum_time = 0.0
        self.queue_time = 0.0
        self.classical_time = 0.0
        self.total_time = 0.0
        self._backend_name = None

    def extract_from_job(self, job, backend_name):
        """从 job 对象提取时间信息"""
        self._backend_name = backend_name
        try:
            backend_lower = backend_name.lower()
            if backend_lower.startswith("ibm"):
                self._extract_ibm_time(job)
            elif backend_lower.startswith("aws"):
                self._extract_aws_time(job)
            else:
                self._extract_local_time()
        except Exception as e:
            print(f"    ⚠ 提取时间信息失败: {e}")
            self._extract_local_time()

    def _extract_ibm_time(self, job):
        """从 IBM Quantum job 提取时间"""
        try:
            if hasattr(job, "metrics"):
                metrics = job.metrics()
                self.quantum_time = float(
                    metrics.get("usage", {}).get("quantum_seconds", 0.0)
                )
                classical_from_ibm = float(
                    metrics.get("usage", {}).get("classical_seconds", 0.0)
                )
                timestamps = metrics.get("timestamps", {})
                if timestamps:
                    from datetime import datetime

                    created = timestamps.get("created")
                    running = timestamps.get("running")
                    if created and running:
                        t_created = datetime.fromisoformat(
                            created.replace("Z", "+00:00")
                        )
                        t_running = datetime.fromisoformat(
                            running.replace("Z", "+00:00")
                        )
                        self.queue_time = (t_running - t_created).total_seconds()
                if (
                    self.quantum_time == 0.0
                    and self.submit_time
                    and self.result_received
                ):
                    self.quantum_time = self.result_received - self.submit_time
            else:
                self._extract_local_time()
        except Exception:
            self._extract_local_time()

    def _extract_aws_time(self, job):
        """从 AWS Braket job 提取时间"""
        try:
            metadata = None
            if hasattr(job, "metadata"):
                metadata = job.metadata()
            elif hasattr(job, "_job") and hasattr(job._job, "metadata"):
                metadata = job._job.metadata()
            if metadata:
                from datetime import datetime

                created = metadata.get("createdAt")
                started = metadata.get("startedAt")
                ended = metadata.get("endedAt")
                if created:
                    t_created = datetime.fromisoformat(created.replace("Z", "+00:00"))
                if started:
                    t_started = datetime.fromisoformat(started.replace("Z", "+00:00"))
                else:
                    t_started = t_created
                if ended:
                    t_ended = datetime.fromisoformat(ended.replace("Z", "+00:00"))
                else:
                    t_ended = t_started
                if started and ended:
                    self.quantum_time = (t_ended - t_started).total_seconds()
                if created and started:
                    self.queue_time = (t_started - t_created).total_seconds()
            else:
                self._extract_local_time()
        except Exception:
            self._extract_local_time()

    def _extract_local_time(self):
        """本地模拟器：使用实际测量时间"""
        if self.submit_time and self.result_received:
            self.quantum_time = self.result_received - self.submit_time
        self.queue_time = 0.0

    def finalize(self):
        """计算最终时间"""
        if self.iteration_start and self.iteration_end:
            self.total_time = self.iteration_end - self.iteration_start
        if self.total_time > 0:
            self.classical_time = max(
                0.0, self.total_time - self.quantum_time - self.queue_time
            )

    def to_dict(self):
        """输出时间字典"""
        self.finalize()
        return {
            "quantum_time": round(self.quantum_time, 3),
            "queue_time": round(self.queue_time, 3),
            "classical_time": round(self.classical_time, 3),
            "total_time": round(self.total_time, 3),
        }


def generate_timing_summary(all_timings, backend_name):
    """生成时间汇总报告"""
    if not all_timings:
        return None
    total_quantum = sum(t.get("quantum_time", 0) for t in all_timings)
    total_queue = sum(t.get("queue_time", 0) for t in all_timings)
    total_classical = sum(t.get("classical_time", 0) for t in all_timings)
    total_time = sum(t.get("total_time", 0) for t in all_timings)
    num_iters = len(all_timings)
    return {
        "backend": backend_name,
        "total_quantum_time": round(total_quantum, 3),
        "total_queue_time": round(total_queue, 3),
        "total_classical_time": round(total_classical, 3),
        "total_time": round(total_time, 3),
        "num_iterations": num_iters,
        "average_per_iteration": {
            "quantum_time": round(total_quantum / max(1, num_iters), 3),
            "queue_time": round(total_queue / max(1, num_iters), 3),
            "classical_time": round(total_classical / max(1, num_iters), 3),
            "total_time": round(total_time / max(1, num_iters), 3),
        },
        "per_iteration": all_timings,
    }


# ====================
# 噪声模型创建
# ====================

def create_noise_model(p_single=0.01, p_double=0.05, p_meas=0.03):
    """从IBM量子硬件获取真实噪声模型（带缓存）
    
    Args:
        p_single: 单比特门错误率（当无法连接IBM服务或加载缓存时使用）
        p_double: 双比特门错误率（当无法连接IBM服务或加载缓存时使用）
        p_meas: 测量错误率（当无法连接IBM服务或加载缓存时使用）
    """
    noise_model_file = "ibm_fez_noise.pkl"
    
    # 1. 尝试从本地文件加载噪声模型
    if os.path.exists(noise_model_file):
        try:
            import pickle
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
        import pickle
        
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
        from qiskit_aer.noise import NoiseModel, pauli_error, thermal_relaxation_error
        
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

class MockResult:
    """模拟量子结果类，用于解析蛋白质结构"""
    def __init__(self, eigenstate, eigenvalue, counts, total_shots):
        self.eigenstate = eigenstate  # 字典 {bitstring: 1.0}
        self.eigenvalue = eigenvalue
        self.counts = counts
        self.total_shots = total_shots
        self.probabilities = {k[::-1]: v/total_shots for k, v in counts.items()}
    
    def get_eigenstate(self): return self.eigenstate
    def get_eigenvalue(self): return self.eigenvalue
    def get_counts(self): return self.counts
    def get_total_shots(self): return self.total_shots
    def get_most_probable_state(self):
        if not self.counts: return self.eigenstate
        max_c = -1
        best_s = None
        for s, c in self.counts.items():
            if c > max_c:
                max_c = c
                best_s = s
        return best_s

# ====================
# 参数配置
# ====================
parser = argparse.ArgumentParser(description='蛋白质折叠量子算法 - 采样器模式')
parser.add_argument('--backend', default='local', help='量子后端选择: local (本地模拟器，优先使用AerSimulator), local_aer (强制使用AerSimulator), aws_sv1 (AWS模拟器), aws_garnet (AWS量子芯片), aws_ionq (AWS IonQ量子设备), aws_forte (AWS IonQ Forte量子设备), ibm (IBM量子设备), ibm_simulator (IBM模拟器)')
parser.add_argument('--random_seed', type=int, default=42, help='随机种子，用于确保结果可重现')
parser.add_argument('--max_optimization_iterations', type=int, default=10, help='最大优化迭代次数')
parser.add_argument('--ansatz_reps', type=int, default=1, help='变分量子线路的重复层数')
parser.add_argument('--main_chain', default='APRLRFY', help='蛋白质主链氨基酸序列')
parser.add_argument('--penalty_back', type=float, default=10, help='几何约束惩罚系数')
parser.add_argument('--penalty_chiral', type=float, default=10, help='手性约束惩罚系数')
parser.add_argument('--penalty_local_overlap', type=float, default=10, help='局部重叠惩罚系数')
parser.add_argument('--alpha', type=float, default=0.25, help='CVaR参数: 选择最低能量的alpha比例样本')
parser.add_argument('--shots', type=int, default=100, help='量子测量采样次数')
parser.add_argument('--max_results', type=int, default=1, help='每次迭代的结果数量')
parser.add_argument('--aws_region', default=None, help='AWS区域设置（可选）')

# 优化增强参数
parser.add_argument('--optimizer', default='COBYLA', help='优化器选择: COBYLA (默认), SPSA, SLSQP')

# 自适应Shots参数
parser.add_argument('--adaptive_shots', action='store_true', help='开启自适应Shots策略')
parser.add_argument('--min_shots', type=int, default=100, help='自适应Shots的最小采样数 (默认为100)')
parser.add_argument('--max_shots', type=int, default=2000, help='自适应Shots的最大采样数 (默认为2000)')
parser.add_argument('--unique_structures', action='store_true', help='开启结构去重：仅返回折叠结构不同的最优结果')
parser.add_argument('--restarts', type=int, default=1, help='独立实验运行次数 (Multi-Restart)，用于避免局部最优，默认为1')
parser.add_argument('--use_noise', action='store_true', help='使用噪声模型模拟真实量子硬件噪声')
parser.add_argument('--noise_single', type=float, default=0.01, help='单比特门错误率 (默认: 0.01)')
parser.add_argument('--noise_double', type=float, default=0.05, help='双比特门错误率 (默认: 0.05)')
parser.add_argument('--noise_meas', type=float, default=0.03, help='测量错误率 (默认: 0.03)')
parser.add_argument('--dry_run', action='store_true', help='干跑模式：仅在真实提交前进行本地预检')
parser.add_argument('--resume', type=str, default=None, help='断点续传：指定结果目录以恢复历史任务')
parser.add_argument('--initial_params', type=str, default=None, help='Warm-Start：从 JSON 文件加载初始参数向量')
args = parser.parse_args()

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = os.path.join("results", f"{TIMESTAMP}_{args.backend}_sampler")
if args.use_noise:
    RESULT_DIR = RESULT_DIR + "_noise"
os.makedirs(RESULT_DIR, exist_ok=True)

# ====================
# 量子后端配置与运行逻辑
# ====================

def setup_sampler_backend(backend_name, aws_region=None, shots=1000, use_noise=False, noise_single=0.01, noise_double=0.05, noise_meas=0.03):
    """配置支持采样器模式的量子后端，兼容多种量子云服务，并支持 SamplerV2
    
    Args:
        backend_name: 后端名称
        aws_region: AWS区域（仅AWS后端需要）
        shots: 采样次数
        use_noise: 是否使用噪声模型（仅本地模拟器支持）
        noise_single: 单比特门错误率
        noise_double: 双比特门错误率
        noise_meas: 测量错误率
    """
    backend_info = {'backend': None, 'sampler_v2': None}
    try:
        if backend_name.lower() == 'aws_sv1':
            from qiskit_braket_provider import BraketProvider
            if aws_region: os.environ['AWS_DEFAULT_REGION'] = aws_region
            provider = BraketProvider()
            backend = provider.get_backend('SV1')
            print(f"✓ AWS SV1 已连接")
            backend_info['backend'] = backend
        elif backend_name.lower() == 'aws_garnet':
            from qiskit_braket_provider import BraketProvider
            if aws_region: os.environ['AWS_DEFAULT_REGION'] = aws_region
            provider = BraketProvider()
            backend = provider.get_backend('Garnet')
            print(f"✓ AWS Garnet 已连接")
            backend_info['backend'] = backend
        elif backend_name.lower() == 'aws_ionq':
            from qiskit_braket_provider import BraketProvider
            if aws_region: os.environ['AWS_DEFAULT_REGION'] = aws_region
            provider = BraketProvider()
            backend = provider.get_backend('IonQ Device')
            print(f"✓ AWS IonQ Harmony 已连接")
            backend_info['backend'] = backend
        elif backend_name.lower() == 'aws_forte':
            from qiskit_braket_provider import BraketProvider
            if aws_region: os.environ['AWS_DEFAULT_REGION'] = aws_region
            provider = BraketProvider()
            backend = provider.get_backend('Forte 1')
            print(f"✓ AWS IonQ Forte-1 已连接")
            backend_info['backend'] = backend
        elif backend_name.lower() == 'ibm':
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService()
            backend = service.least_busy(operational=True, simulator=False)
            print(f"✓ IBM 量子后端已连接: {backend.name}")
            backend_info['backend'] = backend
            # 初始化 IBM SamplerV2
            try:
                from qiskit_ibm_runtime import SamplerV2 as IBMSampler
                backend_info['sampler_v2'] = IBMSampler(mode=backend)
                print(f"✓ 已配置 IBM SamplerV2 接口")
            except: pass
        elif backend_name.lower() == 'ibm_simulator':
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService()
            backend = service.backend("ibmq_qasm_simulator")
            print(f"✓ IBM 模拟器已连接: {backend.name}")
            backend_info['backend'] = backend
        elif backend_name.lower() in ('local_aer', 'local'):
            try:
                from qiskit_aer import AerSimulator
                
                # 配置模拟器，启用噪声模型（如果需要）
                simulator_options = {}
                if use_noise:
                    noise_model = create_noise_model(p_single=noise_single, p_double=noise_double, p_meas=noise_meas)
                    simulator_options['noise_model'] = noise_model
                    print(f"✓ 已启用噪声模型 (单比特: {noise_single}, 双比特: {noise_double}, 测量: {noise_meas})")
                
                backend = AerSimulator(**simulator_options)
                print("✓ 本地 AerSimulator 已就绪")
                
                # 根据是否使用噪声选择合适的 Sampler
                if use_noise:
                    # 使用 BackendSamplerV2 以支持噪声模型
                    from qiskit.primitives import BackendSamplerV2
                    backend_info['sampler_v2'] = BackendSamplerV2(backend=backend)
                    backend_info['sampler_v2'].options.default_shots = shots
                    print("✓ 已配置本地 BackendSamplerV2 (支持噪声模型)")
                else:
                    # 使用 StatevectorSampler 进行理想模拟
                    from qiskit.primitives import StatevectorSampler
                    backend_info['sampler_v2'] = StatevectorSampler()
                    print("✓ 已配置本地 StatevectorSampler (理想模拟)")
            except ImportError:
                from qiskit.providers.basic_provider import BasicSimulator
                backend = BasicSimulator()
                print("✓ 本地 BasicSimulator 已就绪")
            backend_info['backend'] = backend
        else:
            from qiskit.providers.basic_provider import BasicSimulator
            backend = BasicSimulator()
            backend_info['backend'] = backend
            print("✓ 本地 BasicSimulator 已就绪")
            
        return backend_info
    except Exception as e:
        print(f"✗ 后端设置失败: {e}")
        from qiskit.providers.basic_provider import BasicSimulator
        backend_info['backend'] = BasicSimulator()
        return backend_info

# ====================
# 能量计算逻辑 (优化版)
# ====================

def preprocess_qubit_op(qubit_op):
    """
    预处理哈密顿量，提取 Z 算子的比特索引，加速后续能量计算。
    
    Returns:
        list: [(coeff, [z_indices])] 列表
    """
    processed_terms = []
    for pauli_str, coeff in qubit_op.to_list():
        if coeff.real == 0:
            continue
        z_indices = []
        # 处理反转逻辑，因为 Qiskit 比特顺序是反向的
        for i, char in enumerate(reversed(pauli_str)):
            if char == 'Z':
                z_indices.append(i)
            elif char == 'X' or char == 'Y':
                # 基态期望值为 0，标记为无效项
                z_indices = None
                break
        if z_indices is not None:
            processed_terms.append((coeff.real, z_indices))
    return processed_terms

def estimate_energy_fast(bitstring, processed_terms):
    """
    使用预处理后的项快速计算单个比特串的能量。
    """
    energy = 0.0
    bit_list = [int(b) for b in reversed(bitstring)] # 虽然还是用了 reversed，但比多次循环字符串快
    
    for coeff, z_indices in processed_terms:
        val = 1.0
        for idx in z_indices:
            if bit_list[idx] == 1:
                val *= -1.0
        energy += coeff * val
    return energy

def calculate_cvar_energy_efficient(counts, qubit_op, alpha, processed_terms=None):
    """
    高效的 CVaR 计算，不使用列表展开，直接使用频率统计。
    """
    if processed_terms is None:
        processed_terms = preprocess_qubit_op(qubit_op)
        
    # 计算每个唯一比特串的能量
    bitstring_energies = []
    for bitstring, count in counts.items():
        e = estimate_energy_fast(bitstring, processed_terms)
        bitstring_energies.append((e, count))
    
    # 按能量升序排列
    bitstring_energies.sort(key=lambda x: x[0])
    
    total_shots = sum(counts.values())
    num_keep = max(1, int(total_shots * alpha))
    
    # 累加前 alpha 比例的能量
    cvar_sum = 0.0
    processed_shots = 0
    
    for energy, count in bitstring_energies:
        remaining_needed = num_keep - processed_shots
        if remaining_needed <= 0:
            break
        
        shots_to_take = min(count, remaining_needed)
        cvar_sum += energy * shots_to_take
        processed_shots += shots_to_take
    
    return cvar_sum / num_keep

def extract_top_results(counts, qubit_op, top_n, processed_terms=None):
    """
    从量子测量结果中提取能量最低的 top_n 个结果。
    """
    if processed_terms is None:
        processed_terms = preprocess_qubit_op(qubit_op)
        
    bitstring_energies = []
    for bitstring, count in counts.items():
        energy = estimate_energy_fast(bitstring, processed_terms)
        bitstring_energies.append((bitstring, energy, count))
    
    bitstring_energies.sort(key=lambda x: x[1])
    return bitstring_energies[:top_n]

# ====================
# 主计算流程
# ====================
@metadata_logger
def main():
    print(f"🚀 正在启动蛋白质折叠计算任务 (采样器模式)...")
    print(f"🔗 启动服务器计算任务 | 蛋白质序列: {args.main_chain}")
    
    # 显示参数配置
    print(f"\n[配置] 运行参数概览:")
    print(f"  - 量子后端: {args.backend}")
    print(f"  - 优化器: {args.optimizer}")
    print(f"  - 最大优化迭代次数: {args.max_optimization_iterations}")
    print(f"  - Ansatz 重复层数: {args.ansatz_reps}")
    print(f"  - CVaR Alpha 参数: {args.alpha}")
    print(f"  - 量子采样次数 (Shots): {args.shots}")
    print(f"  - 实验轮数 (Restarts): {args.restarts}")
    print(f"  - 结果保存目录: {RESULT_DIR}")
    
    # 噪声模型信息
    print(f"  - 噪声模拟: {'✅ 已启用' if args.use_noise else '❌ 已禁用'}")
    if args.use_noise:
        print(f"    * 1-Qubit Error: {args.noise_single}")
        print(f"    * 2-Qubit Error: {args.noise_double}")
        print(f"    * Readout Error: {args.noise_meas}")
    
    # 导入蛋白质折叠相关模块
    print(f"\n[模块] 正在导入核心组件...")
    try:
        from protein_folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
        from protein_folding.peptide.peptide import Peptide
        from protein_folding.protein_folding_problem import ProteinFoldingProblem
        from protein_folding.penalty_parameters import PenaltyParameters
        from protein_folding.protein_folding_result import ProteinFoldingResult
        from src.protein_folding.utils.detailed_pdb_generator import convert_xyz_to_detailed_pdb
        print("✅ 蛋白质折叠核心模块导入成功")
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return

    # 1. 构建蛋白质折叠问题模型
    print(f"\n[结构] 正在构建蛋白质主链与物理模型...")
    # 创建肽对象：包含主链序列和侧链序列（这里设为空字符串）
    peptide = Peptide(args.main_chain, [""] * len(args.main_chain))
    
    # 定义约束项的惩罚系数
    penalty_terms = PenaltyParameters(args.penalty_chiral, args.penalty_back, args.penalty_local_overlap)
    
    # 创建蛋白质折叠问题实例
    problem = ProteinFoldingProblem(peptide, MiyazawaJerniganInteraction(), penalty_terms)
    
    # 生成对应的量子比特哈密顿量
    qubit_op = problem.qubit_op()
    num_qubits = qubit_op.num_qubits  # 获取所需量子比特数
    print(f"✅ 问题建模完成 | 所需量子比特数: {num_qubits}")

    # 预处理哈密顿量以加速计算
    processed_hamiltonian = preprocess_qubit_op(qubit_op)
    print(f"✅ 哈密顿量预处理完成 ({len(processed_hamiltonian)} 项)")

    # 2. 根据参数选择合适的量子计算后端
    print(f"\n[算法] 正在配置量子后端与采样器...")
    backend_info = setup_sampler_backend(
        args.backend, 
        args.aws_region, 
        args.shots,
        args.use_noise,
        noise_single=args.noise_single,
        noise_double=args.noise_double,
        noise_meas=args.noise_meas
    )
    backend = backend_info['backend']
    sampler_v2 = backend_info['sampler_v2']
    print(f"✅ 后端初始化就绪: {args.backend}")

    print(f"\n[电路] 正在构建与转译 Variational Ansatz...")
    # 使用RealAmplitudes作为变分波函数 ansatz
    ansatz = RealAmplitudes(num_qubits=num_qubits, reps=args.ansatz_reps)
    
    # 将ansatz分解为基本门操作，便于后续处理
    decomposed = ansatz.decompose()
    
    # 创建清洁的量子电路（移除barrier等不必要的元素）
    clean_circuit = QuantumCircuit(num_qubits)
    for inst in decomposed.data:
        # 移除barrier指令，因为它不影响量子计算结果但可能影响转译
        if inst.operation.name != 'barrier':
            clean_circuit.append(inst.operation, inst.qubits, inst.clbits)
    
    # 添加测量门，以便获取量子比特的状态
    clean_circuit.measure_all()
    
    # 根据后端类型决定是否转译电路
    if args.backend.lower() in ['local', 'local_aer', 'local_qiskit']:
        transpiled_circuit = clean_circuit
    else:
        transpiled_circuit = transpile(clean_circuit, backend=backend,
                                        initial_layout=list(range(clean_circuit.num_qubits)),
                                        optimization_level=3
                                      )
    
    # 缓存电路指标
    ops = transpiled_circuit.count_ops()
    single_q = sum(ops.get(gate, 0) for gate in ['h', 'rx', 'ry', 'rz', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg', 'u1', 'u2', 'u3', 'p'])
    two_q = sum(ops.get(gate, 0) for gate in ['cx', 'cz', 'swap', 'ecr', 'rxx', 'ryy', 'rzz'])
    depth = transpiled_circuit.depth()
    
    print(f"✅ 电路转译完成:")
    print(f"   - 逻辑量子比特: {clean_circuit.num_qubits}")
    print(f"   - 物理量子比特: {transpiled_circuit.num_qubits}")
    print(f"   - 门统计: 单比特={single_q}, 双比特={two_q}, 深度={depth}")
    
    # 干跑模式 (Dry-Run)
    if args.dry_run:
        print("\n    [Dry-Run] 正在执行 Sampler 预检...")
        try:
            test_params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
            if sampler_v2:
                job = sampler_v2.run([(transpiled_circuit, test_params)], shots=10)
            else:
                bound_c = transpiled_circuit.assign_parameters(test_params)
                job = backend.run(bound_c, shots=10)
            JobRecorder.record_job(job, RESULT_DIR, label="dry_run_sampler")
            job.result()
            print("    ✓ Sampler 预检通过")
        except Exception as e:
            print(f"    ❌ Sampler 预检失败: {e}")
            raise e

    all_conv_data = []
    all_iteration_timings = []
    global_candidates = [] # 初始化全球候选池
    best_energy = float('inf')
    best_results_tuple = None # 将在第一次实验后初始化
    best_trial_idx = 1
    
    # 创建全局 iteration_details.csv 文件（在 RESULT_DIR 层）
    iteration_csv_path = os.path.join(RESULT_DIR, "iteration_details.csv")
    try:
        with open(iteration_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['trial_idx', 'iteration', 'backend', 'total_gates', 'single_qubit_gates', 'two_qubit_gates', 'circuit_depth', 'shots', 'energy', 'cumulative_shots', 'cvar_energy', 'quantum_time', 'queue_time', 'classical_time', 'total_time'])
        print(f"✓ 已创建全局迭代记录文件: {iteration_csv_path}")
    except Exception as e:
        print(f"⚠ 创建 CSV 文件失败: {e}")
        iteration_csv_path = None
    
    # ====================
    # 多实验运行循环 (Multi-Restart Loop)
    # ====================
    print(f"\n[任务] 开始多轮运行实验 (共 {args.restarts} 轮)...")
    
    best_energy = float('inf')
    best_trial_idx = -1
    best_results_tuple = None
    global_candidates = []

    for trial_idx in range(args.restarts):
        current_seed = args.random_seed + trial_idx
        np.random.seed(current_seed)
        print(f"\n▶ 实验 {trial_idx + 1}/{args.restarts} 启动 (随机种子: {current_seed})")
        
        # 定义状态变量以追踪当前实验
        convergence_history = []
        std_history = []
        iteration_results = []
        all_top_energies = []
        cumulative_shots_history = []
        iteration_shots_history = []
        all_iteration_timings = []
        
        _last_objective_time = [time.perf_counter()]
        _trial_best_energy = [float('inf')] # 追踪当前 Trial 的最佳能量
        
        trial_dir = os.path.join(RESULT_DIR, f"trial_{trial_idx + 1}")
        os.makedirs(trial_dir, exist_ok=True)
        iteration_result_dir = os.path.join(trial_dir, "iter_all_results")
        os.makedirs(iteration_result_dir, exist_ok=True)

        def objective_function(params):
            iter_tracker = QuantumTimeTracker()
            iter_tracker.iteration_start = time.perf_counter()
            
            # 执行量子电路
            current_shots = args.shots
            if args.adaptive_shots:
                max_iter = args.max_optimization_iterations
                current_iter = len(convergence_history)
                if max_iter > 1:
                    ratio = min(current_iter / (max_iter - 1), 1.0)
                    current_shots = int(args.min_shots + (args.max_shots - args.min_shots) * ratio)

            iter_tracker.submit_time = time.perf_counter()
            try:
                if sampler_v2:
                    job = sampler_v2.run([(transpiled_circuit, params)], shots=current_shots)
                    result = job.result()[0]
                    data_name = 'meas' if 'meas' in result.data else next(iter(result.data))
                    counts = result.data[data_name].get_counts()
                else:
                    bound_circ = transpiled_circuit.assign_parameters(params)
                    job = backend.run(bound_circ, shots=current_shots)
                    counts = job.result().get_counts()
            except Exception as e:
                print(f"   ❌ 量子计算作业失败: {e}")
                return 100.0
            
            iter_tracker.result_received = time.perf_counter()
            iter_tracker.iteration_end = iter_tracker.result_received
            iter_tracker.extract_from_job(job, args.backend)
            
            actual_shots = sum(counts.values())
            # 使用高效算法计算 CVaR 能量
            energy = calculate_cvar_energy_efficient(counts, qubit_op, args.alpha, processed_hamiltonian)
            
            # 更新历史记录
            convergence_history.append(energy)
            iteration_shots_history.append(actual_shots)
            cumulative_shots_history.append((cumulative_shots_history[-1] if cumulative_shots_history else 0) + actual_shots)

            # 方差估算 (标准差)
            energies_std = [estimate_energy_fast(bs, processed_hamiltonian) for bs in counts.keys()]
            weights_std = [counts[bs] for bs in counts.keys()]
            mean_std = np.average(energies_std, weights=weights_std) if energies_std else 0
            std_e = np.sqrt(np.average((np.array(energies_std) - mean_std)**2, weights=weights_std)) if energies_std else 0
            std_history.append(std_e)
            
            # 提取 Top 结果
            top_results = extract_top_results(counts, qubit_op, args.max_results, processed_hamiltonian)
            top_energies = [r[1] for r in top_results]
            all_top_energies.append(top_energies)

            # 检查是否发现新低能量（结构解析节流）
            energy_improved = False
            if energy < _trial_best_energy[0]:
                _trial_best_energy[0] = energy
                energy_improved = True
            
            protein_structure_info = {}
            if energy_improved or len(convergence_history) % 10 == 0:
                try:
                    best_bitstring = top_results[0][0]
                    raw_res = MockResult({best_bitstring: 1.0}, top_results[0][1], {best_bitstring: top_results[0][2]}, actual_shots)
                    p_res = problem.interpret(raw_res)
                    xyz_data = p_res.protein_shape_file_gen.get_xyz_data()
                    protein_structure_info = {
                        "turn_sequence": p_res.turn_sequence,
                        "xyz_coordinates": [[row[0], float(row[1]), float(row[2]), float(row[3])] for row in xyz_data] if xyz_data is not None else [],
                        "best_bitstring": best_bitstring
                    }
                    if energy_improved:
                        print(f"   ⭐ 发现更低能量: {energy:.4f} | 序列: {p_res.turn_sequence}")
                except:
                    pass

            # 时间统计
            current_time = time.perf_counter()
            iter_time = current_time - _last_objective_time[0]
            q_time = iter_tracker.quantum_time
            queue_time = iter_tracker.queue_time
            c_time = max(0.0, iter_time - q_time - queue_time)
            
            timing_info = {
                "quantum_time": round(q_time, 3),
                "queue_time": round(queue_time, 3),
                "classical_time": round(c_time, 3),
                "total_time": round(iter_time, 3)
            }
            if len(convergence_history) % 5 == 0:
                print(f"   [迭代 {len(convergence_history)}] 能量={energy:.4f} | ⏱ 量子={q_time:.2f}s, 经典={c_time:.2f}s")
            
            all_iteration_timings.append(timing_info)
            _last_objective_time[0] = current_time
            
            iteration_data = {
                "iteration": len(convergence_history),
                "cvar_energy": energy,
                "top_results": top_results,
                "actual_shots": actual_shots,
                "timing": timing_info,
                "protein_structure": protein_structure_info,
                "counts": counts
            }
            iteration_results.append(iteration_data)
            
            # 保存单轮 JSON (包含结构信息)
            with open(os.path.join(iteration_result_dir, f'iteration_{len(convergence_history)}_result.json'), 'w') as f:
                json.dump({
                    "iteration": len(convergence_history),
                    "energy": energy,
                    "top_results": [{"bitstring": r[0], "energy": float(r[1]), "count": int(r[2])} for r in top_results],
                    "timing": timing_info,
                    "protein_structure": protein_structure_info,
                    "shots": actual_shots,
                    "circuit_metrics": {"single": single_q, "two": two_q, "depth": depth}
                }, f, indent=2)
            
            # 写入全局 CSV
            if iteration_csv_path:
                try:
                    with open(iteration_csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([trial_idx + 1, len(convergence_history), args.backend, single_q + two_q, single_q, two_q, depth, actual_shots, energy, cumulative_shots_history[-1], energy, q_time, queue_time, c_time, iter_time])
                except: pass
            
            return energy

        # 5. 执行优化计算
        print(f"   初始参数已生成，维度: {ansatz.num_parameters}")
        initial_params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
        
        if args.optimizer.upper() == 'SPSA':
            optimizer = SPSA(maxiter=args.max_optimization_iterations)
            result = optimizer.minimize(objective_function, initial_params)
            res = type('obj', (object,), {'x': result.x, 'fun': result.fun})
        else:
            method = 'COBYLA' if args.optimizer.upper() == 'COBYLA' else 'SLSQP'
            res = minimize(objective_function, initial_params, method=method, options={'maxiter': args.max_optimization_iterations})
        
        print(f"  [优化结束] 最低能量: {res.fun:.4f}")
        
        for iter_data in iteration_results:
            for cand in iter_data.get("top_results", []): global_candidates.append(cand)

        if best_results_tuple is None or res.fun < best_energy:
            best_energy = res.fun
            best_trial_idx = trial_idx + 1
            best_results_tuple = (res, convergence_history, std_history, iteration_results, all_top_energies, cumulative_shots_history, iteration_shots_history)
            print(f"  ★ 发现新最佳实验: {best_trial_idx} (能量: {best_energy:.4f})")
            
        print(f"  [实验 {trial_idx + 1} 结束] 当前最佳能量: {best_energy:.4f}")


    # ====================
    # 全局分析与去重 (Global Analysis)
    # ====================
    print(f"\n" + "="*60)
    print(f"📊 所有实验结束 | 最佳实验: Trial {best_trial_idx} (最低能量: {best_energy:.4f})")
    print(f"正在进行跨运行结果汇总与去重...")
    
    # 解包最佳结果供可视化
    res, convergence_history, std_history, iteration_results, all_top_energies, cumulative_shots_history, iteration_shots_history = best_results_tuple
    
    # 构建可视化数据对象
    all_conv_data = [{
        'counts': list(range(len(convergence_history))), 
        'values': convergence_history, 
        'stds': std_history,
        'cumulative_shots': cumulative_shots_history, 
        'iteration_shots': iteration_shots_history,
        'label': f'Best Trial ({best_trial_idx})'
    }]
    
    # 添加 Top 子能量曲线
    if all_top_energies:
        for i in range(min(args.max_results, len(all_top_energies[0]))):
            top_i_energies = [energies[i] if i < len(energies) else None for energies in all_top_energies]
            valid_counts = []
            valid_energies = []
            for j, eval_energy in enumerate(top_i_energies):
                if eval_energy is not None:
                    valid_counts.append(j)
                    valid_energies.append(eval_energy)
            all_conv_data.append({
                'counts': valid_counts, 
                'values': valid_energies, 
                'label': f'Trial {best_trial_idx} Top {i+1} Energy'
            })
    
    # 汇总去重逻辑
    unique_candidates = {}
    for iter_data in iteration_results:
        for bs, en, count in iter_data.get("top_results", []):
            if bs not in unique_candidates or en < unique_candidates[bs][0]:
                unique_candidates[bs] = (en, count)
    
    sorted_candidates = sorted(unique_candidates.items(), key=lambda x: x[1][0])
    
    if args.unique_structures:
        print(f"    - 正在执行跨运行结构去重 (目标: {args.max_results} 个不同结构)...")
        final_top_results = []
        seen_structures = set()
        
        for bitstring, (energy, count) in sorted_candidates:
            try:
                temp_mock_result = MockResult({bitstring: 1.0}, energy, {bitstring: count}, 100)
                temp_result = problem.interpret(temp_mock_result)
                structure_sig = str(temp_result.turn_sequence)
                
                if structure_sig not in seen_structures:
                    seen_structures.add(structure_sig)
                    final_top_results.append((bitstring, energy, count))
                    
                if len(final_top_results) >= args.max_results:
                    break
            except Exception as e:
                print(f"    ⚠ 解析比特串 {bitstring} 失败: {e}")
                continue
    else:
        final_top_results = [(bs, en, count) for bs, (en, count) in sorted_candidates[:args.max_results]]

    # --- 结果解析与保存 ---
    print(f"\n[存档] 正在保存最终结果到 {RESULT_DIR}...")
    
    # 查找最佳实验的最终计数
    final_counts = iteration_results[-1].get("counts", {}) if iteration_results else {}
    total_shots = sum(final_counts.values()) if final_counts else args.shots
    
    # 为每个最优结果生成蛋白质结构和文件
    for idx, (bitstring, energy, count) in enumerate(final_top_results):
        print(f"    - 结果 {idx+1}/{len(final_top_results)}: 能量 = {energy:.4f}, 采样频次 = {count}")
        
        raw_res = MockResult({bitstring: 1.0}, energy, final_counts, total_shots)
        result = problem.interpret(raw_res)
        print(f"      转向序列: {result.turn_sequence}")
        
        xyz_data = result.protein_shape_file_gen.get_xyz_data()
        
        # 1. 保存 JSON
        json_path = os.path.join(RESULT_DIR, f'result_{idx+1}_energy_{energy:.4f}.json')
        with open(json_path, 'w') as f:
            json.dump({"rank": idx+1, "energy": energy, "turn": result.turn_sequence, "bits": bitstring, "count": count}, f, indent=2)
        
        # 2. 保存 PDB
        if xyz_data is not None:
            pdb_path = os.path.join(RESULT_DIR, f'structure_{idx+1}_energy_{energy:.4f}.pdb')
            convert_xyz_to_detailed_pdb(xyz_data, pdb_path, f"Structure {idx+1}")
        
        # 3. 保存 PNG
        try:
            fig_struct = result.get_figure(title=f"Result {idx+1} (E={energy:.4f})")
            png_path = os.path.join(RESULT_DIR, f"structure_{idx+1}_energy_{energy:.4f}.png")
            fig_struct.savefig(png_path)
            plt.close(fig_struct)
        except: pass

    # 生成优化过程的收敛图
    plt.figure(figsize=(12, 8))
    for idx, data in enumerate(all_conv_data):
        label = data.get('label', f'Run {idx+1}')
        # CVaR Energy 使用虚线，其他使用实线
        if 'CVaR' in label:
            plt.plot(data['counts'], data['values'], marker='o', label=label, linewidth=2, linestyle='--')
        else:
            plt.plot(data['counts'], data['values'], marker='o', label=label, linewidth=2)
    plt.xlabel("Evaluation Counts")
    plt.ylabel("Energy")
    plt.title(f"VQE Sampler Convergence ({args.main_chain}) - CVaR (alpha={args.alpha})")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULT_DIR, "vqe_optimization_summary.png"))
    plt.close()
    
    # 双y轴图：展示能量和每次迭代的shots数
    if iteration_shots_history and len(iteration_shots_history) > 0:
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # 定义颜色映射，为每个运行结果使用相同颜色的不同样式
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_conv_data)))
        
        # 绘制能量曲线
        energy_lines = []
        labels = []
        for idx, data in enumerate(all_conv_data):
            color = colors[idx]
            label = data.get('label', f'Run {idx+1}')
            # CVaR Energy 使用虚线，其他使用实线
            if 'CVaR' in label:
                values = np.array(data['values'])
                line, = ax1.plot(data['counts'], values, marker='o', label=label, 
                         linewidth=3, color=color, linestyle='--')
                # 绘制误差带
                stds = np.array(data.get('stds', []))
                if len(stds) == len(values):
                     ax1.fill_between(data['counts'], values - stds, values + stds, 
                                      color=color, alpha=0.2, label=f'{label} Std Range')
            else:
                line, = ax1.plot(data['counts'], data['values'], marker='o', label=label, 
                         linewidth=3, color=color)
            energy_lines.append(line)
            labels.append(label)
        ax1.set_xlabel('Evaluation Counts')
        ax1.set_ylabel('Energy', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3)
        
        # 创建第二个y轴用于shots
        shots_bars = []
        ax2 = ax1.twinx()
        # 使用每次迭代的实际shots数，而不是累计shots
        bars = ax2.bar(range(len(iteration_shots_history)), iteration_shots_history, alpha=0.3, width=0.5, 
               color='gray', edgecolor='gray', linewidth=0.5, 
               label='Shots per Iteration')
        ax2.set_ylabel('Shots per Iteration', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        
        # 只使用能量曲线的图例，避免重复
        ax1.legend(energy_lines, labels, loc='upper right')
        
        plt.title(f"VQE Sampler Convergence with Shots per Iteration ({args.main_chain}) - CVaR (alpha={args.alpha})")
        fig.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, "vqe_sampler_optimization_with_shots.png"))
        plt.close()
        print(f"✓ 带shots信息的VQE采样器优化图已保存为 {os.path.join(RESULT_DIR, 'vqe_sampler_optimization_with_shots.png')}")
    
    print(f"\n✓ 蛋白质折叠计算完成！")
    
    # 导出最优参数向量（供 Warm-Start 使用）
    try:
        print(f"\n正在导出最优参数向量...")
        # 从最佳实验的优化结果中获取最优参数
        if res is not None and hasattr(res, 'x'):
            best_params = res.x
            best_params_file = os.path.join(RESULT_DIR, "best_params.json")
            with open(best_params_file, 'w') as f:
                json.dump({
                    "initial_point": best_params.tolist(),
                    "energy": best_energy,
                    "num_parameters": len(best_params),
                    "best_trial": best_trial_idx
                }, f, indent=2)
            print(f"✓ 最优参数已保存到: {best_params_file}")
            print(f"  - 参数维度: {len(best_params)}")
            print(f"  - 最优能量: {best_energy:.4f}")
            print(f"  - 最佳实验: Trial {best_trial_idx}")
            print(f"  提示: 可使用 --initial_params {best_params_file} 进行 Warm-Start")
        else:
            print(f"⚠ 无法找到最优参数，跳过导出")
    except Exception as e:
        print(f"⚠ 导出最优参数时出错: {e}")
    
    print(f"\n🎉 蛋白质折叠计算任务完成！")
    
    # 时间汇总
    try:
        timing_summary = generate_timing_summary(all_iteration_timings, args.backend)
        if timing_summary:
            timing_path = os.path.join(RESULT_DIR, "timing_summary.json")
            with open(timing_path, "w", encoding="utf-8") as f:
                json.dump(timing_summary, f, indent=2)
            print(f"\n[时间汇总]:")
            print(f"  - 总量子时间: {timing_summary['total_quantum_time']:.3f}s")
            print(f"  - 总队列时间: {timing_summary['total_queue_time']:.3f}s")
            print(f"  - 总经典时间: {timing_summary['total_classical_time']:.3f}s")
            print(f"  - 任务总耗时: {timing_summary['total_time']:.3f}s")
    except Exception:
        pass

    # 导出最优向量
    try:
        if res is not None and hasattr(res, 'x'):
            best_params_file = os.path.join(RESULT_DIR, "best_params.json")
            with open(best_params_file, 'w') as f:
                json.dump({
                    "initial_point": res.x.tolist(),
                    "energy": best_energy,
                    "best_trial": best_trial_idx
                }, f, indent=2)
            print(f"✅ 最优参数向量已导出: {best_params_file}")
            print(f"   (提示: 可使用 --initial_params 进行热启动)")
    except Exception as e:
        print(f"⚠ 导出最优参数失败: {e}")
    
    print(f"\n🚀 所有结果已保存在: {RESULT_DIR}")
    
    # 最后生成 metrics.json 以供 metadata_logger 汇总到 protein_folding_jobs_detailed.csv
    try:
        metrics_path = os.path.join(RESULT_DIR, "metrics.json")
        total_shots = sum(iteration_shots_history) if iteration_shots_history else 0
        total_iters = len(convergence_history)
        
        # 整理转译指标
        transpile_metrics = {
            "logical_qubits": clean_circuit.num_qubits,
            "physical_qubits": transpiled_circuit.num_qubits,
            "depth": depth,
            "total_gates": single_q + two_q,
            "single_qubit_gates": single_q,
            "two_qubit_gates": two_q,
            "ops": dict(transpiled_circuit.count_ops())
        }
        
        # 整理收敛指标
        convergence_metrics = {
            "iterations_total": total_iters,
            "best_energy": float(best_energy),
            "convergence_history": convergence_history
        }
        
        metrics = {
            "backend": args.backend,
            "shots_requested": int(args.shots),
            "shots_actual_total": int(total_shots),
            "iteration_count": int(total_iters),
            "outcome_summary": f"min_energy={float(best_energy):.6f}",
            "qubits_used": clean_circuit.num_qubits,
            "qubits_full": problem._qubit_op_full().num_qubits if hasattr(problem, '_qubit_op_full') else clean_circuit.num_qubits,
            "total_quantum_time": timing_summary.get("total_quantum_time", 0.0) if timing_summary else 0.0,
            "total_queue_time": timing_summary.get("total_queue_time", 0.0) if timing_summary else 0.0,
            "total_classical_time": timing_summary.get("total_classical_time", 0.0) if timing_summary else 0.0,
            "total_gates": transpile_metrics["total_gates"],
            "single_qubit_gates": transpile_metrics["single_qubit_gates"],
            "two_qubit_gates": transpile_metrics["two_qubit_gates"],
            "circuit_depth": transpile_metrics["depth"],
            "transpile_metrics": transpile_metrics,
            "convergence_metrics": convergence_metrics
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"✅ 全局指标摘要已保存 (用于元数据记录): {metrics_path}")
    except Exception as e:
        print(f"⚠ 生成最终指标失败: {e}")


if __name__ == "__main__":
    main()
