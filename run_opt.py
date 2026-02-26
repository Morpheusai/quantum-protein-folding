#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
蛋白质折叠量子算法 

功能特性：
1. 支持多种量子后端：本地模拟器、AWS SV1、AWS Garnet、IBM量子设备
2. 解决了量子后端测量门冲突问题
3. 使用VQE算法优化蛋白质折叠能量
4. 支持多轮独立实验及结果汇总分析

兼容性修复：
1. 解决 AWS 后端 "Cannot measure previously measured qubit" 报错
2. 强制在每次迭代中使用干净的 Ansatz 副本
3. 改用Estimator替代Sampler，提高各后端兼容性
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
# 设置UTF-8环境以解决Windows编码问题并配置模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.name == 'nt':  # Windows系统
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'

sys.path.insert(0, os.path.join(current_dir, 'src'))
from lib.job_metadata_logger import JobMetadataLogger

# 创建元数据记录器实例
metadata_logger = JobMetadataLogger("protein_folding_jobs_detailed.csv")
# 1. 图形后端配置：设置非GUI后端以解决服务器环境下的显示问题
# 在服务器环境下，无头模式运行，避免因缺少显示设备而引发的 RuntimeError
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# 忽略警告信息，保持输出简洁
warnings.filterwarnings('ignore')

# ====================
# 参数配置
# ====================
parser = argparse.ArgumentParser()
parser.add_argument('--backend', default='local', help='local (本地模拟器，优先使用AerSimulator), local_aer (强制使用AerSimulator), aws_sv1, aws_garnet, aws_ionq, aws_forte, ibm, ibm_simulator')
parser.add_argument('--random_seed', type=int, default=23)
parser.add_argument('--max_optimization_iterations', type=int, default=10)
parser.add_argument('--ansatz_reps', type=int, default=1)
parser.add_argument('--main_chain', default='APRLRFY')
parser.add_argument('--penalty_back', type=float, default=10)
parser.add_argument('--penalty_chiral', type=float, default=10)
parser.add_argument('--penalty_local_overlap', type=float, default=10)
parser.add_argument('--shots', type=int, default=100)
parser.add_argument('--restarts', type=int, default=1, help='多重启实验次数')
parser.add_argument('--max_results', type=int, default=5, help='要保留的最高质量不同结构数量')
parser.add_argument('--unique_structures', action='store_true', default=True, help='启用跨实验结构去重')
parser.add_argument('--adaptive_shots', action='store_true', help='启用自适应采样')
parser.add_argument('--min_shots', type=int, default=100)
parser.add_argument('--max_shots', type=int, default=1000)
parser.add_argument('--aws_region', default=None)
parser.add_argument('--dry_run', action='store_true', help='干跑模式：仅在真实提交前进行本地预检')
parser.add_argument('--resume', type=str, default=None, help='断点续传：指定结果目录以恢复历史任务')
parser.add_argument('--initial_params', type=str, default=None, help='Warm-Start：从 JSON 文件加载初始参数向量')
args = parser.parse_args()

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = os.path.join("results", f"{TIMESTAMP}_{args.backend}_estimator")
os.makedirs(RESULT_DIR, exist_ok=True)


# ====================
# 辅助类 (保持独立脚本特性)
# ====================

class MockResult:
    """模拟量子结果类，用于解析蛋白质结构"""
    def __init__(self, eigenstate, eigenvalue):
        self.eigenstate = eigenstate # 字典 {bitstring: prob}
        self.eigenvalue = eigenvalue

class JobRecorder:
    """AWS 任务记录器，用于持久化 Job ID 以防丢失"""
    
    @staticmethod
    def record_job(job, result_dir, label="job"):
        """记录任务 ID 到文件"""
        if not result_dir:
            return
            
        jobs_dir = os.path.join(result_dir, "jobs")
        os.makedirs(jobs_dir, exist_ok=True)
        
        try:
            job_id = "local_simulation"
            if hasattr(job, 'job_id'):
                job_id = job.job_id()
            
            job_info = {
                "job_id": job_id,
                "timestamp": str(np.datetime64('now')),
                "status": "submitted",
                "label": label
            }
            
            import datetime
            ts = datetime.datetime.now().strftime("%H%M%S_%f")
            job_file = os.path.join(jobs_dir, f"{label}_{ts}.json")
            
            with open(job_file, 'w') as f:
                json.dump(job_info, f, indent=2)
            
            print(f"    [AWS 保护] 任务已记录: {job_id} -> {os.path.basename(job_file)}")
        except Exception as e:
            print(f"    ⚠ 记录任务失败: {e}")

class AdaptiveEstimatorV2:
    """EstimatorV2 包装器，支持自适应精度和任务记录"""
    
    def __init__(self, base_estimator, args, result_dir=None):
        self.base_estimator = base_estimator
        self.args = args
        self.result_dir = result_dir
        self.call_count = 0
        self.max_iter = getattr(args, 'max_optimization_iterations', 100)
    
    def run(self, pubs, precision=None):
        self.call_count += 1
        
        label = f"estimator_step_{self.call_count}"
        
        # 如果启用了自适应精度
        if hasattr(self.args, 'adaptive_shots') and self.args.adaptive_shots:
            min_shots = getattr(self.args, 'min_shots', 100)
            max_shots = getattr(self.args, 'max_shots', 1000)
            
            # 使用简单的进度比例计算当前 shots
            ratio = min(self.call_count / (self.max_iter * 2), 1.0)
            current_shots = int(min_shots + (max_shots - min_shots) * ratio)
            
            # 将 shots 转换为 precision: precision = 1 / sqrt(shots)
            adaptive_precision = 1 / (current_shots**0.5)
            
            # 如果外部明确传入了更小的 precision，则保留小的那个以保证质量
            if precision is not None:
                precision = min(precision, adaptive_precision)
            else:
                precision = adaptive_precision
        
        job = self.base_estimator.run(pubs, precision=precision)
        # 记录 Job ID
        JobRecorder.record_job(job, self.result_dir, label=f"estimator_step_{self.call_count}")
        return job

class StructuralLoggingEstimator:
    """Estimator 的包装器，拦截 run 调用以获取完整的参数向量并执行结构解析"""
    def __init__(self, base_estimator, structural_callback, result_dir=None):
        self.base_estimator = base_estimator
        self.structural_callback = structural_callback
        self.result_dir = result_dir
        self.call_count = 0
    
    def run(self, pubs, precision=None):
        self.call_count += 1
        # pubs 是一个列表，每个元素通常是 (ansatz, operator, parameters)
        for pub in pubs:
            ansatz, operator, parameters = pub
            # parameters 可能包含多个点 (评估批次)
            if parameters.ndim == 1:
                # 单个评估点
                self.structural_callback(parameters)
            else:
                # 批量评估点
                for p in parameters:
                    self.structural_callback(p)
        
        job = self.base_estimator.run(pubs, precision=precision)
        # 记录 Job ID (如果 base_estimator 不是 AdaptiveEstimatorV2，由这里补齐记录)
        if not isinstance(self.base_estimator, AdaptiveEstimatorV2):
             JobRecorder.record_job(job, self.result_dir, label=f"struct_step_{self.call_count}")
             
        return job

# ====================
# 量子后端配置与运行逻辑
# ====================

def setup_v2_backend(backend_name, aws_region=None, shots=1000):
    """适配 Qiskit 2.x V2 Primitives，使用Estimator而非Sampler以提高各后端兼容性"""
    info = {'backend': None, 'estimator': None}
    try:
        if backend_name.lower() == 'aws_sv1':
            from qiskit_braket_provider import BraketProvider
            from qiskit.primitives import BackendEstimatorV2
            if aws_region: os.environ['AWS_DEFAULT_REGION'] = aws_region
            provider = BraketProvider()
            backend = provider.get_backend('SV1')
            info['backend'] = backend
            estimator = BackendEstimatorV2(backend=backend)
            # 设置 default_precision 以确保硬件执行正确的采样次数
            estimator.options.default_precision = 1 / (shots**0.5)
            estimator.options.default_shots = shots
            info['estimator'] = estimator
            print(f"✓ AWS SV1 (V2 Estimator) 已连接")
        elif backend_name.lower() == 'aws_garnet':
            from qiskit_braket_provider import BraketProvider
            from qiskit.primitives import BackendEstimatorV2
            if aws_region: os.environ['AWS_DEFAULT_REGION'] = aws_region
            provider = BraketProvider()
            backend = provider.get_backend('Garnet')
            info['backend'] = backend
            estimator = BackendEstimatorV2(backend=backend)
            # V2 Primitives 优先使用精度(precision)控制采样
            estimator.options.default_precision = 1 / (shots**0.5)
            estimator.options.default_shots = shots
            info['estimator'] = estimator
            print(f"✓ AWS Garnet (V2 Estimator) 已连接")
        elif backend_name.lower() == 'aws_ionq':
            from qiskit_braket_provider import BraketProvider
            from qiskit.primitives import BackendEstimatorV2
            if aws_region: os.environ['AWS_DEFAULT_REGION'] = aws_region
            provider = BraketProvider()
            # 获取 IonQ Harmony (11 qubits)
            backend = provider.get_backend('IonQ Device')
            info['backend'] = backend
            estimator = BackendEstimatorV2(backend=backend)
            estimator.options.default_precision = 1 / (shots**0.5)
            estimator.options.default_shots = shots
            info['estimator'] = estimator
            print(f"✓ AWS IonQ Harmony (V2 Estimator) 已连接")
        elif backend_name.lower() == 'aws_forte':
            from qiskit_braket_provider import BraketProvider
            from qiskit.primitives import BackendEstimatorV2
            if aws_region: os.environ['AWS_DEFAULT_REGION'] = aws_region
            provider = BraketProvider()
            # 获取 IonQ Forte-1 (30+ qubits)
            backend = provider.get_backend('Forte 1')
            info['backend'] = backend
            estimator = BackendEstimatorV2(backend=backend)
            estimator.options.default_precision = 1 / (shots**0.5)
            estimator.options.default_shots = shots
            info['estimator'] = estimator
            print(f"✓ AWS IonQ Forte-1 (V2 Estimator) 已连接")
        elif backend_name.lower() == 'ibm':
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
                service = QiskitRuntimeService()
                # 使用性能最好的可用设备
                backend = service.least_busy(operational=True, simulator=False)
                info['backend'] = backend
                estimator = Estimator(mode=backend)
                # 设置选项
                # 设置选项，确保精度与采样数同步更新
                estimator.options.default_precision = 1 / (shots**0.5)
                estimator.options.default_shots = shots
                info['estimator'] = estimator
                print(f"✓ IBM 量子后端已连接: {backend.name}")
            except Exception as e:
                print(f"✗ IBM 真实硬件连接失败: {e}")
                print("  提示: 请确保已通过 'qiskit-ibm-runtime' 配置 IBM Quantum 访问凭据")
                # 回退到IBM模拟器
                try:
                    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
                    service = QiskitRuntimeService()
                    # 使用IBM模拟器
                    backend = service.backend("ibmq_qasm_simulator")
                    info['backend'] = backend
                    estimator = Estimator(mode=backend)
                    estimator.options.default_precision = 1 / (shots**0.5)
                    estimator.options.default_shots = shots
                    info['estimator'] = estimator
                    print(f"✓ IBM 模拟器已连接: {backend.name}")
                except Exception as sim_e:
                    print(f"✗ IBM 模拟器连接也失败: {sim_e}")
                    # 最终回退到本地模拟器
                    from qiskit.primitives import StatevectorEstimator
                    info['estimator'] = StatevectorEstimator()
                    print("  ✓ 已回退到本地 StatevectorEstimator")
        elif backend_name.lower() == 'local_aer':
            # 使用AerSimulator作为本地后端
            try:
                from qiskit_aer import AerSimulator
                from qiskit.primitives import BackendEstimatorV2
                backend = AerSimulator()
                estimator = BackendEstimatorV2(backend=backend)
                estimator.options.default_precision = 1 / (shots**0.5)
                estimator.options.default_shots = shots
                info['backend'] = backend
                info['estimator'] = estimator
                print("✓ 本地 AerSimulator (V2 Estimator) 已就绪")
            except ImportError:
                print("⚠ AerSimulator 不可用，回退到 StatevectorEstimator")
                from qiskit.primitives import StatevectorEstimator
                info['estimator'] = StatevectorEstimator()
                print("  ✓ 已回退到本地 StatevectorEstimator")
        else:
            # 默认本地后端，优先使用AerSimulator，如果不可用则回退到StatevectorEstimator
            try:
                from qiskit_aer import AerSimulator
                from qiskit.primitives import BackendEstimatorV2
                backend = AerSimulator()
                estimator = BackendEstimatorV2(backend=backend)
                estimator.options.default_precision = 1 / (shots**0.5)
                estimator.options.default_shots = shots
                info['backend'] = backend
                info['estimator'] = estimator
                print("✓ 本地 AerSimulator (V2 Estimator) 已就绪")
            except ImportError:
                from qiskit.primitives import StatevectorEstimator
                info['estimator'] = StatevectorEstimator()
                print("✓ 本地 StatevectorEstimator 已就绪")
        # 尝试创建 SamplerV2
        try:
            if info.get('backend') and backend_name.lower() in ('local', 'local_aer'):
                from qiskit.primitives import StatevectorSampler
                info['sampler_v2'] = StatevectorSampler()
            elif info.get('backend') and backend_name.lower() == 'ibm':
                from qiskit_ibm_runtime import SamplerV2 as IBMSampler
                info['sampler_v2'] = IBMSampler(mode=info['backend'])
            else:
                info['sampler_v2'] = None
        except Exception:
            info['sampler_v2'] = None
            
        return info
    except Exception as e:
        print(f"✗ 后端设置失败: {e}")
        from qiskit.primitives import StatevectorEstimator
        return {'backend': None, 'estimator': StatevectorEstimator()}

def run_vqe_iteration(qubit_op, ansatz, optimizer, estimator, backend=None, result_dir=None, problem=None, sampler=None, trial_idx=1, iteration_csv_path=None):
    from qiskit_algorithms import VQE
    from qiskit import transpile

    working_ansatz = copy.deepcopy(ansatz)
    working_ansatz.remove_final_measurements() 

    if backend is not None:
        working_ansatz = transpile(working_ansatz, backend=backend,
                                  initial_layout=list(range(working_ansatz.num_qubits)) if args.backend not in ['local', 'aws_sv1', 'ibm_simulator'] else None,
                                  optimization_level=3)

    convergence = {
        'counts': [], 
        'values': [], 
        'cumulative_shots': [], 
        'iteration_shots': [],
        'single_qubit_gates': [],
        'two_qubit_gates': [],
        'circuit_depth': [],
        'best_params': None,
        'best_energy': float('inf')
    }
    
    # 迭代结果保存目录
    iteration_result_dir = None
    if result_dir:
        iteration_result_dir = os.path.join(result_dir, "iter_all_results")
        os.makedirs(iteration_result_dir, exist_ok=True)
    
    def callback(eval_count, parameters, mean, std):
        convergence['counts'].append(eval_count)
        convergence['values'].append(mean)
        
        if mean < convergence['best_energy']:
            convergence['best_energy'] = mean
            convergence['best_params'] = parameters

        current_step_shots = args.shots
        if args.adaptive_shots:
            ratio = min(len(convergence['counts']) / args.max_optimization_iterations, 1.0)
            current_step_shots = int(args.min_shots + (args.max_shots - args.min_shots) * ratio)

        convergence['iteration_shots'].append(current_step_shots)
        convergence['cumulative_shots'].append(sum(convergence['iteration_shots']))

        # 计算电路门数量和深度
        try:
            bound_circuit = working_ansatz.assign_parameters({p: v for p, v in zip(working_ansatz.parameters, parameters)})
            ops = bound_circuit.count_ops()
            single_qubit_gates = 0
            two_qubit_gates = 0
            for op, count in ops.items():
                if op in ['h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg', 'rx', 'ry', 'rz', 'u1', 'u2', 'u3']:
                    single_qubit_gates += count
                elif op in ['cx', 'cz', 'swap', 'ch', 'cy', 'cs', 'csdg', 'ct', 'ctdg', 'crx', 'cry', 'crz', 'cu1', 'cu2', 'cu3']:
                    two_qubit_gates += count
            circuit_depth = bound_circuit.depth()
            convergence['single_qubit_gates'].append(single_qubit_gates)
            convergence['two_qubit_gates'].append(two_qubit_gates)
            convergence['circuit_depth'].append(circuit_depth)
        except Exception as e:
            print(f"⚠ 计算电路门数量和深度时出错: {e}")
            convergence['single_qubit_gates'].append(0)
            convergence['two_qubit_gates'].append(0)
            convergence['circuit_depth'].append(0)

        # 保存每步迭代的结果 (如果有 result_dir 和 problem)
        if iteration_result_dir and problem:
            try:
                save_path = os.path.join(iteration_result_dir, f'iteration_{eval_count}_result.json')
                
                protein_info = state.get('last_protein_info', {})

                with open(save_path, 'w') as f:
                    json.dump({
                        "iteration": eval_count,
                        "energy": float(mean),
                        "shots": int(current_step_shots),
                        "protein_structure": protein_info,
                        "single_qubit_gates": convergence['single_qubit_gates'][-1],
                        "two_qubit_gates": convergence['two_qubit_gates'][-1],
                        "circuit_depth": convergence['circuit_depth'][-1]
                    }, f, indent=2)
                
                # 写入 CSV 记录
                if iteration_csv_path:
                    try:
                        single_q = convergence['single_qubit_gates'][-1]
                        two_q = convergence['two_qubit_gates'][-1]
                        total_gates = single_q + two_q
                        with open(iteration_csv_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                trial_idx,
                                eval_count,
                                args.backend if hasattr(args, 'backend') else 'unknown',
                                total_gates,
                                single_q,
                                two_q,
                                convergence['circuit_depth'][-1],
                                current_step_shots,
                                float(mean),
                                convergence['cumulative_shots'][-1]
                            ])
                    except Exception as e:
                        print(f"⚠ 写入 CSV 失败: {e}")
            except Exception:
                pass

    # 结构解析回调 (用于 StructuralLoggingEstimator)
    state = {'last_protein_info': {}}
    
    def structural_callback(parameters):
        if not (result_dir and problem and sampler):
            return
        
        protein_info = {}
        try:
            sampling_circuit = working_ansatz.copy()
            if not sampling_circuit.get_instructions('measure'):
                 sampling_circuit.measure_all()
            
            param_dict = {p: v for p, v in zip(sampling_circuit.parameters, parameters)}
            bound_circuit = sampling_circuit.assign_parameters(param_dict)
            
            current_step_shots = args.shots if args else 100
            if hasattr(sampler, 'run'):
                job = sampler.run([bound_circuit], shots=current_step_shots)
                sampler_result = job.result()
                counts = sampler_result[0].data.meas.get_counts()
            else:
                job = sampler.run(bound_circuit, shots=current_step_shots)
                sampler_result = job.result()
                counts = sampler_result.quasi_dists[0].binary_probabilities()
            
            best_bs = max(counts, key=counts.get)
            
            class MockResultIter:
                def __init__(self, bs):
                    self.eigenstate = {bs: 1.0}
                    self.eigenvalue = 0.0
            
            raw_iter_res = MockResultIter(best_bs)
            interpreted = problem.interpret(raw_iter_res)
            xyz = interpreted.protein_shape_file_gen.get_xyz_data()
            
            protein_info = {
                "turn_sequence": interpreted.turn_sequence if hasattr(interpreted, 'turn_sequence') else "",
                "xyz_coordinates": [list(row) for row in xyz] if xyz is not None else [],
                "best_bitstring": best_bs
            }
        except Exception as e:
            print(f"⚠ Warning: Structural interpretation in callback failed: {e}")
            protein_info = {"error": str(e)}
        
        state['last_protein_info'] = protein_info

    # 最终包装 Estimator
    final_estimator = estimator
    if args.adaptive_shots:
        final_estimator = AdaptiveEstimatorV2(final_estimator, args, result_dir=result_dir)
    
    # 包装结构解析逻辑
    final_estimator = StructuralLoggingEstimator(final_estimator, structural_callback, result_dir=result_dir)

    # 干跑模式 (Dry-Run)
    if args.dry_run:
        print("\n    [Dry-Run] 正在执行 Estimator 预检...")
        try:
            dry_params = np.random.uniform(-np.pi, np.pi, working_ansatz.num_parameters)
            job = final_estimator.run([(working_ansatz, qubit_op, dry_params)], precision=0.3)
            job.result()
            print("    ✓ Estimator 预检通过")
        except Exception as e:
            print(f"    ❌ Estimator 预检失败: {e}")
            raise e

    vqe = VQE(estimator=final_estimator, ansatz=working_ansatz, optimizer=optimizer, callback=callback)
    result = vqe.compute_minimum_eigenvalue(qubit_op)
    
    # 使用 result.optimal_point 重建最优参数字典（这是最准确的）
    # 回调函数中的 parameters 可能不完整，所以这里必须重建
    convergence['best_params_dict'] = {p: v for p, v in zip(working_ansatz.parameters, result.optimal_point)}
    
    return result, convergence



# ====================
# 蛋白质折叠主计算流程
# ====================
@metadata_logger
def main():
    print("正在启动蛋白质折叠算法...")
    print(f"🚀 启动服务器计算任务 | 序列: {args.main_chain}")
    
    # 显示参数配置
    print(f"\n参数配置:")
    print(f"  - 量子后端: {args.backend}")
    print(f"  - 随机种子: {args.random_seed}")
    print(f"  - 最大优化迭代次数: {args.max_optimization_iterations}")
    print(f"  - Ansatz重复次数: {args.ansatz_reps}")
    print(f"  - 主链序列: {args.main_chain}")
    print(f"  - 几何约束惩罚: {args.penalty_back}")
    print(f"  - 手性约束惩罚: {args.penalty_chiral}")
    print(f"  - 局部重叠惩罚: {args.penalty_local_overlap}")
    print(f"  - 量子采样次数: {args.shots}")
    print(f"  - 最大结果数量: {args.max_results}")
    print(f"  - 结果目录: {RESULT_DIR}")

    try:
        from qiskit_algorithms.utils import algorithm_globals
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit.circuit.library import RealAmplitudes
        
        # 蛋白质折叠核心模块导入
        # Miyazawa-Jernigan相互作用模型：用于计算氨基酸间的相互作用能量
        from protein_folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
        # Peptide类：表示待折叠的肽链结构
        from protein_folding.peptide.peptide import Peptide
        # ProteinFoldingProblem类：将蛋白质折叠问题转换为量子计算问题
        from protein_folding.protein_folding_problem import ProteinFoldingProblem
        # PenaltyParameters类：定义约束项的惩罚系数
        from protein_folding.penalty_parameters import PenaltyParameters
        
        print("✓ 成功导入蛋白质折叠模块")
        
        # 设置随机种子
        algorithm_globals.random_seed = args.random_seed
        print("✓ 随机种子设置完成")

        # 加载 Warm-Start 初始参数 (如果指定)
        if args.initial_params:
            try:
                with open(args.initial_params, 'r') as f:
                    params_data = json.load(f)
                    args.initial_point = np.array(params_data['initial_point'])
                    print(f"✓ 已加载 Warm-Start 参数向量 (维度: {len(args.initial_point)})")
            except Exception as e:
                print(f"⚠ 加载初始参数失败: {e}，将使用随机初始化")
                args.initial_point = None
        else:
            args.initial_point = None
    except ImportError as e:
        print(f"✗ 模块导入失败: {e}")
        return

    # 定义蛋白质主链
    print(f"\n正在定义蛋白质结构...")
    main_chain = args.main_chain
    print(f"✓ 主链序列: {main_chain}")
    
    # 定义侧链
    side_chains = [""] * len(main_chain)  # 本例中不考虑侧链
    print(f"✓ 侧链序列: {side_chains}")
    
    # 创建相互作用模型
    print(f"\n正在创建相互作用模型...")
    mj_interaction = MiyazawaJerniganInteraction()
    print("✓ Miyazawa-Jernigan相互作用模型创建完成")
    
    # 定义惩罚参数
    print(f"\n正在设置物理约束参数...")
    penalty_back = args.penalty_back
    penalty_chiral = args.penalty_chiral
    penalty_1 = args.penalty_local_overlap
    penalty_terms = PenaltyParameters(penalty_chiral, penalty_back, penalty_1)
    print(f"✓ 惩罚参数设置完成: chiral={penalty_chiral}, back={penalty_back}, local_overlap={penalty_1}")
    
    # 创建肽对象
    print(f"\n正在创建肽对象...")
    peptide = Peptide(main_chain, side_chains)
    print("✓ 肽对象创建完成")
    
    # 创建蛋白质折叠问题
    print(f"\n正在构建蛋白质折叠问题...")
    problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)
    qubit_op = problem.qubit_op()
    print(f"✓ 量子比特算子构建完成: {qubit_op}")
    print(f"  - 量子比特数量: {qubit_op.num_qubits}")
    
    # 初始化后端与算法组件
    print(f"\n正在使用VQE算法求解...")
    backend_info = setup_v2_backend(args.backend, args.aws_region, args.shots)
    optimizer = COBYLA(maxiter=args.max_optimization_iterations)
    print("✓ 优化器设置完成")
    print(f"  - 使用优化器: {type(optimizer).__name__}")
    print(f"  - 最大迭代次数: {args.max_optimization_iterations}")
    
    # 构建变分量子线路 Ansatz (RealAmplitudes 默认不含测量门)
    # RealAmplitudes 是一种常用的参数化量子线路，适用于变分量子特征求解器
    base_ansatz = RealAmplitudes(num_qubits=qubit_op.num_qubits, reps=args.ansatz_reps)
    print("✓ 变分波函数设置完成")
    print(f"  - 量子比特数量: {base_ansatz.num_qubits}")
    print(f"  - 参数数量: {base_ansatz.num_parameters}")
    print(f"  - 电路重复次数: {args.ansatz_reps}")

    all_conv_data = []
    global_candidates = [] # [(bitstring, energy, trial_idx)]

    print(f"\n正在执行 {args.restarts} 轮独立实验 (Multi-Restart)...")
    if args.adaptive_shots:
        print(f"  - 已启用自适应采样: {args.min_shots} -> {args.max_shots}")
    
    best_overall_energy = float('inf')
    best_restart_idx = -1

    # 创建全局 iteration_details.csv 文件（在 RESULT_DIR 层）
    iteration_csv_path = os.path.join(RESULT_DIR, "iteration_details.csv")
    try:
        with open(iteration_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['trial_idx', 'iteration', 'backend', 'total_gates', 'single_qubit_gates', 'two_qubit_gates', 'circuit_depth', 'shots', 'energy', 'cumulative_shots'])
        print(f"✓ 已创建全局迭代记录文件: {iteration_csv_path}")
    except Exception as e:
        print(f"⚠ 创建 CSV 文件失败: {e}")
        iteration_csv_path = None

    # Define a MockResult class for interpreting bitstrings not directly from VQE
    class MockResult:
        def __init__(self, eigenstate_dict, eigenvalue):
            self.eigenstate = eigenstate_dict
            self.eigenvalue = eigenvalue
            # Add a dummy optimal_point if needed by interpret, though it usually uses eigenstate
            self.optimal_point = None 

    for i in range(args.restarts):
        print(f"\n============================================================")
        print(f"--- 实验 {i+1}/{args.restarts} (Multi-Restart) ---")
        algorithm_globals.random_seed = args.random_seed + i
        print(f"随机种子: {algorithm_globals.random_seed}")
        
        # 执行VQE迭代
        # 为每轮实验创建子目录
        trial_dir = os.path.join(RESULT_DIR, f"trial_{i+1}")
        os.makedirs(trial_dir, exist_ok=True)
        
        raw_result, conv_data = run_vqe_iteration(
            qubit_op, base_ansatz, optimizer, 
            backend_info['estimator'], backend_info.get('backend'),
            result_dir=trial_dir, problem=problem, sampler=backend_info.get('sampler_v2'),
            trial_idx=i+1, iteration_csv_path=iteration_csv_path
        )
        
        trial_label = f'Run {i+1}'
        conv_data['label'] = trial_label
        all_conv_data.append(conv_data)
        
        current_energy = float(raw_result.eigenvalue.real) # Ensure real part for energy
        print(f"  [优化结束] 能量: {current_energy:.4f}")
        
        if current_energy < best_overall_energy:
            best_overall_energy = current_energy
            best_restart_idx = i + 1
            print(f"  ★ 发现新最佳实验: {i+1} (能量: {current_energy:.4f})")

        # 收集候选
        print(f"    - 正在从最优参数还原结构候选...")
        try:
            # 使用该轮实验实际使用的 Ansatz 和最佳参数映射
            trial_ansatz = conv_data.get('ansatz', base_ansatz)
            best_params_dict = conv_data.get('best_params_dict')
            
            if best_params_dict:
                best_circuit = trial_ansatz.assign_parameters(best_params_dict)
            else:
                best_params = raw_result.optimal_point
                best_circuit = trial_ansatz.assign_parameters(best_params)
            
            is_simulator = args.backend.lower() in ['local', 'aws_sv1', 'ibm_simulator']
            if is_simulator or backend_info.get('backend') is None:
                from qiskit.quantum_info import Statevector
                probs = Statevector.from_instruction(best_circuit).probabilities_dict()
                # Get the bitstring with the highest probability
                top_bitstring = max(probs, key=probs.get) if probs else None
                if top_bitstring:
                    global_candidates.append((top_bitstring, current_energy, i+1))
                else:
                    print(f"    ⚠ 提取候选失败: 无法从模拟器状态向量中获取有效比特串。")
            else:
                from qiskit import transpile
                from qiskit.primitives import BackendSamplerV2
                # Transpile and measure for hardware/sampler
                meas_circuit = transpile(best_circuit, backend=backend_info['backend'], optimization_level=3)
                meas_circuit.measure_all()
                sampler = BackendSamplerV2(backend=backend_info['backend'])
                job = sampler.run([meas_circuit], shots=max(100, args.shots)) # Use at least 100 shots for sampling
                sampler_res = job.result()[0]
                counts = getattr(sampler_res.data, list(sampler_res.data.keys())[0]).get_counts()
                # Get the bitstring with the highest count
                top_bitstring = max(counts, key=counts.get) if counts else None
                if top_bitstring:
                    global_candidates.append((top_bitstring, current_energy, i+1))
                else:
                    print(f"    ⚠ 提取候选失败: 无法从采样器结果中获取有效比特串。")
        except Exception as e:
            print(f"    ⚠ 提取候选失败: {e}")

    # 全局汇总与去重
    print(f"\n============================================================")
    print(f"所有实验结束。最佳实验: {best_restart_idx} (能量: {best_overall_energy:.4f})")
    print(f"正在进行跨运行结果汇总与去重...")
    
    global_candidates.sort(key=lambda x: x[1]) # Sort by energy
    final_top_results = []
    seen_structures = set()
    
    for bitstring, energy, restart_idx in global_candidates:
        if len(final_top_results) >= args.max_results:
            break
        
        if bitstring is None: # Skip if bitstring extraction failed
            continue

        try:
            temp_res = MockResult({bitstring: 1.0}, energy) # Create a mock result for interpretation
            interpreted = problem.interpret(raw_result=temp_res)
            struct_key = interpreted.turn_sequence # Use turn_sequence for deduplication
            
            if args.unique_structures:
                if struct_key not in seen_structures:
                    seen_structures.add(struct_key)
                    final_top_results.append((bitstring, energy, restart_idx, interpreted))
            else:
                # If not enforcing unique structures, just add it
                final_top_results.append((bitstring, energy, restart_idx, interpreted))
        except Exception as e:
            print(f"    ⚠ 结构分析失败 (bitstring: {bitstring}, energy: {energy:.4f}): {e}")
            continue

    print(f"    - 完成。共找到 {len(final_top_results)} 个独特候选结构。")

    # 循环保存最终结果
    for idx, (bitstring, energy, restart_idx, result) in enumerate(final_top_results):
        print(f"\n  - 结果 {idx+1}/{len(final_top_results)}: 能量 = {energy:.4f}, 来自实验 {restart_idx}")
        
        # Ensure result is interpreted if it wasn't already (should be by now)
        if result is None:
            temp_res = MockResult({bitstring: 1.0}, energy)
            result = problem.interpret(raw_result=temp_res)

        energy_str = f"{abs(energy):.4f}".replace('.', '_') # For filename safety
        
        # Extract XYZ data
        xyz_data = None
        try:
            xyz_data = result.protein_shape_file_gen.get_xyz_data()
            print(f"    ✓ 坐标数据获取完成")
        except Exception as e:
            print(f"    ⚠ 获取坐标数据时出错: {e}")

        # JSON
        result_data = {
            "result_index": idx + 1,
            "original_restart": restart_idx,
            "energy": energy,
            "main_turns": result.protein_shape_decoder.main_turns,
            "side_turns": result.protein_shape_decoder.side_turns,
            "turn_sequence": result.turn_sequence,
            "main_chain_sequence": args.main_chain,
            "sequence": args.main_chain,
            "shots_requested": args.shots,
            "optimization_convergence": {
                "evaluation_counts": all_conv_data[restart_idx-1]['counts'],
                "energy_values": all_conv_data[restart_idx-1]['values'],
                "cumulative_shots": all_conv_data[restart_idx-1]['cumulative_shots'],
                "iteration_shots": all_conv_data[restart_idx-1]['iteration_shots'],
                "single_qubit_gates": all_conv_data[restart_idx-1]['single_qubit_gates'],
                "two_qubit_gates": all_conv_data[restart_idx-1]['two_qubit_gates'],
                "circuit_depth": all_conv_data[restart_idx-1]['circuit_depth']
            },
            "xyz_coordinates": [list(row) for row in xyz_data] if xyz_data is not None and len(xyz_data) > 0 else []
        }
        json_path = os.path.join(RESULT_DIR, f'result_{idx+1}_energy_{energy_str}.json')
        with open(json_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        print(f"    ✓ 核心参数已保存为 {json_path}")
        
        # PDB
        if xyz_data is not None:
            from src.protein_folding.utils.detailed_pdb_generator import convert_xyz_to_detailed_pdb
            pdb_path = os.path.join(RESULT_DIR, f'structure_{idx+1}_energy_{energy_str}.pdb')
            convert_xyz_to_detailed_pdb(xyz_data, pdb_path, f"Structure {idx+1} (E={energy:.4f})")
            print(f"    ✓ 详细PDB文件已保存为 {pdb_path}")
        
        # PNG
        try:
            fig_struct = result.get_figure(title=f"Result {idx+1} (E={energy:.4f})")
            png_path = os.path.join(RESULT_DIR, f"structure_{idx+1}_energy_{energy_str}.png")
            fig_struct.savefig(png_path)
            plt.close(fig_struct)
            print(f"    ✓ 3D图形已保存为 {png_path}")
        except Exception as e:
            print(f"    ⚠ 生成3D图形时出错: {e}")

    # 显示量子计算统计信息
    print(f"\n  - 量子比特数量: {base_ansatz.num_qubits}")
    print(f"  - 电路参数数量: {base_ansatz.num_parameters}")
    print(f"  - 总函数评估次数: {sum(len(data['counts']) for data in all_conv_data) if all_conv_data else 0}")
    
    # 汇总绘图
    try:
        print(f"\n正在生成VQE优化过程的折线图...")
            
        # 基础版本：仅能量收敛图
        plt.figure(figsize=(12, 8))
        for idx, data in enumerate(all_conv_data):
            plt.plot(data['counts'], data['values'], marker='o', label=data['label'], linewidth=2)
        
        plt.xlabel("Evaluation Counts")
        plt.ylabel("Energy")
        plt.title(f"VQE Convergence Comparison ({args.main_chain})")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(RESULT_DIR, "vqe_optimization_summary.png"))
        plt.close()
        print(f"✓ VQE优化过程图已保存为 {os.path.join(RESULT_DIR, 'vqe_optimization_summary.png')}")
            
        # 双y轴图：展示能量和每次迭代的shots数（改进版）
        if all('iteration_shots' in data and len(data['iteration_shots']) > 0 for data in all_conv_data):
            fig, ax1 = plt.subplots(figsize=(12, 8))
            
            # 定义颜色映射，为每个运行结果使用相同颜色的不同样式
            colors = plt.cm.tab10(np.linspace(0, 1, len(all_conv_data)))
            
            # 创建第二个y轴用于shots（先绘制shots，这样折线图会显示在上面）
            shots_bars = []
            ax2 = ax1.twinx()
            # 使用每次迭代的实际shots数，而不是累计shots
            # 统一使用灰色，与 run_opt_sampler.py 保持一致
            for idx, data in enumerate(all_conv_data):
                # Ensure counts and iteration_shots have the same length
                if len(data['counts']) == len(data['iteration_shots']):
                    bars = ax2.bar(data['counts'], data['iteration_shots'], alpha=0.15, width=0.5, 
                           color='gray', edgecolor='gray', linewidth=0.5, 
                           label=f'Run {idx+1} Shots per Iteration')
                    shots_bars.append(bars)
            ax2.set_ylabel('Shots per Iteration', color='black')
            ax2.tick_params(axis='y', labelcolor='black')
            
            # 绘制能量曲线（后绘制，这样会显示在shots柱状图上面）
            energy_lines = []
            for idx, data in enumerate(all_conv_data):
                color = colors[idx]
                line, = ax1.plot(data['counts'], data['values'], marker='o', label=data['label'], 
                         linewidth=3, color=color)
                energy_lines.append(line)
            ax1.set_xlabel('Evaluation Counts')
            ax1.set_ylabel('Energy', color='black')
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.grid(True, alpha=0.3)
            
            # 只使用能量曲线的图例，避免重复
            ax1.legend(energy_lines, [f'Run {idx+1}' for idx in range(len(energy_lines))], loc='upper right')
            
            plt.title(f"VQE Convergence with Shots per Iteration ({args.main_chain})")
            fig.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, "vqe_optimization_with_shots.png"))
            plt.close()
            print(f"✓ 带shots信息的VQE优化图已保存为 {os.path.join(RESULT_DIR, 'vqe_optimization_with_shots.png')}")
            
    except Exception as e:
        print(f"⚠ 生成VQE优化过程图时出错: {e}")
    

    print(f"\n✓ 蛋白质折叠计算完成！")
    
    # 导出最优参数向量（供 Warm-Start 使用）
    try:
        print(f"\n正在导出最优参数向量...")
        # 从最佳实验的收敛数据中获取最优参数
        best_trial_data = all_conv_data[best_restart_idx - 1]  # best_restart_idx 从 1 开始
        # 优先从 VQE 结果中直接获取最优参数（这是最准确的）
        if 'best_params_dict' in best_trial_data and best_trial_data['best_params_dict']:
            best_params_dict = best_trial_data['best_params_dict']
            best_params = np.array(list(best_params_dict.values()))
        elif 'best_params' in best_trial_data and best_trial_data['best_params'] is not None:
            # 使用回调函数中的参数（可能不完整，作为备选）
            best_params = best_trial_data['best_params']
        else:
            print(f"⚠ 无法找到最优参数，跳过导出")
            best_params = None
        
        if best_params is not None:
            best_params_file = os.path.join(RESULT_DIR, "best_params.json")
            with open(best_params_file, 'w') as f:
                json.dump({
                    "initial_point": best_params.tolist(),
                    "energy": best_overall_energy,
                    "num_parameters": len(best_params),
                    "best_trial": best_restart_idx
                }, f, indent=2)
            print(f"✓ 最优参数已保存到: {best_params_file}")
            print(f"  - 参数维度: {len(best_params)}")
            print(f"  - 最优能量: {best_overall_energy:.4f}")
            print(f"  - 最佳实验: Trial {best_restart_idx}")
            print(f"  提示: 可使用 --initial_params {best_params_file} 进行 Warm-Start")
    except Exception as e:
        print(f"⚠ 导出最优参数时出错: {e}")
    
    print(f"🎉 任务完成。结果目录: {RESULT_DIR}")
    
    try:
        metrics_path = os.path.join(RESULT_DIR, "metrics.json")
        total_shots = 0
        total_iters = 0
        for d in all_conv_data:
            if isinstance(d, dict) and 'iteration_shots' in d and isinstance(d['iteration_shots'], list):
                try:
                    total_shots += sum(int(x) for x in d['iteration_shots'])
                    total_iters += len(d['iteration_shots'])
                except:
                    pass
        try:
            from qiskit import transpile
            transpiled = transpile(base_ansatz, backend=backend_info.get('backend'), optimization_level=3) if backend_info.get('backend') else base_ansatz
            ops = transpiled.count_ops()
            twoq = int(ops.get('cx', 0)) + int(ops.get('cz', 0)) + int(ops.get('swap', 0))
            transpile_metrics = {
                "logical_qubits": int(base_ansatz.num_qubits),
                "physical_qubits": int(transpiled.num_qubits),
                "depth": int(transpiled.depth() or 0),
                "two_qubit_gates": twoq,
                "ops": {k: int(v) for k, v in ops.items()}
            }
        except Exception:
            transpile_metrics = {}
        try:
            per_run = []
            for d in all_conv_data:
                values = d.get('values', [])
                counts = d.get('counts', [])
                iters = len(values)
                start = float(values[0]) if iters > 0 else None
                end = float(values[-1]) if iters > 0 else None
                min_e = float(min(values)) if iters > 0 else None
                avg_drop = float((values[0] - values[-1]) / max(1, iters - 1)) if iters > 1 else 0.0
                per_run.append({
                    "iterations": iters,
                    "energy_start": start,
                    "energy_end": end,
                    "min_energy": min_e,
                    "avg_decrease_per_iter": avg_drop
                })
            best_end = None
            if best_restart_idx > 0:
                br = all_conv_data[best_restart_idx - 1]
                vals = br.get('values', [])
                best_end = float(vals[-1]) if len(vals) > 0 else None
            convergence_metrics = {
                "iterations_total": int(sum(len(d.get('values', [])) for d in all_conv_data)),
                "best_run_index": int(best_restart_idx),
                "best_energy": float(best_overall_energy),
                "energy_end_best_run": best_end,
                "per_run": per_run
            }
        except Exception:
            convergence_metrics = {}
        metrics = {
            "backend": args.backend,
            "shots_requested": int(args.shots),
            "shots_actual_total": int(total_shots),
            "iteration_count": int(total_iters),
            "outcome_summary": f"min_energy={float(best_overall_energy):.6f}",
            "qubits_used": int(base_ansatz.num_qubits),
            "qubits_full": int(problem._qubit_op_full().num_qubits),
            "transpile_metrics": transpile_metrics,
            "convergence_metrics": convergence_metrics
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ 指标摘要已保存到: {metrics_path}")
    except Exception:
        pass
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 蛋白质折叠模拟运行成功！")
    else:
        print("\n❌ 蛋白质折叠模拟运行失败。")
        sys.exit(1)
