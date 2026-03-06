"""
STFC 量子蛋白质侧链优化程序

该程序提供了三种优化蛋白质侧链构象的方法：
1. Exact（精确计算）：暴力搜索所有可能的位串，计算精确能量
2. QAOA（量子近似优化算法）：使用量子电路进行优化
3. SA（模拟退火）：经典优化方法作为对比

使用示例:
    # 精确计算
    python run_stfc_qf.py -res 2 -rot 2 -m exact
    
    # 模拟退火
    python run_stfc_qf.py -res 2 -rot 2 -m sa
    
    # QAOA（量子）- 本地模拟器
    python run_stfc_qf.py -res 2 -rot 2 -m qaoa -p 1 -s 100
    
    # QAOA（量子）- AWS 后端
    python run_stfc_qf.py -res 2 -rot 2 -m qaoa -p 1 -s 100 --backend aws_sv1

基于论文：
"Quantum Algorithm for Protein Side-Chain Optimisation: Comparing Quantum to Classical Methods"
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import warnings
from lib.job_metadata_logger import JobMetadataLogger
from lib.quantum_optimizer import QuantumTimeTracker

# 忽略 Scipy 稀疏矩阵效率警告（因 Qiskit 模拟器内部使用引起）
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning
    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass

# 尝试导入 AWS Braket Provider
try:
    from qiskit_braket_provider import BraketProvider
    BRAKET_AVAILABLE = True
except ImportError:
    BRAKET_AVAILABLE = False

# 添加 STFC-QF 模块到 Python 路径
stfc_qf_path = Path(__file__).parent / "stfc-qf" / "quantum-protein-folding"
sys.path.insert(0, str(stfc_qf_path))

# 导入 STFC-QF 核心模块
from quantum_protein_folding.quantum_protein_folding.ising import (
    get_hamiltonian,
    get_q_hamiltonian,
)
from quantum_protein_folding.quantum_protein_folding.bitstrings import (
    get_min_energy_bitstring,
    calculate_bitstring_energies,
    generate_bitstrings,
    bitstring_to_int,
)
from quantum_protein_folding.quantum_protein_folding.qaoa import (
    get_initial_parameters,
    get_symmetry_preserving_initial_state,
)
from quantum_protein_folding.quantum_protein_folding.config import (
    EXACT_ENERGY_DATA_DIR,
)

# 尝试导入 Qiskit（用于 QAOA）
QISKIT_AVAILABLE = False
QISKIT_VERSION = None

# 全局定义采样器类以避免 NameError
BackendSampler = None
StatevectorSampler = None
BackendSamplerV2 = None

try:
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
    
    # 尝试 Qiskit 2.x 的导入方式
    try:
        from qiskit.primitives import StatevectorSampler, BackendSamplerV2
        from qiskit.transpiler import PassManager
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit_algorithms.minimum_eigensolvers import QAOA
        QISKIT_VERSION = 2
        print(f"使用 Qiskit {QISKIT_VERSION}.x 版本 (BackendSamplerV2)")
    except ImportError as e:
        print(f"Qiskit 2.x 导入失败: {e}")
        # 回退到 Qiskit 1.x 的导入方式
        try:
            from qiskit.primitives import BackendSampler
            from qiskit.transpiler import PassManager
            from qiskit_algorithms.optimizers import COBYLA
            from qiskit_algorithms.minimum_eigensolvers import QAOA
            QISKIT_VERSION = 1
            print(f"使用 Qiskit {QISKIT_VERSION}.x 版本 (BackendSampler)")
        except ImportError as e:
            QISKIT_AVAILABLE = False
            QISKIT_VERSION = None
            print(f"Qiskit 导入完全失败: {e}")
            
except ImportError as e:
    QISKIT_AVAILABLE = False
    QISKIT_VERSION = None
    print(f"Warning: Qiskit import failed: {e}")

# 尝试导入 SciPy（用于模拟退火）
try:
    from scipy.optimize import dual_annealing
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class GroundStateTracker:
    """基态追踪器类，用于追踪 QAOA 优化过程中是否找到基态"""

    def __init__(self, int_ground_state: int, args: Any, qiskit_version: int = 1):
        self.shots_to_ground_state = 0
        self.iterations = 0
        self.int_ground_state = int_ground_state
        self.num_shots = 0
        self.gs_found = False
        self.args = args
        self.qiskit_version = qiskit_version
        self.energy_history = []
        self.iteration_history = []
        self.circuit_info = {
            'total_gates': [],
            'single_qubit_gates': [],
            'two_qubit_gates': [],
            'circuit_depth': [],
            'shots': [],
            'cumulative_shots': [],
            'timing': []
        }
        self.timing_history = []

    def callback(self, *args, **kwargs):
        # 调试信息：打印 callback 接收到的所有参数
        print(f"\n[CALLBACK DEBUG] 收到 {len(args)} 个位置参数，{len(kwargs)} 个关键字参数")
        for i, arg in enumerate(args):
            try:
                print(f"  args[{i}] = {type(arg).__name__}: {arg}")
            except Exception as e:
                print(f"  args[{i}] = {type(arg).__name__} (无法打印：{e})")
        
        if self.qiskit_version == 2:
            # Qiskit 2.x callback 可能有不同的参数格式
            # 尝试多种可能的格式
            
            # 格式 1: callback(intermediate_result, optimizer_state, f_val, metadata)
            # 格式 2: callback(x_current, f_val_current, *metadata)
            # 格式 3: callback(opt_result, *args)
            
            energy_recorded = False
            iteration_incremented = False
            
            # 尝试从不同位置提取能量值
            for idx in [2, 1, 0]:
                if len(args) > idx:
                    try:
                        energy_val = float(args[idx])
                        self.energy_history.append(energy_val)
                        print(f"  [CALLBACK] 从 args[{idx}] 记录能量：{energy_val}")
                        energy_recorded = True
                        break
                    except (TypeError, ValueError):
                        continue
            
            if not energy_recorded:
                print(f"  [CALLBACK] 未能从任何位置提取能量值")
            
            # 检查 metadata（通常在最后一个参数）
            if len(args) >= 4:
                metadata = args[3]
            elif len(args) >= 2 and isinstance(args[-1], dict):
                metadata = args[-1]
            else:
                metadata = None
            
            if metadata is not None and isinstance(metadata, dict):
                if "best_measurements" in metadata:
                    best_measurements = metadata["best_measurements"]
                    
                    for measurement in best_measurements:
                        if isinstance(measurement, dict) and measurement.get("state") == self.int_ground_state:
                            self.shots_to_ground_state = self.num_shots
                            self.gs_found = True
                            break
                    
                    # 无论是否找到基态，都增加迭代计数
                    self.iterations += 1
                    self.num_shots += self.args.shots
                    iteration_incremented = True
                    print(f"  [CALLBACK] iterations={self.iterations}, gs_found={self.gs_found}")
                    
                    # 记录电路信息
                    self.circuit_info['shots'].append(self.args.shots)
                    self.circuit_info['cumulative_shots'].append(self.num_shots)
                    
                    # 记录时间分解
                    if self.timing_history:
                        self.circuit_info['timing'].append(self.timing_history[-1])
                else:
                    # metadata 存在但没有 best_measurements
                    self.iterations += 1
                    self.num_shots += self.args.shots
                    iteration_incremented = True
                    print(f"  [CALLBACK] metadata 无 best_measurements, iterations={self.iterations}")
            
            # 如果没有 metadata，但有至少 2 个参数，仍然增加迭代
            if not iteration_incremented and len(args) >= 2:
                self.iterations += 1
                self.num_shots += self.args.shots
                print(f"  [CALLBACK] 无 metadata, iterations={self.iterations}")
                
        else:
            # Qiskit 1.x 的处理逻辑
            if len(args) >= 1:
                all_bitstrings = args[0]
                if hasattr(all_bitstrings, '__iter__') and self.int_ground_state in all_bitstrings:
                    self.shots_to_ground_state = self.num_shots
                    self.gs_found = True
                else:
                    self.num_shots += self.args.shots
                    self.iterations += 1


class CallTracker:
    """调用追踪器类，用于模拟退火"""
    
    def __init__(self):
        self.calls = 0


class FoundGroundState(Exception):
    """找到基态异常，用于提前终止模拟退火"""
    pass


def check_energy_file_exists(num_res: int, num_rot: int) -> bool:
    """检查精确能量数据文件是否存在"""
    exact_energy_filepath = f"{EXACT_ENERGY_DATA_DIR}/res-{num_res}-rot-{num_rot}.json"
    if not os.path.exists(exact_energy_filepath):
        print(f"Error: Exact energy data file not found: {exact_energy_filepath}")
        print("Please run create_energy_files.py first to generate required data.")
        return False
    return True


def get_ground_state_info(num_res: int, num_rot: int) -> Optional[str]:
    """获取基态位串信息"""
    exact_energy_filepath = f"{EXACT_ENERGY_DATA_DIR}/res-{num_res}-rot-{num_rot}.json"
    if not os.path.exists(exact_energy_filepath):
        return None
    return get_min_energy_bitstring(filepath=exact_energy_filepath)


def create_output_directory(output_dir: str) -> Path:
    """创建输出目录"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_csv_result(csv_path: Path, result_dict: Dict[str, Any]) -> None:
    """保存结果到 CSV 文件"""
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result_dict.keys())
        writer.writeheader()
        writer.writerow(result_dict)


class TimingSamplerWrapper:
    """采样器包装类，用于拦截 run 调用并记录时间信息"""
    def __init__(self, base_sampler, tracker_list, backend_name):
        self._base_sampler = base_sampler
        self._tracker_list = tracker_list
        self._backend_name = backend_name

    def run(self, pubs, **kwargs):
        submit_time = time.perf_counter()
        job = self._base_sampler.run(pubs, **kwargs)
        
        # 记录原始 result 方法
        original_result = job.result
        
        def wrapped_result(*args, **kwargs_inner):
            res = original_result(*args, **kwargs_inner)
            received_time = time.perf_counter()
            
            timing = QuantumTimeTracker()
            timing.submit_time = submit_time
            timing.result_received = received_time
            timing.iteration_start = submit_time
            timing.iteration_end = received_time
            timing.extract_from_job(job, self._backend_name)
            
            self._tracker_list.append(timing.to_dict())
            
            # 在终端输出时间分解
            last_timing = self._tracker_list[-1]
            print(
                f"    [TIMING] Quantum={last_timing['quantum_time']:.3f}s, "
                f"Queue={last_timing['queue_time']:.3f}s, "
                f"Classical={last_timing['classical_time']:.3f}s, "
                f"Total={last_timing['total_time']:.3f}s"
            )
            
            return res
            
        # 动态替换方法
        job.result = wrapped_result
        return job


def create_sampler(backend: str, simulator: str, shots: int, tracker_list=None):
    """创建采样器"""
    # 检查 Qiskit 是否可用
    if not QISKIT_AVAILABLE:
        print("错误: Qiskit 不可用，无法创建采样器。请安装 Qiskit:")
        print("  pip install qiskit")
        return None
    
    if backend == "local":
        backend_method = "matrix_product_state" if simulator == "MPS" else "statevector"
        
        if QISKIT_VERSION == 2:
            if StatevectorSampler is None:
                print("错误: StatevectorSampler 不可用")
                return None
            sampler = StatevectorSampler(default_shots=shots, seed=42)
        else:
            if BackendSampler is None:
                print("错误: BackendSampler 不可用")
                return None
            simulator_obj = AerSimulator(method=backend_method)
            backend_options = {
                "seed_simulator": 42,
                "shots": shots,
                "optimization_level": 3,
                "resilience_level": 3,
            }
            sampler = BackendSampler(
                backend=simulator_obj, options=backend_options, bound_pass_manager=PassManager()
            )
        
        if tracker_list is not None:
            return TimingSamplerWrapper(sampler, tracker_list, backend)
        return sampler
    else:
        if not BRAKET_AVAILABLE:
            print("错误: qiskit-braket-provider 未安装。请安装:")
            print("  pip install qiskit-braket-provider")
            print("并配置 AWS 凭据。")
            return None
        
        # 根据 Qiskit 版本选择合适的采样器
        if QISKIT_VERSION == 2:
            if BackendSamplerV2 is None:
                print("错误: BackendSamplerV2 不可用")
                return None
        else:
            if BackendSampler is None:
                print("错误: BackendSampler 不可用")
                return None
        
        provider = BraketProvider()
        backend_names = {
            "aws_sv1": "SV1",
            "aws_garnet": "Garnet",
            "aws_forte": "Forte"
        }
        
        if backend not in backend_names:
            print(f"错误: 未知的后端 {backend}")
            return None
        
        backend_obj = provider.get_backend(backend_names[backend])
        
        if QISKIT_VERSION == 2:
            # Qiskit 2.x 使用不同的选项格式
            backend_options = {"default_shots": shots}
            base_sampler = BackendSamplerV2(backend=backend_obj, options=backend_options)
            # 使用 TranspilingSampler 包装以避免 Braket 后端的重复测量问题
            import sys
            sys.path.insert(0, '/home/ubuntu/workspace/quantum/qthesis-pf/src')
            from backend.transpiling_sampler import TranspilingSampler
            sampler = TranspilingSampler(base_sampler, backend_obj)
        else:
            # Qiskit 1.x 使用原来的选项格式
            backend_options = {"shots": shots}
            sampler = BackendSampler(
                backend=backend_obj, options=backend_options, bound_pass_manager=PassManager()
            )
        
        if tracker_list is not None:
            return TimingSamplerWrapper(sampler, tracker_list, backend)
        return sampler


def parse_args():
    """命令行参数解析器"""
    parser = argparse.ArgumentParser(
        prog="run_stfc_qf",
        description="Quantum Protein Side-Chain Optimisation - STFC-QF"
    )
    parser.add_argument(
        "-res", "--num_res",
        type=int,
        default=2,
        help="Number of residues (amino acids) - 残基数量（氨基酸数）"
    )
    parser.add_argument(
        "-rot", "--num_rot",
        type=int,
        default=2,
        help="Number of rotamers per residue - 每个残基的旋转异构体数"
    )
    parser.add_argument(
        "-m", "--method",
        type=str,
        choices=["qaoa", "sa", "exact", "all"],
        default="qaoa",
        help="Optimization method: qaoa (quantum), sa (simulated annealing), exact (brute force), all (run all methods) - 优化方法"
    )
    parser.add_argument(
        "-p", "--layers",
        type=int,
        default=1,
        help="Number of ansatz layers for QAOA (default: 1) - QAOA 拟设层数"
    )
    parser.add_argument(
        "-alpha", "--alpha",
        type=float,
        default=0.02,
        help="Alpha parameter for CVaR aggregation (default: 0.02) - CVaR 聚合参数"
    )
    parser.add_argument(
        "-s", "--shots",
        type=int,
        default=1000,
        help="Number of shots for quantum execution (default: 1000) - 量子执行测量次数"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["local", "aws_sv1", "aws_garnet", "aws_forte"],
        default="local",
        help="Quantum backend: local (aer simulator), aws_sv1, aws_garnet, aws_forte (default: local) - 量子后端"
    )
    parser.add_argument(
        "--simulator",
        type=str,
        choices=["MPS", "SV"],
        default="MPS",
        help="Simulation method for local backend: MPS (matrix-product-state) or SV (statevector) (default: MPS) - 模拟器类型"
    )
    parser.add_argument(
        "--out", "--output_dir",
        type=str,
        default="./results",
        help="Output directory (default: ./results) - 输出目录"
    )
    return parser.parse_args()


def run_exact(num_res: int, num_rot: int, output_dir: str) -> Optional[Dict[str, Any]]:
    """运行精确能量计算（暴力搜索）"""
    print(f"\n=== Running Exact Energy Calculation ===")
    print(f"Residues: {num_res}, Rotamers: {num_rot}")
    
    if not check_energy_file_exists(num_res, num_rot):
        return None
    
    ground_state_bitstring = get_ground_state_info(num_res, num_rot)
    if ground_state_bitstring is None:
        return None
    
    print(f"Ground state bitstring: {ground_state_bitstring}")
    
    bitstrings = generate_bitstrings(num_res, num_rot)
    print(f"Total bitstrings: {len(bitstrings)}")
    
    hamiltonian = get_hamiltonian(num_rot=num_rot, num_res=num_res)
    q_hamiltonian = get_q_hamiltonian(num_res * num_rot, hamiltonian)
    energies = calculate_bitstring_energies(bitstrings, q_hamiltonian)
    
    output_path = create_output_directory(output_dir)
    csv_path = output_path / f"exact_res{num_res}_rot{num_rot}.csv"
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bitstring", "energy"])
        for bitstring in bitstrings:
            writer.writerow([bitstring, energies[bitstring]])
    
    print(f"Results saved to: {csv_path}")
    
    plot_energy_distribution(bitstrings, energies, num_res, num_rot, output_path)
    
    min_energy = min(energies.values())
    return {
        "ground_state": ground_state_bitstring,
        "optimal_eigenvalue": min_energy
    }


def run_qaoa(
    num_res: int,
    num_rot: int,
    layers: int,
    alpha: float,
    shots: int,
    simulator: str,
    output_dir: str,
    backend: str = "local"
) -> Optional[Dict[str, Any]]:
    """运行 QAOA 优化"""
    if not QISKIT_AVAILABLE:
        print("Error: Qiskit is not installed. Please install qiskit, qiskit-aer, and qiskit-algorithms.")
        return None
    
    print(f"\n=== Running QAOA ===")
    print(f"Residues: {num_res}, Rotamers: {num_rot}")
    print(f"Layers (p): {layers}, Alpha: {alpha}, Shots: {shots}")
    print(f"Backend: {backend}")
    if backend == "local":
        print(f"Simulator: {simulator}")
    
    if not check_energy_file_exists(num_res, num_rot):
        return None
    
    bitstring_ground_state = get_ground_state_info(num_res, num_rot)
    if bitstring_ground_state is None:
        return None
    
    int_ground_state = bitstring_to_int(bitstring_ground_state)
    print(f"Ground state bitstring: {bitstring_ground_state}")
    
    hamiltonian = get_hamiltonian(num_rot=num_rot, num_res=num_res)
    q_hamiltonian = get_q_hamiltonian(num_res * num_rot, hamiltonian)
    
    from quantum_protein_folding.quantum_protein_folding.ising import get_xy_mixer
    mixer = get_xy_mixer(num_res=num_res, num_rot=num_rot)
    
    initial_point = get_initial_parameters(layers, cost_bound=0.1, mixer_bound=1.0)
    initial_state = get_symmetry_preserving_initial_state(num_res=num_res, num_rot=num_rot)
    
    class Args:
        def __init__(self, shots):
            self.shots = shots

    tracker = GroundStateTracker(int_ground_state, Args(shots), qiskit_version=QISKIT_VERSION)
    
    # 创建采样器 - 根据后端类型决定是否使用包装器
    if backend == "local":
        # 本地后端：使用包装器来跟踪时间
        sampler = create_sampler(backend, simulator, shots, tracker_list=tracker.timing_history)
    else:
        # AWS 后端：使用 TranspilingSampler 避免重复测量问题，同时使用 TimingSamplerWrapper 跟踪时间
        import sys
        sys.path.insert(0, '/home/ubuntu/workspace/quantum/qthesis-pf/src')
        from backend.transpiling_sampler import TranspilingSampler
        
        # 先创建基础 sampler（不带 tracker）
        base_sampler = create_sampler(backend, simulator, shots)
        
        # 获取 backend 对象用于 TranspilingSampler
        if not BRAKET_AVAILABLE:
            print("错误：qiskit-braket-provider 未安装")
            return None
        
        from qiskit_braket_provider import BraketProvider
        provider = BraketProvider()
        backend_names = {"aws_sv1": "SV1", "aws_garnet": "Garnet", "aws_forte": "Forte"}
        backend_obj = provider.get_backend(backend_names.get(backend, "SV1"))
        
        # 用 TranspilingSampler 包装以解决重复测量问题
        transpiled_sampler = TranspilingSampler(base_sampler, backend_obj)
        
        # 再用 TimingSamplerWrapper 包装以跟踪时间
        sampler = TimingSamplerWrapper(transpiled_sampler, tracker.timing_history, backend)
    if sampler is None:
        return None
    
    # 创建 QAOA 实例
    qaoa = QAOA(
        sampler=sampler,
        optimizer=COBYLA(),
        reps=layers,
        initial_state=initial_state,
        mixer=mixer,
        initial_point=initial_point,
        callback=tracker.callback,
        aggregation=alpha,
    )
    qaoa.optimizer.set_options(maxiter=10_000)
        
    # 对于 AWS 后端，直接在量子计算机上进行完整的参数优化
    if backend != "local":
        print("在量子后端上执行完整的 QAOA 优化...")
        print(f"优化器：COBYLA, 最大迭代次数：10,000")
        print(f"每次迭代都将在 {backend} 上执行量子电路\n")
        
        # 直接使用 AWS 量子后端进行完整的参数优化
        print("开始量子优化过程...")
        
        # 使用量子后端直接优化参数（每次迭代都在量子计算机上执行）
        result = qaoa.compute_minimum_eigenvalue(q_hamiltonian)
        
        print(f"\n✓ 量子优化完成！")
        print(f"最优能量：{result.eigenvalue.real:.6f}")
        print(f"最优比特串：{result.best_measurement['bitstring']}")
        print(f"找到基态：{'是' if tracker.gs_found else '否'}")
        print(f"总迭代次数：{tracker.iterations}")
        print(f"能量历史长度：{len(tracker.energy_history)}")
        print(f"时间记录长度：{len(tracker.timing_history)}")
        
        # 返回结果和 timing_history，以及迭代信息
        return result, tracker.timing_history, tracker.iterations, tracker.energy_history
    
    # 本地后端的处理逻辑保持不变
    print("在本地模拟器上执行 QAOA 优化...")
    result = qaoa.compute_minimum_eigenvalue(q_hamiltonian)
    
    # 打印调试信息
    print(f"\n✓ 量子优化完成！")
    print(f"最优能量：{result.eigenvalue.real:.6f}")
    if hasattr(result, 'best_measurement') and result.best_measurement:
        print(f"最优比特串：{result.best_measurement.get('bitstring', 'N/A')}")
    print(f"找到基态：{'是' if tracker.gs_found else '否'}")
    print(f"总迭代次数：{tracker.iterations}")
    print(f"能量历史长度：{len(tracker.energy_history)}")
    print(f"时间记录长度：{len(tracker.timing_history)}")
    
    # 返回结果和 timing_history，以及迭代信息
    return result, tracker.timing_history, tracker.iterations, tracker.energy_history


def run_simulated_annealing(
    num_res: int,
    num_rot: int,
    output_dir: str
) -> Optional[Dict[str, Any]]:
    """运行模拟退火优化"""
    if not SCIPY_AVAILABLE:
        print("Error: SciPy is not installed. Please install scipy.")
        return None
    
    print(f"\n=== Running Simulated Annealing ===")
    print(f"Residues: {num_res}, Rotamers: {num_rot}")
    
    if not check_energy_file_exists(num_res, num_rot):
        return None
    
    ground_state_bitstring = get_ground_state_info(num_res, num_rot)
    if ground_state_bitstring is None:
        return None
    
    print(f"Ground state bitstring: {ground_state_bitstring}")
    
    hamiltonian = get_hamiltonian(num_rot=num_rot, num_res=num_res)
    num_bits = num_rot * num_res
    q_hamiltonian = get_q_hamiltonian(num_qubits=num_bits, h_matrix=hamiltonian)
    
    tracker = CallTracker()
    
    def objective_function(x):
        binary_state = [1 if xi > 0.5 else 0 for xi in x]
        binary_state = "".join(str(digit) for digit in binary_state)
        tracker.calls += 1
        energy = calculate_bitstring_energies(
            bitstrings=[binary_state], hamiltonian=q_hamiltonian
        )[binary_state]
        
        if binary_state == ground_state_bitstring:
            raise FoundGroundState()
        
        return energy
    
    print("Running simulated annealing...")
    try:
        result = dual_annealing(
            func=objective_function,
            x0=np.random.rand(num_bits),
            bounds=[(0.0, 1.0)] * num_bits,
            maxiter=1000,
            initial_temp=1.0,
            visit=2.62,
            accept=-5.0,
            no_local_search=True,
        )
    except FoundGroundState:
        print("Ground state found!")
        result = None
    
    output_path = create_output_directory(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_path / f"sa_res{num_res}_rot{num_rot}_{timestamp}.csv"
    
    result_dict = {
        "num_res": num_res,
        "num_rot": num_rot,
        "ground_state": ground_state_bitstring,
        "gs_found": tracker.calls > 0,
        "calls": tracker.calls,
        "total_complexity": tracker.calls * num_bits,
    }
    
    save_csv_result(csv_path, result_dict)
    print(f"Results saved to: {csv_path}")
    
    try:
        iterations_dir = output_path / "iterations"
        os.makedirs(iterations_dir, exist_ok=True)
        import json as _json
        # 如果 dual_annealing 未抛 FoundGroundState，无法收集能量轨迹，按调用次数记录空能量
        total_iters = int(tracker.calls)
        energies = getattr(tracker, "energy_history", [])
        for idx in range(1, total_iters + 1):
            energy_val = float(energies[idx - 1]) if idx - 1 < len(energies) else None
            payload = {
                "index": idx,
                "eval_count": idx,
                "energy": energy_val,
                "shots": 0,
                "backend": "classical",
                "optimization_convergence": {
                    "evaluation_counts": list(range(1, idx + 1)),
                    "energy_values": [float(e) for e in energies[:idx]] if energies else [],
                    "cumulative_shots": [0] * idx,
                    "iteration_shots": [0] * idx
                },
                "protein_structure": {
                    "turn_sequence": [],
                    "xyz_coordinates": []
                },
                "timestamp": datetime.now().isoformat()
            }
            with (iterations_dir / f"iteration_{idx}_result.json").open("w", encoding="utf-8") as f:
                _json.dump(payload, f, indent=2)
    except Exception:
        pass
    
    return result_dict


def plot_energy_convergence(
    iteration_history: list,
    energy_history: list,
    num_res: int,
    num_rot: int,
    layers: int,
    output_dir: Path
) -> None:
    """绘制 QAOA 能量收敛图"""
    output_dir = Path(output_dir)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(iteration_history, energy_history, 'b-', linewidth=2, marker='o', markersize=4, label='Energy')
    
    min_energy_idx = np.argmin(energy_history)
    ax.plot(iteration_history[min_energy_idx], energy_history[min_energy_idx], 
            'r*', markersize=15, label=f'Min Energy: {energy_history[min_energy_idx]:.6f}')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Energy', fontsize=12)
    ax.set_title(f'QAOA Energy Convergence\nResidues: {num_res}, Rotamers: {num_rot}, Layers: {layers}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f"energy_convergence_res{num_res}_rot{num_rot}_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Energy convergence plot saved to: {plot_path}")


def plot_energy_distribution(
    bitstrings: list,
    energies: dict,
    num_res: int,
    num_rot: int,
    output_dir: Path
) -> None:
    """绘制能量分布图"""
    output_dir = Path(output_dir)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    energy_values = [energies[bs] for bs in bitstrings]
    
    n, bins, patches = ax.hist(energy_values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    
    min_energy = min(energy_values)
    ax.axvline(x=min_energy, color='red', linestyle='--', linewidth=2, 
                label=f'Ground State Energy: {min_energy:.6f}')
    
    ax.set_xlabel('Energy', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Energy Distribution\nResidues: {num_res}, Rotamers: {num_rot}, Total States: {len(bitstrings)}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f"energy_distribution_res{num_res}_rot{num_rot}_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Energy distribution plot saved to: {plot_path}")


def plot_comparison(
    results: Dict[str, Any],
    num_res: int,
    num_rot: int,
    output_dir: Path
) -> None:
    """绘制优化结果对比图"""
    output_dir = Path(output_dir)
    
    methods = []
    energies = []
    labels = []
    
    if 'exact' in results:
        exact_energy = results['exact'].get('optimal_eigenvalue', 0)
        if exact_energy == 0:
            exact_csv = output_dir / f"exact_res{num_res}_rot{num_rot}.csv"
            if exact_csv.exists():
                df = pd.read_csv(exact_csv)
                exact_energy = df['energy'].min()
        methods.append('Exact')
        energies.append(exact_energy)
        labels.append(f'Exact\n{exact_energy:.6f}')
    
    if 'qaoa' in results:
        # 处理 SamplingVQEResult 对象
        qaoa_result = results['qaoa']
        if hasattr(qaoa_result, 'eigenvalue'):
            qaoa_energy = float(qaoa_result.eigenvalue.real) if hasattr(qaoa_result.eigenvalue, 'real') else float(qaoa_result.eigenvalue)
            gs_found = False  # 暂时设为 False，实际值在 tracker 中
        elif isinstance(qaoa_result, dict):
            qaoa_energy = qaoa_result.get('optimal_eigenvalue', 0)
            gs_found = qaoa_result.get('gs_found', False)
        else:
            qaoa_energy = 0
            gs_found = False
        methods.append('QAOA')
        energies.append(qaoa_energy)
        labels.append(f'QAOA\n{qaoa_energy:.6f}\n(GS found: {gs_found})')
    
    if 'sa' in results:
        methods.append('SA')
        if 'exact' in results:
            exact_energy = results['exact'].get('optimal_eigenvalue', 0)
            if exact_energy == 0:
                exact_csv = output_dir / f"exact_res{num_res}_rot{num_rot}.csv"
                if exact_csv.exists():
                    df = pd.read_csv(exact_csv)
                    exact_energy = df['energy'].min()
            energies.append(exact_energy)
            gs_found = results['sa'].get('gs_found', False)
            labels.append(f'SA\n{exact_energy:.6f}\n(GS found: {gs_found})')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax.bar(range(len(methods)), energies, color=colors[:len(methods)], 
                 alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, fontsize=11)
    
    for i, (bar, energy) in enumerate(zip(bars, energies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{energy:.6f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Energy', fontsize=12)
    ax.set_title(f'Optimization Methods Comparison\nResidues: {num_res}, Rotamers: {num_rot}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f"methods_comparison_res{num_res}_rot{num_rot}_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Methods comparison plot saved to: {plot_path}")


metadata_logger = JobMetadataLogger("protein_folding_jobs_detailed.csv")

@metadata_logger
def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 60)
    print("STFC Quantum Protein Side-Chain Optimisation")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_name = "all" if args.method == "all" else args.method
    output_dir = Path(args.out) / f"{timestamp}_{method_name}_stfc_qf"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    results = {}
    
    if args.method == "exact" or args.method == "all":
        result = run_exact(args.num_res, args.num_rot, str(output_dir))
        results['exact'] = result
    
    if args.method == "qaoa" or args.method == "all":
        qaoa_result = run_qaoa(
            args.num_res, args.num_rot, args.layers, args.alpha,
            args.shots, args.simulator, str(output_dir), args.backend
        )
        # 解析返回值
        if isinstance(qaoa_result, tuple) and len(qaoa_result) == 4:
            result, timing_history, iterations, energy_history = qaoa_result
            results['timing_history'] = timing_history
            results['iterations'] = iterations
            results['energy_history'] = energy_history
            print(f"\n[DEBUG] 主函数接收到的数据:")
            print(f"  - iterations: {iterations}")
            print(f"  - energy_history 长度：{len(energy_history)}")
            print(f"  - timing_history 长度：{len(timing_history)}")
        elif isinstance(qaoa_result, tuple) and len(qaoa_result) == 2:
            result, timing_history = qaoa_result
            results['timing_history'] = timing_history
            results['iterations'] = 0
            results['energy_history'] = []
        else:
            result = qaoa_result
            results['iterations'] = 0
            results['energy_history'] = []
        results['qaoa'] = result
    
    if args.method == "sa" or args.method == "all":
        result = run_simulated_annealing(args.num_res, args.num_rot, str(output_dir))
        results['sa'] = result
    
    if len(results) > 1:
        plot_comparison(results, args.num_res, args.num_rot, output_dir)
    
    print("\n" + "=" * 60)
    print("Execution completed!")
    print("=" * 60)
    
    try:
        import json
        metrics_path = output_dir / "metrics.json"
        qubits_used = int(args.num_res * args.num_rot)
        shots_req = int(args.shots) if (args.method == "qaoa" or args.method == "all") else 0
        
        # 从结果中汇总数据
        qaoa_res = results.get('qaoa', {})
        
        # 处理结果对象 - 支持 SamplingVQEResult 和字典两种格式
        if hasattr(qaoa_res, 'eigenvalue'):
            # SamplingVQEResult 对象
            opt_history = getattr(qaoa_res, 'optimization_history', {})
            
            # 优先使用 tracker 中记录的实际迭代数和能量历史
            iterations = results.get('iterations', 0)
            energy_values = results.get('energy_history', [])
            
            print(f"\n[DEBUG] Metrics 生成前的数据:")
            print(f"  - iterations from results: {iterations}")
            print(f"  - energy_history from results: {len(energy_values)} items")
            
            # 如果 tracker 没有记录，则尝试从 optimization_history 提取
            if not energy_values and not iterations:
                if isinstance(opt_history, dict):
                    energy_values = opt_history.get('values', [])
                elif hasattr(opt_history, '__len__'):
                    energy_values = list(opt_history)
                
                # 如果 optimization_history 为空，才使用 timing_history 的长度
                if not energy_values and results.get('timing_history'):
                    energy_values = [0] * len(results['timing_history'])
                    iterations = len(results['timing_history'])
            
            # 从 results 中提取时间信息（如果有）
            timing_data = results.get('timing_history', [])
            total_quantum = sum(t.get('quantum_time', 0.0) for t in timing_data)
            total_queue = sum(t.get('queue_time', 0.0) for t in timing_data)
            total_classical = sum(t.get('classical_time', 0.0) for t in timing_data)
            
            qaoa_dict = {
                'iterations': iterations,
                'optimal_eigenvalue': float(qaoa_res.eigenvalue.real) if hasattr(qaoa_res.eigenvalue, 'real') else float(qaoa_res.eigenvalue),
                'total_quantum_time': total_quantum,
                'total_queue_time': total_queue,
                'total_classical_time': total_classical,
                'total_gates': 0,
                'single_qubit_gates': 0,
                'two_qubit_gates': 0,
                'circuit_depth': 0,
                'shots': shots_req
            }
        elif isinstance(qaoa_res, dict):
            qaoa_dict = qaoa_res
        else:
            qaoa_dict = {}
        
        shots_actual = qaoa_dict.get('shots', 0) * qaoa_dict.get('iterations', 0)
        
        metrics = {
            "backend": args.backend,
            "shots_requested": shots_req,
            "shots_actual_total": shots_actual,
            "iteration_count": qaoa_dict.get('iterations', 0),
            "outcome_summary": str(qaoa_dict.get('optimal_eigenvalue', "")),
            "qubits_used": qubits_used,
            "qubits_full": qubits_used,
            "total_quantum_time": round(qaoa_dict.get('total_quantum_time', 0.0), 3),
            "total_queue_time": round(qaoa_dict.get('total_queue_time', 0.0), 3),
            "total_classical_time": round(qaoa_dict.get('total_classical_time', 0.0), 3),
            "total_gates": qaoa_dict.get('total_gates', 0),
            "single_qubit_gates": qaoa_dict.get('single_qubit_gates', 0),
            "two_qubit_gates": qaoa_dict.get('two_qubit_gates', 0),
            "circuit_depth": qaoa_dict.get('circuit_depth', 0),
            "transpile_metrics": {
                "total_gates": qaoa_dict.get('total_gates', 0),
                "single_qubit_gates": qaoa_dict.get('single_qubit_gates', 0),
                "two_qubit_gates": qaoa_dict.get('two_qubit_gates', 0),
                "circuit_depth": qaoa_dict.get('circuit_depth', 0)
            },
            "convergence_metrics": {}
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")
        if qaoa_res:
            print(f"\n[DONE] Timing Summary:")
            print(f"  - Total Iterations: {metrics['iteration_count']}")
            print(f"  - Total Quantum Time: {metrics['total_quantum_time']:.3f}s")
            print(f"  - Total Queue Time: {metrics['total_queue_time']:.3f}s")
            print(f"  - Total Classical Time: {metrics['total_classical_time']:.3f}s")
            # 计算总时间（由 metrics 中的三个部分累加）
            total_duration = metrics['total_quantum_time'] + metrics['total_queue_time'] + metrics['total_classical_time']
            print(f"  - Total Execution Time: {total_duration:.3f}s")
    except Exception as e:
        print(f"Saving metrics failed: {e}")


if __name__ == "__main__":
    main()
