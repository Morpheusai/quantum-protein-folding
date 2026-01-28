#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
蛋白质折叠量子算法 - 采样器模式 (Sampler Mode)

功能特点：
1. 支持本地模拟器和AWS Braket后端
2. 使用CVaR (Conditional Value at Risk) 优化策略
3. 基于采样的能量计算方法
4. 支持多轮独立实验及结果汇总分析
"""

import argparse
import os
import sys
import warnings
import datetime
import json
import numpy as np
import copy
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RealAmplitudes
from qiskit_braket_provider import BraketProvider
from scipy.optimize import minimize
import traceback
from job_metadata_logger import JobMetadataLogger

# 创建元数据记录器实例
metadata_logger = JobMetadataLogger("protein_folding_jobs.csv")

# 环境配置
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.name == 'nt':
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, os.path.join(current_dir, 'src'))

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ====================
# 参数配置
# ====================
parser = argparse.ArgumentParser(description='蛋白质折叠量子算法 - 采样器模式')
parser.add_argument('--backend', default='local', help='量子后端选择: local (本地模拟器，优先使用AerSimulator), local_aer (强制使用AerSimulator), aws_sv1 (AWS模拟器), aws_garnet (AWS量子芯片), aws_ionq (AWS IonQ量子设备), aws_forte (AWS IonQ Forte量子设备), ibm (IBM量子设备), ibm_simulator (IBM模拟器)')
parser.add_argument('--random_seed', type=int, default=23, help='随机种子，用于确保结果可重现')
parser.add_argument('--max_optimization_iterations', type=int, default=10, help='最大优化迭代次数')
parser.add_argument('--ansatz_reps', type=int, default=1, help='变分量子线路的重复层数')
parser.add_argument('--main_chain', default='APRLRFY', help='蛋白质主链氨基酸序列')
parser.add_argument('--penalty_back', type=float, default=10, help='几何约束惩罚系数')
parser.add_argument('--penalty_chiral', type=float, default=10, help='手性约束惩罚系数')
parser.add_argument('--penalty_local_overlap', type=float, default=10, help='局部重叠惩罚系数')
parser.add_argument('--alpha', type=float, default=0.1, help='CVaR参数: 选择最低能量的alpha比例样本')
parser.add_argument('--shots', type=int, default=100, help='量子测量采样次数')
parser.add_argument('--max_results', type=int, default=1, help='每次迭代的结果数量')
parser.add_argument('--aws_region', default=None, help='AWS区域设置（可选）')
args = parser.parse_args()

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = os.path.join("results", f"{TIMESTAMP}_{args.backend}_sampler")
os.makedirs(RESULT_DIR, exist_ok=True)

# ====================
# 量子后端配置与运行逻辑
# ====================

def setup_sampler_backend(backend_name, aws_region=None, shots=1000):
    """配置支持采样器模式的量子后端，兼容多种量子云服务"""
    try:
        if backend_name.lower() == 'aws_sv1':
            from qiskit_braket_provider import BraketProvider
            if aws_region: os.environ['AWS_DEFAULT_REGION'] = aws_region
            provider = BraketProvider()
            backend = provider.get_backend('SV1')
            print(f"✓ AWS SV1 已连接")
            return backend
        elif backend_name.lower() == 'aws_garnet':
            from qiskit_braket_provider import BraketProvider
            if aws_region: os.environ['AWS_DEFAULT_REGION'] = aws_region
            provider = BraketProvider()
            backend = provider.get_backend('Garnet')
            print(f"✓ AWS Garnet 已连接")
            return backend
        elif backend_name.lower() == 'aws_ionq':
            from qiskit_braket_provider import BraketProvider
            if aws_region: os.environ['AWS_DEFAULT_REGION'] = aws_region
            provider = BraketProvider()
            # 获取 IonQ Harmony (11 qubits)
            backend = provider.get_backend('IonQ Device')
            print(f"✓ AWS IonQ Harmony 已连接")
            return backend
        elif backend_name.lower() == 'aws_forte':
            from qiskit_braket_provider import BraketProvider
            if aws_region: os.environ['AWS_DEFAULT_REGION'] = aws_region
            provider = BraketProvider()
            # 获取 IonQ Forte-1 (30+ qubits)
            backend = provider.get_backend('Forte 1')
            print(f"✓ AWS IonQ Forte-1 已连接")
            return backend
        elif backend_name.lower() == 'ibm':
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService
                service = QiskitRuntimeService()
                # 使用性能最好的可用设备
                backend = service.least_busy(operational=True, simulator=False)
                print(f"✓ IBM 量子后端已连接: {backend.name}")
                return backend
            except Exception as e:
                print(f"✗ IBM 真实硬件连接失败: {e}")
                print("  提示: 请确保已通过 'qiskit-ibm-runtime' 配置 IBM Quantum 访问凭据")
                # 回退到IBM模拟器
                try:
                    from qiskit_ibm_runtime import QiskitRuntimeService
                    service = QiskitRuntimeService()
                    # 使用IBM模拟器
                    backend = service.backend("ibmq_qasm_simulator")
                    print(f"✓ IBM 模拟器已连接: {backend.name}")
                    return backend
                except Exception as sim_e:
                    print(f"✗ IBM 模拟器连接也失败: {sim_e}")
                    # 最终回退到本地模拟器
                    from qiskit.providers.basic_provider import BasicSimulator
                    backend = BasicSimulator()
                    print("  ✓ 已回退到本地 BasicSimulator")
                    return backend
        elif backend_name.lower() == 'ibm_simulator':
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService()
            # 使用IBM模拟器
            backend = service.backend("ibmq_qasm_simulator")
            print(f"✓ IBM 模拟器已连接: {backend.name}")
            return backend
        elif backend_name.lower() == 'local_aer':
            # 使用AerSimulator作为本地后端
            try:
                from qiskit_aer import AerSimulator
                backend = AerSimulator()
                print("✓ 本地 AerSimulator 已就绪")
            except ImportError:
                print("⚠ AerSimulator 不可用，回退到 BasicSimulator")
                from qiskit.providers.basic_provider import BasicSimulator
                backend = BasicSimulator()
                print("✓ 本地 BasicSimulator 已就绪")
            return backend
        else:
            # 默认本地后端，优先使用AerSimulator，如果不可用则回退到BasicSimulator
            try:
                from qiskit_aer import AerSimulator
                backend = AerSimulator()
                print("✓ 本地 AerSimulator 已就绪")
            except ImportError:
                from qiskit.providers.basic_provider import BasicSimulator
                backend = BasicSimulator()
                print("✓ 本地 BasicSimulator 已就绪")
            return backend
    except Exception as e:
        print(f"✗ 后端设置失败: {e}")
        traceback.print_exc()  # 输出详细的错误追踪
        from qiskit.providers.basic_provider import BasicSimulator
        return BasicSimulator()

# ====================
# 能量计算逻辑
# ====================

def estimate_energy_from_bitstring(bitstring, qubit_op):
    """
    计算单个采样比特串对应的哈密顿量能量值
    
    Args:
        bitstring (str): 量子测量得到的比特串，如 '010110'
        qubit_op: 量子比特哈密顿量算子
        
    Returns:
        float: 对应的能量值
    """
    energy = 0.0
    # 将比特串反转，使其与哈密顿量的索引顺序对应
    bit_list = [int(b) for b in reversed(bitstring)]
    
    # 遍历哈密顿量的每一项 (Pauli算子及其系数)
    for pauli_str, coeff in qubit_op.to_list():
        val = 1.0
        # 对于每个Pauli项，计算其在当前比特态下的期望值
        for i, char in enumerate(reversed(pauli_str)):
            if char == 'Z' and bit_list[i] == 1:
                # Z算子在|1>态下贡献-1，在|0>态下贡献+1
                val *= -1.0
            elif char == 'X' or char == 'Y':
                # X,Y算子在计算能量时会引入非对角项，此处简化处理为0
                val = 0.0 
                break
        energy += coeff.real * val
    return energy

def calculate_cvar_energy(counts, qubit_op, alpha):
    """
    实现CVaR (Conditional Value at Risk) 优化策略
    CVaR是一种风险度量方法，只考虑能量最低的一部分样本
    
    Args:
        counts (dict): 量子测量结果，键为比特串，值为出现次数
        qubit_op: 量子比特哈密顿量算子
        alpha (float): CVaR参数，取值[0,1]，表示使用最低能量样本的比例
        
    Returns:
        float: CVaR能量值（最低alpha比例样本的平均能量）
    """
    energies = []
    total_shots = sum(counts.values())
    
    # 计算每个测量结果对应的能量
    for bitstring, count in counts.items():
        e = estimate_energy_from_bitstring(bitstring, qubit_op)
        # 按照测量次数复制能量值
        energies.extend([e] * count)
    
    # 按能量值升序排列
    energies.sort()
    
    # 计算需要保留的样本数量（最低能量的alpha比例）
    num_keep = max(1, int(total_shots * alpha))
    
    # 返回最低能量样本的平均值
    return np.mean(energies[:num_keep])

def extract_top_results(counts, qubit_op, top_n):
    """
    从量子测量结果中提取能量最低的top_n个结果
    
    Args:
        counts (dict): 量子测量结果，键为比特串，值为出现次数
        qubit_op: 量子比特哈密顿量算子
        top_n (int): 需要提取的最优结果数量
        
    Returns:
        list: 包含top_n个最优结果的列表，每个元素为(bitstring, energy, count)元组
    """
    # 计算每个比特串的能量
    bitstring_energies = []
    for bitstring, count in counts.items():
        energy = estimate_energy_from_bitstring(bitstring, qubit_op)
        bitstring_energies.append((bitstring, energy, count))
    
    # 按能量值升序排列
    bitstring_energies.sort(key=lambda x: x[1])
    
    # 提取能量最低的top_n个结果
    top_results = bitstring_energies[:top_n]
    
    return top_results

# ====================
# 主计算流程
# ====================
@metadata_logger
def main():
    print(f"🚀 启动蛋白质折叠计算任务 (采样器模式) | 序列: {args.main_chain}")
    
    # 导入蛋白质折叠相关模块
    from protein_folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
    from protein_folding.peptide.peptide import Peptide
    from protein_folding.protein_folding_problem import ProteinFoldingProblem
    from protein_folding.penalty_parameters import PenaltyParameters
    from src.protein_folding.utils.detailed_pdb_generator import convert_xyz_to_detailed_pdb

    # 1. 构建蛋白质折叠问题模型
    # 创建肽对象：包含主链序列和侧链序列（这里设为空字符串）
    peptide = Peptide(args.main_chain, [""] * len(args.main_chain))
    
    # 定义约束项的惩罚系数
    penalty_terms = PenaltyParameters(args.penalty_chiral, args.penalty_back, args.penalty_local_overlap)
    
    # 创建蛋白质折叠问题实例
    problem = ProteinFoldingProblem(peptide, MiyazawaJerniganInteraction(), penalty_terms)
    
    # 生成对应的量子比特哈密顿量
    qubit_op = problem.qubit_op()
    num_qubits = qubit_op.num_qubits  # 获取所需量子比特数

    # 2. 根据参数选择合适的量子计算后端
    backend = setup_sampler_backend(args.backend, args.aws_region, args.shots)
    print(f"✓ 量子后端已就绪 | 量子比特数: {num_qubits}")

    # 3. 构建参数化量子电路
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
    # 本地模拟器通常不需要转译，而云量子设备需要针对硬件特性进行转译
    if args.backend.lower() == 'local':
        # 对于本地模拟器，直接使用原始电路
        transpiled_circuit = clean_circuit
    else:
        # 对于AWS后端，需要转译电路以适配目标设备的拓扑和门集
        transpiled_circuit = transpile(clean_circuit,backend=backend,
                                        initial_layout=list(range(num_qubits)) if args.backend != 'local' else None,
                                        optimization_level=3
                                      )
    print(f"   逻辑比特数 (算法需求): {clean_circuit.num_qubits}")
    print(f"   转译后物理比特数 (硬件占用): {transpiled_circuit.num_qubits}")
    all_conv_data = []
    
    # 单次运行，用户如需重复运行可重新调用
    print(f"\n--- 实验 1/1 ---")
    np.random.seed(args.random_seed)
    
    convergence_history = []
    cumulative_shots_history = []  # 记录累计shots数
    iteration_shots_history = []  # 记录每次迭代的实际shots数
    iteration_results = []  # 记录每次迭代的多个结果
    
    # 存储每次迭代的多个最优结果的能量值，用于生成收敛图
    all_top_energies = []

    def objective_function(params):
        """
        优化目标函数
        该函数接受参数，执行量子电路，使用CVaR策略计算能量，并返回用于优化的值
        
        Args:
            params: 变分参数数组
            
        Returns:
            float: CVaR能量值（优化目标）
        """
        # 将参数绑定到量子电路
        bound_circ = transpiled_circuit.assign_parameters(params)
        
        # 在选定的后端上执行量子电路
        job = backend.run(bound_circ, shots=args.shots)
        
        # 获取量子测量结果
        counts = job.result().get_counts()
        
        # 计算实际消耗的 shots 数
        actual_shots = sum(counts.values())
        
        # 使用CVaR策略计算能量（与原始版本一致，更稳定）
        energy = calculate_cvar_energy(counts, qubit_op, args.alpha)
        
        # 同时提取多个最优结果用于后续分析
        top_results = extract_top_results(counts, qubit_op, args.max_results)
        top_energies = [result[1] for result in top_results]
        
        # 记录收敛历史和shots数，用于分析优化过程
        convergence_history.append(energy)
        iteration_shots_history.append(actual_shots)  # 记录每次迭代的实际shots数
        # 使用实际消耗的 shots 数进行累加
        if cumulative_shots_history:
            cumulative_shots_history.append(cumulative_shots_history[-1] + actual_shots)
        else:
            cumulative_shots_history.append(actual_shots)
        
        # 记录每次迭代的多个最优结果的能量值
        all_top_energies.append(top_energies)
        
        # 每隔一次迭代打印一次进度（避免过多输出）
        if len(convergence_history) % 2 == 0:
            print(f"    迭代 {len(convergence_history)}: CVaR能量 = {energy:.4f}, 实际shots = {actual_shots}")
            print(f"    各最优结果能量: {[round(e, 4) for e in top_energies]}")
        
        # 保存所有结果用于后续分析
        iteration_data = {
            "iteration": len(convergence_history),
            "cvar_energy": energy,
            "top_energies": top_energies,
            "top_results": [(result[0], float(result[1]), result[2]) for result in top_results],
            "total_counts": len(counts),
            "actual_shots": actual_shots
        }
        iteration_results.append(iteration_data)
        
        return energy

    # 开始优化 (使用 COBYLA)
    initial_params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
    res = minimize(objective_function, initial_params, method='COBYLA', 
                   options={'maxiter': args.max_optimization_iterations})
    
    # 添加CVaR能量收敛曲线
    all_conv_data.append({'counts': list(range(len(convergence_history))), 'values': convergence_history, 'cumulative_shots': cumulative_shots_history.copy(), 'label': 'CVaR Energy'})
    
    # 添加每个最优结果的收敛曲线
    if all_top_energies:
        for i in range(args.max_results):
            # 提取第i个最优结果的能量值
            top_i_energies = [energies[i] if i < len(energies) else None for energies in all_top_energies]
            # 过滤掉None值
            valid_counts = []
            valid_energies = []
            for j, energy in enumerate(top_i_energies):
                if energy is not None:
                    valid_counts.append(j)
                    valid_energies.append(energy)
            # 添加到all_conv_data
            all_conv_data.append({'counts': valid_counts, 'values': valid_energies, 'cumulative_shots': cumulative_shots_history[:len(valid_counts)], 'label': f'Top {i+1} Energy'})

    # --- 结果解析与保存 ---
    print(f"    - 正在分析最优结果...")
    
    # 使用最优参数生成最终量子电路并执行测量
    final_bound_circuit = transpiled_circuit.assign_parameters(res.x)
    final_job = backend.run(final_bound_circuit, shots=args.shots)
    final_counts = final_job.result().get_counts()
    total_shots = sum(final_counts.values())
    
    # 提取能量最低的多个最优结果
    final_top_results = extract_top_results(final_counts, qubit_op, args.max_results)
    total_shots = sum(final_counts.values())
    
    # 处理迭代结果，只保存必要的信息以减少文件大小
    processed_iteration_results = []
    for iter_data in iteration_results:
        # 处理top_results，确保所有值都是可JSON序列化的
        processed_top_results = []
        for top_result in iter_data.get("top_results", []):
            processed_result = {
                "bitstring": top_result[0],
                "energy": float(top_result[1]),
                "count": top_result[2]
            }
            processed_top_results.append(processed_result)
        
        processed_iter = {
            "iteration": iter_data["iteration"],
            "cvar_energy": iter_data["cvar_energy"],
            "top_energies": [float(e) for e in iter_data.get("top_energies", [])],
            "top_results": processed_top_results,
            "total_counts": iter_data.get("total_counts", 0),
            "actual_shots": iter_data.get("actual_shots", 0)
        }
        processed_iteration_results.append(processed_iter)
    
    # 为每个最优结果生成蛋白质结构和文件
    for idx, (bitstring, energy, count) in enumerate(final_top_results):
        print(f"\n    - 结果 {idx+1}/{args.max_results}: 能量 = {energy:.4f}, 出现次数 = {count}")
        
        # 创建模拟结果对象
        class MockResult:
            def __init__(self, bs, val, counts, total):
                # 设置最高概率比特串的振幅为1（理想情况）
                self.eigenstate = {bs: 1.0}
                # 设置优化得到的能量值
                self.eigenvalue = val
                # 提供完整的概率分布用于后续分析
                self.probabilities = {k[::-1]: v/total for k, v in counts.items()}
        
        # 创建模拟结果并解析为蛋白质结构
        raw_res = MockResult(bitstring, energy, final_counts, total_shots)
        result = problem.interpret(raw_res)
        
        print(f"    - 转向序列: {result.turn_sequence}")
        
        # 1. 保存JSON格式的结果参数
        xyz_data = result.protein_shape_file_gen.get_xyz_data()
        result_data = {
            "result_index": idx + 1,
            "energy": energy,
            "turn_sequence": result.turn_sequence,
            "main_chain_sequence": args.main_chain,
            "shots_requested": args.shots,
            "bitstring": bitstring,
            "count": count,
            "max_results": args.max_results,
            "optimization_convergence": {
                "evaluation_counts": list(range(len(convergence_history))),
                "cvar_energy_values": convergence_history,
                "cumulative_shots": cumulative_shots_history,
                "iteration_shots": iteration_shots_history
            },
            "iteration_results": processed_iteration_results,
            "xyz_coordinates": [list(row) for row in xyz_data] if xyz_data is not None else []
        }
        json_path = os.path.join(RESULT_DIR, f'result_{idx+1}_energy_{energy:.4f}.json')
        with open(json_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        print(f"    ✓ JSON文件已保存: {json_path}")
        
        # 2. 保存详细 PDB 文件
        if xyz_data is not None:
            pdb_path = os.path.join(RESULT_DIR, f'structure_{idx+1}_energy_{energy:.4f}.pdb')
            convert_xyz_to_detailed_pdb(xyz_data, pdb_path, f"Structure {idx+1}")
            print(f"    ✓ PDB文件已保存: {pdb_path}")
        
        # 3. 保存 3D 结构图
        try:
            fig_struct = result.get_figure(title=f"Result {idx+1} (E={energy:.4f})")
            png_path = os.path.join(RESULT_DIR, f"structure_{idx+1}_energy_{energy:.4f}.png")
            fig_struct.savefig(png_path)
            plt.close(fig_struct)
            print(f"    ✓ 结构图已保存: {png_path}")
        except Exception as e:
            print(f"    ⚠ 绘图失败: {e}")

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
                line, = ax1.plot(data['counts'], data['values'], marker='o', label=label, 
                         linewidth=3, color=color, linestyle='--')
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
    
    print(f"\n🎉 蛋白质折叠计算任务完成！所有结果保存在: {RESULT_DIR}")

if __name__ == "__main__":
    main()