# -*- coding: utf-8 -*-
"""
量子蛋白质折叠模拟程序 (Quantum Protein Folding Simulation)

该程序使用量子计算技术模拟蛋白质折叠过程，通过量子变分算法寻找最低能量构象。
主要功能包括：
- 量子比特映射和哈密顿量构建
- CVaR-VQE 多起点优化
- 蛋白质3D结构可视化
- PDB文件生成
- 支持本地模拟器噪声模型
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from qiskit import QuantumCircuit

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加 QuPepFold 模块到 Python 路径
qupepfold_path = Path(__file__).parent / "QuPepFold" / "QuPepFold"
sys.path.insert(0, str(qupepfold_path))

# 导入 QuPepFold 核心模块
from qupepfold.qupepfold import (
    generate_turn2qubit,           # 生成转角到量子比特的映射
    count_interaction_qubits,      # 计算相互作用量子比特数量
    build_mj_interactions,         # 构建 Miyazawa-Jernigan 相互作用矩阵
    optimize_cvar_multistart,      # CVaR-VQE 多起点优化
    build_scalable_ansatz,         # 构建可扩展的量子电路
    statevector_fold_probs,        # 计算状态向量折叠概率
    exact_hamiltonian,             # 计算精确哈密顿量
    turns_from_cfg_bits,           # 从配置比特生成转角
    dihedrals_from_turns,          # 从转角生成二面角
    build_backbone_3d,             # 构建蛋白质骨架3D结构
    write_pdb_with_conect,         # 写入带连接的PDB文件
    plot_energy_breakdown_for_most_negative,  # 绘制能量分解图
)
from lib.job_metadata_logger import JobMetadataLogger


def sampler_fold_probs(qc: QuantumCircuit, hyper: Dict, shots: int = 1024, noise_model=None) -> Dict[str, float]:
    """使用采样方法计算概率分布（支持噪声模型）
    
    Args:
        qc: 量子电路
        hyper: 超参数字典
        shots: 采样次数
        noise_model: 噪声模型（可选）
    
    Returns:
        概率分布字典
    """
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    from qiskit.primitives import SamplerResult
    import numpy as np
    
    # 获取测量的量子比特索引
    num_cfg = int(hyper["numQubitsConfig"])
    num_int = int(hyper["numQubitsInteraction"])
    measured_idx = list(range(num_cfg + num_int))
    
    # 创建带测量的电路副本
    qc_measured = qc.copy()
    qc_measured.measure_all()
    
    # 创建模拟器
    if noise_model is not None:
        simulator = AerSimulator(noise_model=noise_model)
    else:
        simulator = AerSimulator()
    
    # 转译电路
    transpiled_qc = transpile(qc_measured, simulator, optimization_level=3)
    
    # 运行采样
    job = simulator.run(transpiled_qc, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    # 转换为概率分布（只保留测量的量子比特）
    out: Dict[str, float] = {}
    total_shots = sum(counts.values())
    
    for bitstring, count in counts.items():
        # bitstring 是大端序（最高位在最左边），需要提取测量的量子比特
        # Qiskit 的 bitstring 格式: q_{n-1}...q_0
        Q = qc.num_qubits
        # 提取测量比特（按照 measured_idx 的顺序）
        fold = ''.join(bitstring[Q - 1 - idx] for idx in measured_idx)
        out[fold] = out.get(fold, 0.0) + count / total_shots
    
    return out


def create_noise_model(p_single=0.01, p_double=0.05, p_meas=0.03):
    """从IBM量子硬件获取真实噪声模型（带缓存）
    
    Args:
        p_single: 单比特门错误率（当无法连接IBM服务或加载缓存时使用）
        p_double: 双比特门错误率（当无法连接IBM服务或加载缓存时使用）
        p_meas: 测量错误率（当无法连接IBM服务或加载缓存时使用）
    """
    import pickle
    
    noise_model_file = "ibm_fez_noise.pkl"
    
    # 1. 尝试从本地文件加载噪声模型
    if os.path.exists(noise_model_file):
        try:
            print("正在从本地缓存加载IBM量子硬件噪声模型...")
            with open(noise_model_file, "rb") as f:
                noise_model = pickle.load(f)
            print("[SUCCESS] 成功加载本地缓存的噪声模型")
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
        print("[SUCCESS] 成功获取IBM量子硬件噪声模型")
        print(f"  - 后端名称: {backend.name}")
        print(f"  - 噪声模型包含的门: {noise_model.basis_gates}")
        
        # 保存噪声模型到本地文件
        try:
            with open(noise_model_file, "wb") as f:
                pickle.dump(noise_model, f)
            print(f"[SUCCESS] 噪声模型已保存到本地文件: {noise_model_file}")
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


def plot_protein_3d(atoms, seq, title, output_path):
    """
    绘制蛋白质 3D 结构图
    
    该函数使用 matplotlib 绘制蛋白质的3D结构，包括原子、化学键和氨基酸标签。
    
    Args:
        atoms: 原子坐标列表 [{"name": "N", "coords": (x, y, z)}, ...]
        seq: 氨基酸序列
        title: 图表标题
        output_path: 输出文件路径
    """
    # 创建图形和3D坐标轴
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 原子颜色映射
    color_map = {
        'N': '#1f77b4',      # 蓝色 - 氮原子
        'CA': '#d62728',     # 红色 - α碳原子
        'C': '#2ca02c',      # 绿色 - 碳原子
        'O': '#ff7f0e',      # 橙色 - 氧原子
        'CB': '#9467bd'      # 紫色 - β碳原子
    }
    
    # 提取原子坐标和属性
    x_coords = []
    y_coords = []
    z_coords = []
    colors = []
    sizes = []
    
    # 遍历所有原子，提取坐标和设置可视化属性
    for atom in atoms:
        x, y, z = atom['coords']
        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)
        colors.append(color_map.get(atom['name'], 'gray'))  # 根据原子类型设置颜色
        # 增大原子尺寸，CA 原子更大（α碳原子是蛋白质骨架的关键原子）
        sizes.append(200 if atom['name'] == 'CA' else 120)
    
    # 绘制原子（使用更大的尺寸和更高的透明度）
    scatter = ax.scatter(x_coords, y_coords, z_coords, c=colors, s=sizes, alpha=0.9, 
                       edgecolors='black', linewidths=1.5)
    
    # 绘制蛋白质骨架连接线（按照正确的化学键连接）
    # atoms 结构：对于甘氨酸G：N, CA, C, O；对于其他氨基酸：N, CA, CB, C, O
    
    # 首先构建原子索引映射
    atom_indices = {}
    for idx, atom in enumerate(atoms):
        name = atom['name']
        if name not in atom_indices:
            atom_indices[name] = []
        atom_indices[name].append(idx)
    
    # 连接规则：
    # 1. N -> CA (骨架) - 黑色粗线
    # 2. CA -> CB (侧链) - 紫色细线
    # 3. CA -> C (骨架) - 黑色粗线
    # 4. C -> O (羰基) - 橙色细线
    # 5. C -> 下一个氨基酸的 N (肽键) - 红色粗线
    
    # 遍历所有原子，建立连接
    for i, atom in enumerate(atoms):
        name = atom['name']
        coords = atom['coords']
        
        # 找到下一个原子
        if i + 1 < len(atoms):
            next_atom = atoms[i + 1]
            next_name = next_atom['name']
            next_coords = next_atom['coords']
            
            # 连接 N -> CA (骨架键)
            if name == 'N' and next_name == 'CA':
                ax.plot([coords[0], next_coords[0]], 
                       [coords[1], next_coords[1]], 
                       [coords[2], next_coords[2]], 
                       color='black', linewidth=4, alpha=0.9, zorder=1)
            
            # 连接 CA -> CB (侧链)
            elif name == 'CA' and next_name == 'CB':
                ax.plot([coords[0], next_coords[0]], 
                       [coords[1], next_coords[1]], 
                       [coords[2], next_coords[2]], 
                       color='#9467bd', linewidth=3, alpha=0.8, zorder=1)
            
            # 连接 CA -> C (骨架键)
            elif name == 'CA' and next_name == 'C':
                ax.plot([coords[0], next_coords[0]], 
                       [coords[1], next_coords[1]], 
                       [coords[2], next_coords[2]], 
                       color='black', linewidth=4, alpha=0.9, zorder=1)
            
            # 连接 C -> O (羰基)
            elif name == 'C' and next_name == 'O':
                ax.plot([coords[0], next_coords[0]], 
                       [coords[1], next_coords[1]], 
                       [coords[2], next_coords[2]], 
                       color='#ff7f0e', linewidth=3, alpha=0.8, zorder=1)
    
    # 连接肽键：C -> 下一个氨基酸的 N (红色粗线强调)
    # 找到所有 C 原子和 N 原子的索引
    c_indices = [i for i, atom in enumerate(atoms) if atom['name'] == 'C']
    n_indices = [i for i, atom in enumerate(atoms) if atom['name'] == 'N']
    
    # 连接每个 C 到下一个 N（除了最后一个 C）
    for i in range(len(c_indices) - 1):
        c_idx = c_indices[i]
        n_idx = n_indices[i + 1]  # 下一个氨基酸的 N
        if c_idx < len(atoms) and n_idx < len(atoms):
            c_atom = atoms[c_idx]
            n_atom = atoms[n_idx]
            ax.plot([c_atom['coords'][0], n_atom['coords'][0]], 
                   [c_atom['coords'][1], n_atom['coords'][1]], 
                   [c_atom['coords'][2], n_atom['coords'][2]], 
                   color='#d62728', linewidth=5, alpha=1.0, zorder=0)  # 肽键用红色粗线强调
    
    # 标记每个氨基酸的 CA 原子（使用更明显的标签）
    aa_colors = plt.cm.tab20(np.linspace(0, 1, len(seq)))
    ca_indices = [i for i, atom in enumerate(atoms) if atom['name'] == 'CA']
    for idx, ca_idx in enumerate(ca_indices):
        if idx < len(seq):
            x, y, z = x_coords[ca_idx], y_coords[ca_idx], z_coords[ca_idx]
            # 使用更大的字体和更明显的颜色
            ax.text(x, y, z, f' {seq[idx]}', fontsize=14, fontweight='bold', 
                   color='black', bbox=dict(boxstyle='round,pad=0.3', 
                                          facecolor='white', 
                                          edgecolor='black',
                                          alpha=0.8))
    
    # 设置标签和标题
    ax.set_xlabel('X (Å)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (Å)', fontsize=14, fontweight='bold')
    ax.set_zlabel('Z (Å)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 设置背景为白色，提高可读性
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 添加图例
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=color, markersize=10, label=name)
                      for name, color in color_map.items()]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # 设置相等的比例
    max_range = np.array([max(x_coords)-min(x_coords), 
                         max(y_coords)-min(y_coords), 
                         max(z_coords)-min(z_coords)]).max() / 2.0
    mid_x = (max(x_coords)+min(x_coords)) * 0.5
    mid_y = (max(y_coords)+min(y_coords)) * 0.5
    mid_z = (max(z_coords)+min(z_coords)) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 调整视角
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"3D structure plot -> {output_path}")


metadata_logger = JobMetadataLogger("protein_folding_jobs_detailed.csv")

@metadata_logger
def main():
    """
    主函数：量子蛋白质折叠模拟程序入口点
    
    该函数处理命令行参数，执行量子蛋白质折叠模拟，并生成结果文件。
    """
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(prog="run_qupepfold", description="量子蛋白质折叠模拟程序 (Quantum Protein Folding Simulation)")
    # 蛋白质序列参数 (2-10个氨基酸)
    parser.add_argument("--seq", type=str, default='APRLRFY',help="蛋白质序列 (2-10个氨基酸, 例如: APRLRFY)")
    # CVaR-VQE 优化参数
    parser.add_argument("--max_optimization_iterations", type=int, default=10,help="最大优化迭代次数 (默认: 10)")
    parser.add_argument("--alpha", type=float, default=0.025,help="CVaR尾部质量参数 (0<alpha<1, 默认: 0.025)")
    # 量子计算参数
    parser.add_argument("--shots", type=int, default=1024,help="(信息性) 量子测量次数 (默认: 1024)")
    parser.add_argument("--backend", default="local",choices=["local", "aws_sv1", "aws_garnet", "aws_forte"],help="量子计算后端 (默认: local)")
    # 噪声模型参数
    parser.add_argument("--use_noise", action="store_true", help="使用噪声模型模拟真实量子硬件噪声")
    parser.add_argument("--noise_single", type=float, default=0.01, help="单比特门错误率 (默认: 0.01)")
    parser.add_argument("--noise_double", type=float, default=0.05, help="双比特门错误率 (默认: 0.05)")
    parser.add_argument("--noise_meas", type=float, default=0.03, help="测量错误率 (默认: 0.03)")
    # 解析命令行参数
    args = parser.parse_args()

    # 验证蛋白质序列
    seq = args.seq.upper()
    if not (2 <= len(seq) <= 10) or any(c not in "ARNDCEQGHILKMFPSTWYV" for c in seq):
        raise SystemExit("错误: --seq 参数必须是2-10个标准单字母代码的氨基酸。")

    # 创建带时间戳和后端名称的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backend_name = args.backend
    if args.use_noise:
        output_dir = Path("./results") / f"{timestamp}_qupepfold_{backend_name}_noise"
    else:
        output_dir = Path("./results") / f"{timestamp}_qupepfold_{backend_name}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"输出目录: {output_dir}")

    # 构建量子比特映射和超参数 (与核心模块对齐)
    # 生成转角到量子比特的映射
    turn2qubit, fixed_bits, variable_bits = generate_turn2qubit(seq)
    
    # 计算配置量子比特和相互作用量子比特数量
    num_q_cfg = turn2qubit.count("q")  # 配置量子比特数量
    num_q_int = count_interaction_qubits(seq)  # 相互作用量子比特数量

    print("\n" + "=" * 60)
    print("run_qupepfold.py - Qubit 和 Shot 信息")
    print("=" * 60)
    print()
    print("【输入参数】")
    print(f"  蛋白质序列: {seq}")
    print(f"  量子后端: {args.backend}")
    print(f"  CVaR alpha: {args.alpha}")
    print(f"  最大优化迭代次数: {args.max_optimization_iterations}")
    print(f"  Shot 数量: {args.shots}")
    print(f"  噪声模型: {'已启用' if args.use_noise else '已禁用'}")
    if args.use_noise:
        print(f"  - 噪声参数:")
        print(f"    * 单比特门错误率: {args.noise_single}")
        print(f"    * 双比特门错误率: {args.noise_double}")
        print(f"    * 测量错误率: {args.noise_meas}")
    print()
    print("【Qubit 数量】")
    print(f"  配置量子比特: {num_q_cfg}")
    print(f"  相互作用量子比特: {num_q_int}")
    print(f"  总计(含辅助比特): {num_q_cfg + num_q_int + 1}")
    print()
    print("【Qubit 计算说明】")
    print(f"  配置量子比特 = 转角到量子比特映射中的 'q' 数量")
    print(f"  相互作用量子比特 = count_interaction_qubits({seq})")
    print(f"  辅助比特 = 1 (用于 CVaR-VQE)")
    print()
    print("【Shot 配置】")
    print(f"  默认 Shot 数量: 1024")
    print(f"  当前 Shot 数量: {args.shots}")
    print(f"  配置方式: --shots 参数")
    print(f"  应用位置: hyper['numShots']")
    print()
    print("=" * 60)
    print()
    
    # 构建超参数字典
    hyper = {
        "protein": seq,                           # 蛋白质序列
        "turn2qubit": turn2qubit,                 # 转角到量子比特映射
        "numQubitsConfig": num_q_cfg,            # 配置量子比特数量
        "numQubitsInteraction": num_q_int,        # 相互作用量子比特数量
        "interactionEnergy": build_mj_interactions(seq),  # Miyazawa-Jernigan相互作用能量
        "numShots": int(args.shots),              # 量子测量次数
    }

    # 如果启用噪声模型，添加到超参数字典
    if args.use_noise:
        noise_model = create_noise_model(
            p_single=args.noise_single,
            p_double=args.noise_double,
            p_meas=args.noise_meas
        )
        hyper["noise_model"] = noise_model
        print(f"[噪声模型] 已启用噪声模型")
    else:
        hyper["noise_model"] = None

    # CVaR-VQE 多起点优化
    print(f"\n[CVaR-VQE] alpha={args.alpha}, 最大优化迭代次数={args.max_optimization_iterations}")
    best_x, best_cvar, trace, tries_info = optimize_cvar_multistart(hyper, args.max_optimization_iterations, args.alpha)
    print(f"[CVaR-VQE] 最优CVaR能量: {best_cvar:.6f}")
    
    # 收集每次迭代的电路信息
    iteration_details = []
    for idx, (info, cvar_val) in enumerate(zip(tries_info, trace), start=1):
        params = np.asarray(info.get("x", []), float)
        if params.size == 0:
            continue
        qc_iter = build_scalable_ansatz(params, hyper, measure=False)
        decomposed_circuit = qc_iter.decompose()
        total_gates = decomposed_circuit.size()
        single_qubit_gates = sum(1 for op in decomposed_circuit.data if len(op.qubits) == 1)
        two_qubit_gates = sum(1 for op in decomposed_circuit.data if len(op.qubits) == 2)
        circuit_depth = decomposed_circuit.depth()
        
        iter_time = info.get("time", 0.0)
        # For local backend, attribute iteration time to (simulated) quantum time
        if args.backend == "local":
            q_time, c_time = iter_time, 0.0
        else:
            q_time, c_time = 0.0, iter_time
            
        iteration_details.append({
            'trial_idx': 1,
            'iteration': idx,
            'backend': args.backend,
            'total_gates': total_gates,
            'single_qubit_gates': single_qubit_gates,
            'two_qubit_gates': two_qubit_gates,
            'circuit_depth': circuit_depth,
            'shots': int(args.shots),
            'energy': float(cvar_val),
            'cumulative_shots': int(args.shots) * idx,
            'cvar_energy': float(cvar_val),
            'quantum_time': round(q_time, 3),
            'queue_time': 0.0,
            'classical_time': round(c_time, 3),
            'total_time': round(iter_time, 3)
        })
    
    # 生成 iteration_details.csv 文件
    iteration_csv_path = output_dir / "iteration_details.csv"
    with open(iteration_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['trial_idx', 'iteration', 'backend', 'total_gates', 'single_qubit_gates', 
                       'two_qubit_gates', 'circuit_depth', 'shots', 'energy', 'cumulative_shots', 'cvar_energy', 'quantum_time', 'queue_time', 'classical_time', 'total_time'])
        for detail in iteration_details:
            writer.writerow([
                detail['trial_idx'],
                detail['iteration'],
                detail['backend'],
                detail['total_gates'],
                detail['single_qubit_gates'],
                detail['two_qubit_gates'],
                detail['circuit_depth'],
                detail['shots'],
                detail['energy'],
                detail['cumulative_shots'],
                detail['cvar_energy'],
                detail['quantum_time'],
                detail['queue_time'],
                detail['classical_time'],
                detail['total_time']
            ])
    print(f"Iteration details saved to {iteration_csv_path}")

    # 保存每迭代结果（与 run_qthesis.py / run_opt.py 对齐）
    try:
        iterations_dir = output_dir / "iterations"
        os.makedirs(iterations_dir, exist_ok=True)
        import json
        for idx, (info, cvar_val) in enumerate(zip(tries_info, trace), start=1):
            params = np.asarray(info.get("x", []), float)
            if params.size == 0:
                continue
            qc_iter = build_scalable_ansatz(params, hyper, measure=False)
            probs_iter = statevector_fold_probs(qc_iter, hyper)
            if not probs_iter:
                continue
            states_iter = list(probs_iter.keys())
            energies_iter = exact_hamiltonian(states_iter, hyper)
            s_min_idx = int(min(range(len(states_iter)), key=lambda i: energies_iter[i]))
            s_min_energy = states_iter[s_min_idx]
            e_min = float(energies_iter[s_min_idx])
            cfg_bits = s_min_energy[:num_q_cfg]
            turns = turns_from_cfg_bits(cfg_bits, turn2qubit)
            phis, psis = dihedrals_from_turns(turns, len(seq))
            atoms = build_backbone_3d(seq, phis, psis)
            xyz_coordinates = [[atom["name"], *atom["coords"]] for atom in atoms]
            with open(iterations_dir / f"iteration_{idx}_result.json", "w", encoding="utf-8") as f:
                json.dump({
                    "index": idx,
                    "cvar_value": float(cvar_val),
                    "min_energy": e_min,
                    "bitstring": s_min_energy,
                    "shots": int(args.shots),
                    "backend": args.backend,
                    "protein_structure": {
                        "turn_sequence": turns,
                        "xyz_coordinates": xyz_coordinates
                    },
                    "optimization_convergence": {
                        "evaluation_counts": list(range(1, idx + 1)),
                        "cvar_values": trace[:idx],
                        "cumulative_shots": [args.shots * i for i in range(1, idx + 1)],
                        "iteration_shots": [args.shots] * idx
                    },
                    "timestamp": datetime.now().isoformat(),
                    "timing": {
                        "quantum_time": round(q_time, 3),
                        "queue_time": 0.0,
                        "classical_time": round(c_time, 3),
                        "total_time": round(iter_time, 3)
                    }
                }, f, indent=2)
    except Exception:
        pass

    # 在最优解处计算概率分布
    qc = build_scalable_ansatz(best_x, hyper, measure=False)  # 构建可扩展量子电路
    
    # 根据是否启用噪声选择计算方法
    if args.use_noise:
        print(f"[噪声模型] 使用采样方法计算概率分布 (shots={args.shots})")
        probs = sampler_fold_probs(qc, hyper, shots=args.shots, noise_model=hyper.get("noise_model"))
    else:
        probs = statevector_fold_probs(qc, hyper)
        print(f"[理想模拟] 使用状态向量计算精确概率分布")
    
    states = list(probs.keys())                               # 获取所有可能的状态
    energies = exact_hamiltonian(states, hyper)               # 计算每个状态的精确能量

    # 报告：最可能和最低能量的比特串
    s_most_prob = max(states, key=lambda s: probs[s])         # 概率最高的比特串
    s_min_idx = int(min(range(len(states)), key=lambda i: energies[i]))  # 能量最低的索引
    s_min_energy = states[s_min_idx]                          # 能量最低的比特串

    print("\n=== 最优解结果 ===")
    print(f"最可能比特串 : {s_most_prob} (概率={probs[s_most_prob]:.6f})")
    print(f"最低能量比特串 : {s_min_energy} (能量={energies[s_min_idx]:.6f})")

    # CSV 数据导出 (始终启用)
    csv_path = output_dir / "bitstring_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bitstring", "cfg_bits", "probability", "energy"])
        for s, e in zip(states, energies):
            w.writerow([s, s[:num_q_cfg], float(probs[s]), float(e)])
    print(f"\n写入CSV文件 -> {csv_path}")

    # 为最可能和最低能量的比特串生成PDB文件和可视化
    print("\n=== 生成PDB文件和可视化 ===")
    
    # 创建pdb3d子目录
    pdb_dir = output_dir / "pdb3d"
    os.makedirs(pdb_dir, exist_ok=True)
    
    # 为最可能比特串生成PDB文件
    cfg_bits_most_prob = s_most_prob[:num_q_cfg]  # 提取配置比特
    turns_most_prob = turns_from_cfg_bits(cfg_bits_most_prob, turn2qubit)  # 从配置比特生成转角
    phis_most_prob, psis_most_prob = dihedrals_from_turns(turns_most_prob, len(seq))  # 生成二面角
    atoms_most_prob = build_backbone_3d(seq, phis_most_prob, psis_most_prob)  # 构建蛋白质骨架3D结构
    pdb_path_most_prob = pdb_dir / f"fold3d_most_probable_{cfg_bits_most_prob}.pdb"
    write_pdb_with_conect(cfg_bits_most_prob, seq, atoms_most_prob, str(pdb_path_most_prob))
    print(f"最可能结构PDB -> {pdb_path_most_prob}")
    
    # 为最可能比特串生成3D结构图
    plot_path_most_prob = output_dir / f"3d_structure_most_probable_{cfg_bits_most_prob}.png"
    plot_protein_3d(atoms_most_prob, seq, 
                   f"最可能结构 - {seq} (概率={probs[s_most_prob]:.6f})", 
                   str(plot_path_most_prob))
    
    # 为最低能量比特串生成PDB文件
    cfg_bits_min_energy = s_min_energy[:num_q_cfg]  # 提取配置比特
    turns_min_energy = turns_from_cfg_bits(cfg_bits_min_energy, turn2qubit)  # 从配置比特生成转角
    phis_min_energy, psis_min_energy = dihedrals_from_turns(turns_min_energy, len(seq))  # 生成二面角
    atoms_min_energy = build_backbone_3d(seq, phis_min_energy, psis_min_energy)  # 构建蛋白质骨架3D结构
    pdb_path_min_energy = pdb_dir / f"fold3d_lowest_energy_{cfg_bits_min_energy}.pdb"
    write_pdb_with_conect(cfg_bits_min_energy, seq, atoms_min_energy, str(pdb_path_min_energy))
    print(f"最低能量结构PDB -> {pdb_path_min_energy}")
    
    # 为最低能量比特串生成3D结构图
    plot_path_min_energy = output_dir / f"3d_structure_lowest_energy_{cfg_bits_min_energy}.png"
    plot_protein_3d(atoms_min_energy, seq, 
                   f"最低能量结构 - {seq} (能量={energies[s_min_idx]:.6f})", 
                   str(plot_path_min_energy))
    
    # 生成最低能量比特串的能量分解图
    print("\n生成能量分解可视化...")
    plot_energy_breakdown_for_most_negative(probs, hyper, str(output_dir))
    print(f"能量分解图 -> {output_dir / 'most_negative_energy_breakdown.png'}")

    metrics_path = output_dir / "metrics.json"
    last_detail = iteration_details[-1] if iteration_details else {}
    total_iter_time = sum(info.get("time", 0.0) for info in tries_info)
    if args.backend == "local":
        total_q_time, total_c_time = total_iter_time, 0.0
    else:
        total_q_time, total_c_time = 0.0, total_iter_time

    with open(metrics_path, "w", encoding="utf-8") as f:
        import json
        transpile_metrics = {
            "total_gates": last_detail.get('total_gates'),
            "single_qubit_gates": last_detail.get('single_qubit_gates'),
            "two_qubit_gates": last_detail.get('two_qubit_gates'),
            "circuit_depth": last_detail.get('circuit_depth')
        }
        convergence_metrics = {}
        json.dump({
            "backend": args.backend,
            "shots_requested": int(args.shots),
            "shots_actual_total": int(args.shots) * int(args.max_optimization_iterations),
            "iteration_count": int(args.max_optimization_iterations),
            "outcome_summary": f"cvar_min={float(best_cvar):.6f}",
            "qubits_used": int(num_q_cfg + num_q_int + 1),
            "qubits_full": int(num_q_cfg + num_q_int + 1),
            "total_quantum_time": round(total_q_time, 3),
            "total_queue_time": 0.0,
            "total_classical_time": round(total_c_time, 3),
            "total_gates": last_detail.get('total_gates'),
            "single_qubit_gates": last_detail.get('single_qubit_gates'),
            "two_qubit_gates": last_detail.get('two_qubit_gates'),
            "circuit_depth": last_detail.get('circuit_depth'),
            "transpile_metrics": transpile_metrics,
            "convergence_metrics": convergence_metrics
        }, f, indent=2)

    # Dump timing_summary.json
    total_iter_time = sum(info.get("time", 0.0) for info in tries_info)
    if args.backend == "local":
        total_q_time, total_c_time = total_iter_time, 0.0
    else:
        total_q_time, total_c_time = 0.0, total_iter_time
        
    timing_summary = {
        "num_iterations": args.max_optimization_iterations,
        "total_quantum_time": round(total_q_time, 3),
        "total_queue_time": 0.0,
        "total_classical_time": round(total_c_time, 3),
        "total_time": round(total_iter_time, 3)
    }
    timing_path = output_dir / "timing_summary.json"
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing_summary, f, indent=2)
    print(f"\n[SUCCESS] 时间汇总已保存到: {timing_path}")
    print(f"  - 总迭代次数: {timing_summary['num_iterations']}")
    print(f"  - 总量子时间: {timing_summary['total_quantum_time']:.3f}s")
    print(f"  - 总队列时间: {timing_summary['total_queue_time']:.3f}s")
    print(f"  - 总经典时间: {timing_summary['total_classical_time']:.3f}s")
    print(f"  - 总时间: {timing_summary['total_time']:.3f}s")


if __name__ == "__main__":
    """
    程序入口点
    
    当直接运行此脚本时，执行量子蛋白质折叠模拟。
    使用方法：
        python run_qupepfold.py --seq APRLRFY --backend local --tries 10 --alpha 0.025
    
    技术说明：
    - 使用CVaR-VQE算法进行量子优化
    - 支持本地模拟器和AWS量子后端
    - 生成PDB文件和3D可视化结果
    - 输出详细的能量和概率分析
    """
    main()
