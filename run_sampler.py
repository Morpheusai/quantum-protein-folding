#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
蛋白质折叠量子算法 - Sampler模式

功能特点：
1. 支持本地模拟器和AWS Braket后端
2. 使用CVaR (Conditional Value at Risk) 优化策略
3. 基于采样的能量计算方法
4. 支持多轮独立实验及结果汇总分析
"""

# =============================================================================
# 1. 导入依赖库
# =============================================================================
import argparse
import os
import sys
import warnings
import json
import numpy as np

# 设置Python环境编码
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.name == 'nt':
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# 添加项目路径
sys.path.insert(0, os.path.join(current_dir, 'src'))
sys.path.insert(0, os.path.join(current_dir, 'lib'))

# 配置matplotlib为非交互模式
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

# 导入量子计算相关库
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RealAmplitudes
from scipy.optimize import minimize

# 导入自定义模块
from lib.quantum_backend_manager import QuantumBackendManager
from lib.energy_calculator import EnergyCalculator
from lib.protein_folding_builder import ProteinFoldingBuilder
from lib.result_handler import ResultHandler
from lib.quantum_optimizer import QuantumOptimizer, MockQuantumResult
from lib.job_metadata_logger import JobMetadataLogger

# =============================================================================
# 2. 命令行参数配置
# =============================================================================
parser = argparse.ArgumentParser(description='蛋白质折叠量子算法 - 采样器模式')

# 量子后端配置
parser.add_argument('--backend', default='local',help='量子后端选择: local (本地模拟器，优先使用AerSimulator), local_aer (强制使用AerSimulator), aws_sv1 (AWS模拟器), aws_garnet (AWS量子芯片), aws_ionq (AWS IonQ量子设备), aws_forte (AWS IonQ Forte量子设备), ibm (IBM量子设备), ibm_simulator (IBM模拟器)')

# 算法参数
parser.add_argument('--random_seed', type=int, default=23,help='随机种子，用于确保结果可重现')
parser.add_argument('--max_optimization_iterations', type=int, default=10,help='最大优化迭代次数')
parser.add_argument('--ansatz_reps', type=int, default=1,help='变分量子线路的重复层数')

# 蛋白质序列参数
parser.add_argument('--main_chain', default='APRLRFY',help='蛋白质主链氨基酸序列')

# 约束惩罚参数
parser.add_argument('--penalty_back', type=float, default=10,help='几何约束惩罚系数')
parser.add_argument('--penalty_chiral', type=float, default=10,help='手性约束惩罚系数')
parser.add_argument('--penalty_local_overlap', type=float, default=10,help='局部重叠惩罚系数')

# CVaR优化参数
parser.add_argument('--alpha', type=float, default=0.1,help='CVaR参数: 选择最低能量的alpha比例样本')

# 采样参数
parser.add_argument('--shots', type=int, default=1000, help='采样次数')
parser.add_argument('--dry_run', action='store_true', help='干跑模式：仅在真实提交前进行本地预检')
parser.add_argument('--max_results', type=int, default=1,help='每次迭代的结果数量')

# AWS配置
parser.add_argument('--aws_region', default=None,help='AWS区域设置（可选）')

# 优化增强参数
parser.add_argument('--optimizer', default='COBYLA', help='优化器选择: COBYLA (默认), SPSA, SLSQP')
parser.add_argument('--resilience_level', type=int, default=1, help='IBM Quantum 误差抑制等级 (0-3)，默认1')

# 自适应Shots参数
parser.add_argument('--adaptive_shots', action='store_true', help='开启自适应Shots策略')
parser.add_argument('--min_shots', type=int, default=100, help='自适应Shots的最小采样数 (默认为100)')
parser.add_argument('--max_shots', type=int, default=2000, help='自适应Shots的最大采样数 (默认为2000)')
parser.add_argument('--unique_structures', action='store_true', help='开启结构去重：仅返回折叠结构不同的最优结果')
parser.add_argument('--restarts', type=int, default=1, help='独立实验运行次数 (Multi-Restart)，用于避免局部最优，默认为1')
parser.add_argument('--resume', type=str, default=None, help='断点续传：指定结果目录以恢复历史任务')
parser.add_argument('--initial_params', type=str, default=None, help='Warm-Start：从 JSON 文件加载初始参数向量')

args = parser.parse_args()

# =============================================================================
# 3. 全局变量和装饰器
# =============================================================================
metadata_logger = JobMetadataLogger("protein_folding_jobs.csv")


@metadata_logger
def main():
    """
    主函数：执行蛋白质折叠量子计算（Sampler模式）
    
    使用CVaR优化策略和Sampler模式优化蛋白质折叠能量
    """
    # =============================================================================
    # 3.1 初始化与参数验证
    # =============================================================================
    print(f"🚀 启动蛋白质折叠计算任务 (采样器模式) | 序列: {args.main_chain}")
    
    print(f"\n参数配置:")
    print(f"  - 量子后端: {args.backend}")
    print(f"  - 随机种子: {args.random_seed}")
    print(f"  - 重启次数: {args.restarts} (Multi-Restart)")
    print(f"  - 最大优化迭代次数: {args.max_optimization_iterations}")
    print(f"  - Ansatz重复次数: {args.ansatz_reps}")
    print(f"  - 主链序列: {args.main_chain}")
    print(f"  - 几何约束惩罚: {args.penalty_back}")
    print(f"  - 手性约束惩罚: {args.penalty_chiral}")
    print(f"  - 局部重叠惩罚: {args.penalty_local_overlap}")
    print(f"  - CVaR参数alpha: {args.alpha}")
    print(f"  - 量子采样次数: {args.shots}")
    print(f"  - 最大结果数量: {args.max_results}")
    print(f"  - 优化器: {args.optimizer}")
    print(f"  - 误差抑制等级: {args.resilience_level}")
    if args.adaptive_shots:
        print(f"  - 自适应Shots: ON (Min: {args.min_shots}, Max: {args.max_shots})")
    else:
        print(f"  - 自适应Shots: OFF")

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

    # =============================================================================
    # 3.2 构建蛋白质折叠问题
    # =============================================================================
    print(f"\n正在构建蛋白质折叠问题...")
    builder = ProteinFoldingBuilder(
        args.main_chain,
        penalty_chiral=args.penalty_chiral,
        penalty_back=args.penalty_back,
        penalty_local_overlap=args.penalty_local_overlap
    )
    qubit_op = builder.get_qubit_operator()
    problem = builder.get_problem()
    num_qubits = builder.get_num_qubits()
    print(f"✓ 量子比特哈密顿量构建完成，量子比特数: {num_qubits}")

    # =============================================================================
    # 3.3 配置量子后端
    # =============================================================================
    print(f"\n正在配置量子后端...")
    backend_info = QuantumBackendManager.setup_backend(args.backend, args.aws_region, args.shots, use_estimator=False, resilience_level=args.resilience_level)
    backend = backend_info['backend']
    print(f"✓ 量子后端已就绪 | 量子比特数: {num_qubits}")

    # =============================================================================
    # 3.4 构建量子电路
    # =============================================================================
    print(f"\n正在构建参数化量子电路...")
    ansatz = RealAmplitudes(num_qubits=num_qubits, reps=args.ansatz_reps)
    
    # 分解电路并移除barrier指令
    decomposed = ansatz.decompose()
    
    clean_circuit = QuantumCircuit(num_qubits)
    for inst in decomposed.data:
        if inst.operation.name != 'barrier':
            clean_circuit.append(inst.operation, inst.qubits, inst.clbits)
    
    clean_circuit.measure_all()
    
    # 根据后端类型进行电路转译
    if args.backend.lower() in ('local', 'local_aer'):
        transpiled_circuit = clean_circuit
    else:
        transpiled_circuit = transpile(clean_circuit, backend=backend,
                                        initial_layout=list(range(num_qubits)),
                                        optimization_level=3)
    print(f"✓ 量子电路构建完成")
    print(f"   逻辑比特数 (算法需求): {clean_circuit.num_qubits}")
    print(f"   转译后物理比特数 (硬件占用): {transpiled_circuit.num_qubits}")
    
    # =============================================================================
    # 3.5 创建结果目录
    # =============================================================================
    result_dir = ResultHandler.create_result_directory("results", args.backend, mode='sampler')
    print(f"  - 结果目录: {result_dir}")
    
    all_conv_data = []
    
    # =============================================================================
    # 3.6 执行CVaR优化
    # =============================================================================
    best_energy = float('inf')
    best_results_tuple = None
    best_trial_idx = -1
    
    # 全局候选池，用于结构去重
    global_candidates = [] # 存储 (bitstring, energy, count) 形式的元组
    
    for i in range(args.restarts):
        trial_idx = i + 1
        print(f"\n--- 实验 {trial_idx}/{args.restarts} ---")
        
        # 独立的随机种子
        current_seed = args.random_seed + i if args.random_seed is not None else None
        if current_seed is not None:
             np.random.seed(current_seed)
             print(f"  随机种子: {current_seed}")
        
        # 为每个实验创建子目录，避免覆盖
        trial_dir = os.path.join(result_dir, f"trial_{trial_idx}")
        # 注意：create_sampler_optimizer 内部可能不会创建 trial_dir (它假设 result_dir 存在)
        # 实际上 internal logic creates iteration files inside result_dir
        os.makedirs(trial_dir, exist_ok=True)
        
        try:
            # 调用CVaR优化器
            results_tuple = QuantumOptimizer.create_sampler_optimizer(
                transpiled_circuit, qubit_op, backend, args, trial_dir, problem, 
                sampler_v2=backend_info.get('sampler_v2')
            )
            
            # 结果解包: res, convergence_history, std_history, iteration_results, all_top_energies, cumulative_shots_history, iteration_shots_history
            res, convergence_history = results_tuple[0], results_tuple[1]
            iteration_results = results_tuple[3]
            
            # 提取该实验中出现的所有优秀 bitstrings (从 iteration_results 中获取)
            trial_candidates = []
            for iter_data in iteration_results:
                # iter_data['top_results'] 包含 (bitstring, energy, count)
                trial_candidates.extend(iter_data.get('top_results', []))
            
            # 将该实验的优秀候选者加入全局池
            global_candidates.extend(trial_candidates)
            
            # 获取该次实验的最低能量
            min_energy = min(convergence_history) if len(convergence_history) > 0 else float('inf')
            print(f"  [实验 {trial_idx} 结束] 最低能量: {min_energy:.4f}")
            
            if min_energy < best_energy:
                best_energy = min_energy
                best_results_tuple = results_tuple
                best_trial_idx = trial_idx
                print(f"  ★ 发现新最佳结果 (能量: {min_energy:.4f})")
                
        except Exception as e:
            print(f"  ❌ 实验 {trial_idx} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    if best_results_tuple is None:
        print("\n❌ 所有实验均失败，无法产生结果。")
        return

    print(f"\n=============================================================================")
    print(f"所有实验结束。最佳实验: {best_trial_idx if best_trial_idx != -1 else 'N/A'} (能量: {best_energy:.4f})")
    print(f"正在分析全球 {len(global_candidates)} 个候选状态并执行去重...")
    print(f"=============================================================================")
    
    # 解包最佳结果供后续收敛图使用
    res, convergence_history, std_history, iteration_results, all_top_energies, cumulative_shots_history, iteration_shots_history = best_results_tuple
    
    all_conv_data = []
    
    # =============================================================================
    # 3.7 收集收敛数据 (从最佳结果中)
    # =============================================================================
    all_conv_data.append({
        'counts': list(range(len(convergence_history))), 
        'values': convergence_history, 
        'stds': std_history,
        'cumulative_shots': cumulative_shots_history.copy(), 
        'iteration_shots': iteration_shots_history, 
        'label': 'CVaR Energy (Best Trial)'
    })
    
    # 收集每个迭代的前N个最优能量结果
    if all_top_energies:
        for i in range(args.max_results):
            top_i_energies = [energies[i] if i < len(energies) else None for energies in all_top_energies]
            valid_counts = []
            valid_energies = []
            for j, energy in enumerate(top_i_energies):
                if energy is not None:
                    valid_counts.append(j)
                    valid_energies.append(energy)
            all_conv_data.append({'counts': valid_counts, 'values': valid_energies, 'cumulative_shots': cumulative_shots_history[:len(valid_counts)], 'iteration_shots': iteration_shots_history[:len(valid_counts)], 'label': f'Top {i+1} Energy'})

    # =============================================================================
    # 3.8 结果解析与去重逻辑
    # =============================================================================
    print(f"\n--- 全局结果分析与保存 ---")
    
    # 1. 整理全局候选池：去掉重复的 bitstring，保留其能量最低的记录
    merged_candidates = {}
    for bs, energy, count in global_candidates:
        if bs not in merged_candidates or energy < merged_candidates[bs][0]:
            merged_candidates[bs] = (energy, count)
    
    # 转换回列表并按能量排序
    sorted_unique_bs = sorted(merged_candidates.items(), key=lambda x: x[1][0])
    
    final_top_results = []
    if args.unique_structures:
        print(f"    - 正在执行跨实验结构去重 (目标: {args.max_results} 个不同结构)...")
        seen_structures = set()
        
        for bitstring, (energy, count) in sorted_unique_bs:
            try:
                # 构造 MockResult。注意：eigenstate 必须是字典 {bitstring: 1.0} 才能被 interpret 正确解析
                temp_mock_result = MockQuantumResult({bitstring: 1.0}, energy, {bitstring: count}, args.shots)
                temp_result = problem.interpret(temp_mock_result)
                ts = temp_result.turn_sequence
                ts_str = str(ts)
                
                if ts_str not in seen_structures:
                    seen_structures.add(ts_str)
                    final_top_results.append((bitstring, energy, count))
                    if len(final_top_results) >= args.max_results:
                        break
            except Exception:
                continue
                
        if len(final_top_results) < args.max_results:
            print(f"    ⚠ 警告: 仅在全局候选池中找到 {len(final_top_results)} 个唯一结构 (请求 {args.max_results} 个)")
    else:
        # 不去重，直接从排序后的候选池取前 N 个
        final_top_results = [(bs, en, count) for bs, (en, count) in sorted_unique_bs[:args.max_results]]
    
    # 处理迭代结果数据
    processed_iteration_results = []
    for iter_data in iteration_results:
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
            "actual_shots": iter_data.get("actual_shots", 0),
            "protein_structure": iter_data.get("protein_structure", {})  # 从迭代数据中直接获取蛋白质结构信息
        }
        processed_iteration_results.append(processed_iter)
    
    # =============================================================================
    # 3.9 保存每个最优结果
    # =============================================================================
    for idx, (bitstring, energy, count) in enumerate(final_top_results):
        print(f"\n    - 结果 {idx+1}/{len(final_top_results)}: 能量 = {energy:.4f}")
        
        try:
            # 创建模拟量子结果对象。注意：eigenstate 必须是字典 {bitstring: 1.0}
            mock_result = MockQuantumResult({bitstring: 1.0}, energy, {bitstring: count}, count)
            
            # 解析蛋白质结构
            result = problem.interpret(mock_result)
            print(f"    - 蛋白质结构解析完成")
            protein_structure = "".join(result.protein_shape_file_gen.main_chain_aminoacid_list)
            print(f"    - 蛋白质形状: {protein_structure}")
            print(f"    - 蛋白质能量: {energy:.4f}")
            
            # 准备结果数据
            xyz_data = result.protein_shape_file_gen.get_xyz_data()
            xyz_coords = [[str(row[0]), str(row[1]), str(row[2]), str(row[3])] for row in xyz_data]
            energy_value = float(energy)
            energy_str = f"{abs(energy_value):.4f}"
            
            # 保存JSON结果文件
            result_filename = os.path.join(result_dir, f"result_rank_{idx+1}_energy_{energy_str}.json")
            result_dict = {
                "rank": idx + 1,
                "energy": energy_value,
                "turn_sequence": result.turn_sequence,
                "main_chain_sequence": protein_structure,
                "shots_requested": args.shots,
                "bitstring": bitstring,
                "count": count,
                "max_results": args.max_results,
                "backend": args.backend,
                "alpha": args.alpha,
                "optimizer": getattr(args, 'optimizer', 'COBYLA'),
                "penalty_back": args.penalty_back,
                "penalty_chiral": args.penalty_chiral,
                "penalty_local_overlap": args.penalty_local_overlap,
                "analysis_mode": "global_candidate_pool",
                "optimization_convergence": {
                    "evaluation_counts": list(range(len(convergence_history))),
                    "cvar_energy_values": convergence_history,
                    "cumulative_shots": cumulative_shots_history,
                    "iteration_shots": iteration_shots_history
                },
                "xyz_coordinates": xyz_coords
            }
            ResultHandler.save_result(result_dict, result_filename)
            print(f"    - 结果已保存到: {result_filename}")
            
            # 生成PDB文件
            try:
                pdb_filename = os.path.join(result_dir, f"structure_rank_{idx+1}_energy_{energy_str}.pdb")
                ResultHandler.convert_xyz_to_detailed_pdb(result.protein_shape_file_gen.get_xyz_data(), pdb_filename)
                print(f"    - PDB文件已保存到: {pdb_filename}")
            except Exception as e:
                print(f"    - PDB文件生成失败: {e}")
            
            # 生成蛋白质结构图
            try:
                structure_plot_filename = os.path.join(result_dir, f"structure_rank_{idx+1}_energy_{energy_str}.png")
                ResultHandler.plot_protein_structure_3d(result, structure_plot_filename, 
                                                              title=f"Result {idx+1} (E={energy_value:.4f})")
                print(f"    - 结构图已保存到: {structure_plot_filename}")
            except Exception as e:
                print(f"    - 结构图生成失败: {e}")
        except Exception as e:
            print(f"    - 蛋白质结构解析失败: {e}")
            import traceback
            traceback.print_exc()

    # =============================================================================
    # 3.10 生成可视化图表
    # =============================================================================
    
    # 生成VQE优化摘要图
    try:
        print(f"\n正在生成VQE优化过程摘要图...")
        optimization_summary_filename = os.path.join(result_dir, "vqe_optimization_summary.png")
        ResultHandler.plot_vqe_optimization_summary(all_conv_data, optimization_summary_filename, args.main_chain)
        print(f"✓ VQE优化摘要图已保存到: {optimization_summary_filename}")
    except Exception as e:
        print(f"⚠ 生成VQE优化摘要图时出错: {e}")



    # 生成VQE收敛曲线图（带shots信息）
    try:
        print(f"\n正在生成VQE收敛曲线图（带shots信息）...")
        convergence_with_shots_filename = os.path.join(result_dir, "vqe_sampler_optimization_with_shots.png")
        iteration_shots = []
        for iter_data in processed_iteration_results:
            iteration_shots.append(iter_data.get("actual_shots", 0))
        ResultHandler.plot_vqe_convergence_with_shots(all_conv_data, iteration_shots, 
                                                              convergence_with_shots_filename, args.main_chain)
        print(f"✓ VQE收敛曲线图已保存到: {convergence_with_shots_filename}")
    except Exception as e:
        print(f"⚠ 生成VQE收敛曲线图时出错: {e}")

    # =============================================================================
    # 6. 导出最优参数向量（供 Warm-Start 使用）
    # =============================================================================
    try:
        print(f"\n正在导出最优参数向量...")
        # Sampler 优化器返回的是 scipy.optimize.OptimizeResult，最优参数在 res.x 中
        best_params = res.x
        
        best_params_file = os.path.join(result_dir, "best_params.json")
        with open(best_params_file, 'w') as f:
            json.dump({
                "initial_point": best_params.tolist(),
                "energy": best_energy,
                "optimizer": args.optimizer,
                "num_parameters": len(best_params)
            }, f, indent=2)
        print(f"✓ 最优参数已保存到: {best_params_file}")
        print(f"  - 参数维度: {len(best_params)}")
        print(f"  - 最优能量: {best_energy:.4f}")
        print(f"  提示: 可使用 --initial_params {best_params_file} 进行 Warm-Start")
    except Exception as e:
        print(f"⚠ 导出最优参数时出错: {e}")

    # =============================================================================
    # 3.11 任务完成
    # =============================================================================
    print(f"\n✓ 所有任务完成！结果保存在: {result_dir}")


if __name__ == "__main__":
    main()
