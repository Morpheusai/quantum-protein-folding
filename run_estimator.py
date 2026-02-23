#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
蛋白质折叠量子算法 - Estimator模式

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
import json

# 设置路径和编码环境
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.name == 'nt':
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'

sys.path.insert(0, os.path.join(current_dir, 'src'))
sys.path.insert(0, os.path.join(current_dir, 'lib'))

# 配置matplotlib和警告设置
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

# 导入量子计算相关库
from qiskit_algorithms.utils import algorithm_globals
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit.primitives import BackendSamplerV2
from qiskit import transpile

# 导入自定义模块
from lib.quantum_backend_manager import QuantumBackendManager
from lib.protein_folding_builder import ProteinFoldingBuilder
from lib.result_handler import ResultHandler
from lib.quantum_optimizer import QuantumOptimizer
import inspect
print(f"DEBUG: QuantumOptimizer file: {inspect.getfile(QuantumOptimizer)}")
from lib.job_metadata_logger import JobMetadataLogger

# =============================================================================
# 命令行参数配置
# =============================================================================
parser = argparse.ArgumentParser(description='蛋白质折叠量子算法 - Estimator模式')
parser.add_argument('--backend', default='local', 
                   help='量子后端: local (本地模拟器), local_aer (强制AerSimulator), aws_sv1, aws_garnet, aws_ionq, aws_forte, ibm, ibm_simulator')
parser.add_argument('--random_seed', type=int, default=23, help='随机种子')
parser.add_argument('--max_optimization_iterations', type=int, default=10, help='最大优化迭代次数')
parser.add_argument('--ansatz_reps', type=int, default=1, help='Ansatz重复次数')
parser.add_argument('--main_chain', default='APRLRFY', help='主链氨基酸序列')
parser.add_argument('--penalty_back', type=float, default=10, help='几何约束惩罚系数')
parser.add_argument('--penalty_chiral', type=float, default=10, help='手性约束惩罚系数')
parser.add_argument('--penalty_local_overlap', type=float, default=10, help='局部重叠惩罚系数')
parser.add_argument('--shots', type=int, default=100, help='量子采样次数')
parser.add_argument('--dry_run', action='store_true', help='干跑模式：仅在真实提交前进行本地预检')
parser.add_argument('--restarts', type=int, default=1, help='多重启实验次数 (Multi-Restart)')
parser.add_argument('--max_results', type=int, default=5, help='最终展示的最大不同结构数量')
parser.add_argument('--unique_structures', action='store_true', default=True, help='是否执行全局结构去重')
parser.add_argument('--adaptive_shots', action='store_true', help='是否启用自适应精度/采样')
parser.add_argument('--max_shots', type=int, default=1000, help='自适应采样最大值')
parser.add_argument('--aws_region', default=None, help='AWS区域')
parser.add_argument('--optimizer', default='COBYLA', help='优化器选择: COBYLA (默认), SPSA, SLSQP')
parser.add_argument('--resilience_level', type=int, default=1, help='IBM Quantum 误差抑制等级 (0-3)，默认1')
parser.add_argument('--resume', type=str, default=None, help='断点续传：指定结果目录以恢复历史任务')
parser.add_argument('--initial_params', type=str, default=None, help='Warm-Start：从 JSON 文件加载初始参数向量')
args = parser.parse_args()


def run_vqe_iteration(qubit_op, ansatz, optimizer, estimator, backend=None, result_dir=None, problem=None, sampler=None):
    """
    执行单次VQE迭代
    
    Args:
        qubit_op: 量子比特哈密顿量算子
        ansatz: 变分量子线路
        optimizer: 优化器
        estimator: 量子估算器
        backend: 量子后端（可选）
        result_dir: 结果目录（可选，用于保存每步迭代结果）
        problem: 蛋白质折叠问题对象（可选）
        sampler: 量子采样器（可选）
        
    Returns:
        tuple: (VQE结果, 收敛数据字典)
    """
    return QuantumOptimizer.create_vqe_optimizer(
        qubit_op, ansatz, optimizer, estimator, backend, args, 
        result_dir=result_dir, problem=problem, sampler=sampler
    )


# 初始化作业元数据记录器
metadata_logger = JobMetadataLogger("protein_folding_jobs_detailed.csv")


@metadata_logger
def main():
    """
    主函数：执行蛋白质折叠量子计算
    
    使用VQE算法和Estimator模式优化蛋白质折叠能量
    """
    print("正在启动蛋白质折叠算法...")
    print(f"🚀 启动服务器计算任务 | 序列: {args.main_chain}")
    
    # =========================================================================
    # 1. 参数配置显示
    # =========================================================================
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
    print(f"  - 优化器: {args.optimizer}")
    print(f"  - 误差抑制等级: {args.resilience_level}")

    # =========================================================================
    # 2. 初始化设置
    # =========================================================================
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

    # =========================================================================
    # 3. 构建蛋白质折叠问题
    # =========================================================================
    print(f"\n正在构建蛋白质折叠问题...")
    builder = ProteinFoldingBuilder(
        args.main_chain,
        penalty_chiral=args.penalty_chiral,
        penalty_back=args.penalty_back,
        penalty_local_overlap=args.penalty_local_overlap
    )
    qubit_op = builder.get_qubit_operator()
    problem = builder.get_problem()
    print(f"✓ 量子比特算子构建完成，量子比特数: {qubit_op.num_qubits}")

    # =========================================================================
    # 4. 配置量子后端和优化器
    # =========================================================================
    print(f"\n正在配置量子后端...")
    backend_info = QuantumBackendManager.setup_backend(args.backend, args.aws_region, args.shots, use_estimator=True, resilience_level=args.resilience_level)
    
    # 根据参数创建优化器
    if args.optimizer.upper() == 'SPSA':
        from qiskit_algorithms.optimizers import SPSA
        optimizer = SPSA(maxiter=args.max_optimization_iterations)
    elif args.optimizer.upper() == 'SLSQP':
        from qiskit_algorithms.optimizers import SLSQP
        optimizer = SLSQP(maxiter=args.max_optimization_iterations)
    else:
        optimizer = COBYLA(maxiter=args.max_optimization_iterations)
        
    print("✓ 优化器设置完成")
    print(f"  - 使用优化器: {type(optimizer).__name__}")
    print(f"  - 最大迭代次数: {args.max_optimization_iterations}")

    # =========================================================================
    # 5. 构建变分波函数（Ansatz）
    # =========================================================================
    base_ansatz = RealAmplitudes(num_qubits=qubit_op.num_qubits, reps=args.ansatz_reps)
    print("✓ 变分波函数设置完成")
    print(f"  - 量子比特数量: {base_ansatz.num_qubits}")
    print(f"  - 参数数量: {base_ansatz.num_parameters}")
    print(f"  - 电路重复次数: {args.ansatz_reps}")

    # =========================================================================
    # 6. 创建结果目录
    # =========================================================================
    result_dir = ResultHandler.create_result_directory("results", args.backend, mode='estimator')
    print(f"  - 结果目录: {result_dir}")

    all_conv_data = []
    global_candidates = [] # [(bitstring, energy, restart_index)]

    # =========================================================================
    # 7. 执行多轮VQE优化 (Multi-Restart)
    # =========================================================================
    print(f"\n正在执行 {args.restarts} 轮独立实验 (Multi-Restart)...")
    print(f"  - 每轮量子基础采样次数: {args.shots}")
    if args.adaptive_shots:
        print(f"  - 已启用自适应采样: {args.min_shots} -> {args.max_shots}")
    
    best_overall_energy = float('inf')
    best_restart_idx = -1

    for i in range(args.restarts):
        print(f"\n============================================================")
        print(f"--- 实验 {i+1}/{args.restarts} (Multi-Restart) ---")
        algorithm_globals.random_seed = args.random_seed + i
        print(f"随机种子: {algorithm_globals.random_seed}")
        
        # 7.1 执行VQE迭代
        # 为每轮实验创建子目录
        trial_dir = os.path.join(result_dir, f"trial_{i+1}")
        os.makedirs(trial_dir, exist_ok=True)
        
        raw_result, conv_data = run_vqe_iteration(
            qubit_op, base_ansatz, optimizer, 
            backend_info['estimator'], backend_info.get('backend'),
            result_dir=trial_dir, problem=problem, sampler=backend_info.get('sampler_v2')
        )
        
        # 设置结果标签
        trial_label = f'Trial {i+1}'
        conv_data['label'] = trial_label
        all_conv_data.append(conv_data)
        
        current_energy = float(raw_result.eigenvalue)
        print(f"  [优化结束] 能量: {current_energy:.4f}")
        
        if current_energy < best_overall_energy:
            best_overall_energy = current_energy
            best_restart_idx = i + 1
            print(f"  ★ 发现新最佳实验: {i+1} (能量: {current_energy:.4f})")

        # 7.2 收集候选结构 (从 VQE 收敛过程中的 parameters 还原得到)
        # 我们可以选取该轮实验中的最佳点，也可以选取 all_params 中的高质量点
        print(f"    - 正在从最优参数还原结构候选...")
        try:
            # 获取该轮最佳候选和对应的 Ansatz
            trial_ansatz = conv_data.get('ansatz', base_ansatz)
            best_params_dict = conv_data.get('best_params_dict')
            
            # 使用正确的 Ansatz 和参数字典获取该参数下的 bitstrings
            if best_params_dict:
                best_circuit = trial_ansatz.assign_parameters(best_params_dict)
            else:
                best_params = conv_data.get('best_params', raw_result.optimal_point)
                best_circuit = trial_ansatz.assign_parameters(best_params)
            
            is_simulator = args.backend.lower() in ['local', 'aws_sv1', 'ibm_simulator']
            if is_simulator or backend_info.get('backend') is None:
                probs = Statevector.from_instruction(best_circuit).probabilities_dict()
                # 选取概率最高的一个或多个
                top_bitstring = max(probs, key=probs.get)
                global_candidates.append((top_bitstring, current_energy, i+1))
            else:
                # 硬件采样
                meas_circuit = transpile(best_circuit, backend=backend_info['backend'], optimization_level=3)
                meas_circuit.measure_all()
                sampler = BackendSamplerV2(backend=backend_info['backend'])
                job = sampler.run([meas_circuit], shots=max(100, args.shots))
                sampler_res = job.result()[0]
                counts = getattr(sampler_res.data, list(sampler_res.data.keys())[0]).get_counts()
                top_bitstring = max(counts, key=counts.get)
                global_candidates.append((top_bitstring, current_energy, i+1))
        except Exception as e:
            print(f"    ⚠ 提取候选结构失败: {e}")

    # =========================================================================
    # 8. 全局汇总与结构去重
    # =========================================================================
    print(f"\n============================================================")
    print(f"所有实验结束。最佳实验: {best_restart_idx} (能量: {best_overall_energy:.4f})")
    print(f"正在进行跨运行结果汇总与去重...")
    
    # 按能量排序
    global_candidates.sort(key=lambda x: x[1])
    
    final_top_results = []
    seen_structures = set()
    
    # 结构去重核心循环
    for bitstring, energy, restart_idx in global_candidates:
        if len(final_top_results) >= args.max_results:
            break
            
        if args.unique_structures:
            try:
                # 使用精确 3D 结构生成进行去重
                from lib.protein_geometry import ProteinGeometryBuilder
                
                builder = ProteinFoldingBuilder(args.main_chain)
                turn2qubit = builder.get_turn2qubit()
                geo_builder = ProteinGeometryBuilder(args.main_chain)
                
                # 使用精确 3D 结构生成
                atoms = geo_builder.build_3d_structure_from_bitstring(bitstring, turn2qubit)
                
                # 生成转向序列
                cfg_bits = bitstring[:turn2qubit.count('q')]
                config = geo_builder._fill_config_bits(cfg_bits, turn2qubit)
                turns = [int(config[k:k+2], 2) for k in range(0, len(config), 2)]
                struct_key = str(turns)
                
                if struct_key not in seen_structures:
                    seen_structures.add(struct_key)
                    final_top_results.append((bitstring, energy, restart_idx, turns))
            except Exception as e:
                print(f"    ⚠ 结构分析失败: {e}")
                continue
        else:
            final_top_results.append((bitstring, energy, restart_idx, None))

    print(f"    - 完成。共找到 {len(final_top_results)} 个独特候选结构。")
    print(f"    - 开始保存最终结果到 {result_dir}...")

    # =========================================================================
    # 9. 循环保存排名前 N 的独特结果
    # =========================================================================
    for idx, (bitstring, energy, restart_idx, turns) in enumerate(final_top_results):
        print(f"\n    - 结果 {idx+1}/{len(final_top_results)}: 能量 = {energy:.4f}, 来自实验 {restart_idx}")
        
        # 使用精确能量计算器获取能量分解
        from lib.precise_energy_calculator import PreciseEnergyCalculator
        from lib.protein_geometry import ProteinGeometryBuilder
        
        builder = ProteinFoldingBuilder(args.main_chain)
        interaction_matrix = builder.get_interaction_matrix()
        turn2qubit = builder.get_turn2qubit()
        
        # 精确能量计算
        calc = PreciseEnergyCalculator(args.main_chain, interaction_matrix)
        energy_breakdown = calc.calculate_energy_breakdown(bitstring, turn2qubit)
        
        print(f"    - 能量分解:")
        print(f"      Backbone: {energy_breakdown['backbone']:.4f}")
        print(f"      MJ: {energy_breakdown['mj']:.4f}")
        print(f"      Distance: {energy_breakdown['distance']:.4f}")
        print(f"      Locality: {energy_breakdown['locality']:.4f}")
        print(f"      Total: {energy_breakdown['total']:.4f}")
        
        # 精确 3D 结构生成
        geo_builder = ProteinGeometryBuilder(args.main_chain)
        atoms = geo_builder.build_3d_structure_from_bitstring(bitstring, turn2qubit)
        
        # 生成 XYZ 数据
        xyz_data = [[atom["name"], atom["coords"][0], atom["coords"][1], atom["coords"][2]] 
                    for atom in atoms]
        
        print(f"    - 蛋白质结构解析完成")
        print(f"    - 转向序列: {turns}")
        
        energy_str = f"{abs(energy):.4f}"
        
        # 准备结果字典
        result_filename = os.path.join(result_dir, f"result_{idx+1}_energy_{energy_str}.json")
        result_dict = {
            "result_index": idx + 1,
            "original_restart": restart_idx,
            "energy": energy,
            "energy_breakdown": energy_breakdown,
            "turn_sequence": turns,
            "main_chain_sequence": args.main_chain,
            "shots_requested": args.shots,
            "backend": args.backend,
            "optimization_convergence": {
                "evaluation_counts": all_conv_data[restart_idx-1]['counts'],
                "energy_values": all_conv_data[restart_idx-1]['values'],
                "stds": all_conv_data[restart_idx-1].get('stds', [])
            },
            "xyz_coordinates": xyz_data
        }
        ResultHandler.save_result(result_dict, result_filename)
        
        # 使用精确 PDB 文件生成
        pdb_filename = os.path.join(result_dir, f"structure_{idx+1}_energy_{energy_str}.pdb")
        geo_builder.write_pdb_file(atoms, bitstring, pdb_filename)
        print(f"    - PDB 文件已保存: {pdb_filename}")
        
        # 生成 3D 可视化
        png_filename = os.path.join(result_dir, f"structure_{idx+1}_energy_{energy_str}.png")
        ResultHandler.plot_protein_structure_3d_precise(atoms, args.main_chain, png_filename, 
                                                       f"Result {idx+1} (E={energy:.4f})")
        print(f"    - 3D 可视化已保存: {png_filename}")

    # =========================================================================
    # 9. 生成可视化图表
    # =========================================================================
    
    # 9.1 生成VQE优化摘要图
    try:
        print(f"\n正在生成VQE优化过程摘要图...")
        optimization_summary_filename = os.path.join(result_dir, "vqe_optimization_summary.png")
        ResultHandler.plot_vqe_optimization_summary(all_conv_data, optimization_summary_filename, args.main_chain)
        print(f"✓ VQE优化摘要图已保存到: {optimization_summary_filename}")
    except Exception as e:
        print(f"⚠ 生成VQE优化摘要图时出错: {e}")

    # 9.2 生成VQE收敛曲线图（带shots信息）
    try:
        print(f"\n正在生成VQE收敛曲线图（带shots信息）...")
        convergence_with_shots_filename = os.path.join(result_dir, "vqe_optimization_with_shots.png")
        # 只使用第一个VQE运行的shots，确保柱状图与能量曲线对齐
        iteration_shots = []
        if all_conv_data and 'iteration_shots' in all_conv_data[0]:
            iteration_shots = all_conv_data[0]['iteration_shots']
        ResultHandler.plot_vqe_convergence_with_shots(all_conv_data, iteration_shots, 
                                                              convergence_with_shots_filename, args.main_chain)
        print(f"✓ VQE收敛曲线图已保存到: {convergence_with_shots_filename}")
    except Exception as e:
        print(f"⚠ 生成VQE收敛曲线图时出错: {e}")

    # =========================================================================
    # 10. 导出最优参数向量（供 Warm-Start 使用）
    # =========================================================================
    try:
        print(f"\n正在导出最优参数向量...")
        # 优先从 VQE 结果中直接获取最优参数（这是最准确的）
        if hasattr(raw_result, 'optimal_point') and raw_result.optimal_point is not None:
            best_params = raw_result.optimal_point
        elif all_conv_data and 'best_params_dict' in all_conv_data[best_restart_idx - 1]:
            # 如果只有参数字典，转换为数组
            best_params_dict = all_conv_data[best_restart_idx - 1]['best_params_dict']
            best_params = np.array(list(best_params_dict.values()))
        elif all_conv_data and 'best_params' in all_conv_data[best_restart_idx - 1]:
            # 最后才使用回调函数中的参数（可能不完整）
            best_params = all_conv_data[best_restart_idx - 1]['best_params']
        else:
            raise ValueError("无法获取最优参数")
        
        best_params_file = os.path.join(result_dir, "best_params.json")
        with open(best_params_file, 'w') as f:
            json.dump({
                "initial_point": best_params.tolist(),
                "energy": best_overall_energy,
                "optimizer": args.optimizer,
                "num_parameters": len(best_params)
            }, f, indent=2)
        print(f"✓ 最优参数已保存到: {best_params_file}")
        print(f"  - 参数维度: {len(best_params)}")
        print(f"  - 最优能量: {best_overall_energy:.4f}")
        print(f"  提示: 可使用 --initial_params {best_params_file} 进行 Warm-Start")
    except Exception as e:
        print(f"⚠ 导出最优参数时出错: {e}")

    print(f"\n✓ 所有任务完成！结果保存在: {result_dir}")
    
    try:
        metrics_path = os.path.join(result_dir, "metrics.json")
        total_shots = 0
        total_iters = 0
        try:
            for d in all_conv_data:
                if isinstance(d, dict) and 'iteration_shots' in d and isinstance(d['iteration_shots'], list):
                    total_shots += sum(int(x) for x in d['iteration_shots'])
                    total_iters += len(d['iteration_shots'])
        except:
            pass
        try:
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
