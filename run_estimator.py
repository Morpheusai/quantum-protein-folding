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
parser.add_argument('--max_results', type=int, default=1, help='最大结果数量')
parser.add_argument('--aws_region', default=None, help='AWS区域')
args = parser.parse_args()


def run_vqe_iteration(qubit_op, ansatz, optimizer, estimator, backend=None):
    """
    执行单次VQE迭代
    
    Args:
        qubit_op: 量子比特哈密顿量算子
        ansatz: 变分量子线路
        optimizer: 优化器
        estimator: 量子估算器
        backend: 量子后端（可选）
        
    Returns:
        tuple: (VQE结果, 收敛数据字典)
    """
    return QuantumOptimizer.create_vqe_optimizer(qubit_op, ansatz, optimizer, estimator, backend, args)


# 初始化作业元数据记录器
metadata_logger = JobMetadataLogger("protein_folding_jobs.csv")


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

    # =========================================================================
    # 2. 初始化设置
    # =========================================================================
    algorithm_globals.random_seed = args.random_seed
    print("✓ 随机种子设置完成")

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
    backend_info = QuantumBackendManager.setup_backend(args.backend, args.aws_region, args.shots, use_estimator=True)
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

    # =========================================================================
    # 7. 执行多轮VQE优化
    # =========================================================================
    print(f"\n正在计算最多 {args.max_results} 个最优结果...")
    print(f"  - 量子采样次数: {args.shots}")
    
    for i in range(args.max_results):
        print(f"  - 正在计算第 {i+1}/{args.max_results} 个结果...")
        algorithm_globals.random_seed = args.random_seed + i
        
        print(f"    - 当前随机种子: {algorithm_globals.random_seed}")
        
        # 7.1 执行VQE迭代
        raw_result, conv_data = run_vqe_iteration(
            qubit_op, base_ansatz, optimizer, 
            backend_info['estimator'], backend_info.get('backend')
        )
        # 设置结果标签
        conv_data['label'] = f'run{i+1}'
        all_conv_data.append(conv_data)
        
        # 7.2 分析最优参数并提取蛋白质结构
        print(f"    - 正在分析最优参数以提取蛋白质结构...")
        try:
            best_circuit = base_ansatz.assign_parameters(raw_result.optimal_point)
            
            is_simulator = args.backend.lower() in ['local', 'aws_sv1', 'ibm_simulator']
            
            if is_simulator or backend_info.get('backend') is None:
                # 模拟器模式：使用Statevector直接计算概率分布
                raw_result.eigenstate = Statevector.from_instruction(best_circuit).probabilities_dict()
                print(f"    - 通过模拟还原概率分布成功")
            else:
                # 真实硬件模式：通过采样获取概率分布
                print(f"    - 正在通过真实量子硬件采样...")
                meas_circuit = transpile(best_circuit, backend=backend_info['backend'],
                                       initial_layout=list(range(best_circuit.num_qubits)) if args.backend not in ['local', 'aws_sv1', 'ibm_simulator'] else None,
                                       optimization_level=3)
                print(f"   结果电路转译: 逻辑比特数={meas_circuit.num_qubits}, 物理比特数={meas_circuit.width()}")
                meas_circuit.measure_all()
                
                sampler = BackendSamplerV2(backend=backend_info['backend'])
                job = sampler.run([meas_circuit], shots=args.shots)
                sampler_res = job.result()[0]
                
                reg_name = list(sampler_res.data.keys())[0]
                counts = getattr(sampler_res.data, reg_name).get_counts()
                total_shots = sum(counts.values())
                raw_result.eigenstate = {k: v/total_shots for k, v in counts.items()}
                print(f"    - 硬件采样完成，共 {total_shots} 次测量")
        except Exception as e:
            print(f"    - 概率分布还原失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        print(f"    - 最优能量值: {raw_result.eigenvalue:.4f}")
        
        # 7.3 解析蛋白质结构
        result = problem.interpret(raw_result)
        print(f"    - 蛋白质结构解析完成")
        protein_structure = "".join(result.protein_shape_file_gen.main_chain_aminoacid_list)
        print(f"    - 蛋白质形状: {protein_structure}")
        print(f"    - 蛋白质能量: {raw_result.eigenvalue:.4f}")
        
        # =====================================================================
        # 8. 保存结果数据
        # =====================================================================
        
        # 8.1 准备结果数据
        xyz_data = result.protein_shape_file_gen.get_xyz_data()
        xyz_coords = [[str(row[0]), str(row[1]), str(row[2]), str(row[3])] for row in xyz_data]
        energy_value = float(raw_result.eigenvalue)
        energy_str = f"{abs(energy_value):.4f}"
        
        # 8.2 保存JSON结果文件
        result_filename = os.path.join(result_dir, f"result_{i+1}_energy_{energy_str}.json")
        result_dict = {
            "result_index": i + 1,
            "energy": energy_value,
            "turn_sequence": result.turn_sequence,
            "main_chain_sequence": protein_structure,
            "shots_requested": args.shots,
            "backend": args.backend,
            "tries": args.max_optimization_iterations,
            "optimizer": type(optimizer).__name__,
            "penalty_back": args.penalty_back,
            "penalty_chiral": args.penalty_chiral,
            "penalty_local_overlap": args.penalty_local_overlap,
            "optimization_convergence": {
                "evaluation_counts": conv_data['counts'],
                "energy_values": conv_data['values'],
                "cumulative_shots": conv_data['cumulative_shots']
            },
            "xyz_coordinates": xyz_coords
        }
        ResultHandler.save_result(result_dict, result_filename)
        print(f"    - 结果已保存到: {result_filename}")
        
        # 8.3 生成PDB文件
        try:
            pdb_filename = os.path.join(result_dir, f"structure_{i+1}_energy_{energy_str}.pdb")
            ResultHandler.convert_xyz_to_detailed_pdb(result.protein_shape_file_gen.get_xyz_data(), pdb_filename)
            print(f"    - PDB文件已保存到: {pdb_filename}")
        except Exception as e:
            print(f"    - PDB文件生成失败: {e}")
        
        # 8.4 生成蛋白质结构图
        try:
            structure_plot_filename = os.path.join(result_dir, f"structure_{i+1}_energy_{energy_str}.png")
            ResultHandler.plot_protein_structure_3d(result, structure_plot_filename, 
                                                              title=f"Result {i+1} (E={energy_value:.4f})")
            print(f"    - 结构图已保存到: {structure_plot_filename}")
        except Exception as e:
            print(f"    - 结构图生成失败: {e}")

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

    print(f"\n✓ 所有任务完成！结果保存在: {result_dir}")
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 蛋白质折叠模拟运行成功！")
    else:
        print("\n❌ 蛋白质折叠模拟运行失败。")
        sys.exit(1)
