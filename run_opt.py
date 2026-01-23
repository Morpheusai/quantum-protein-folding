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
import numpy as np
import copy
from job_metadata_logger import JobMetadataLogger

# 创建元数据记录器实例
metadata_logger = JobMetadataLogger("protein_folding_jobs.csv")


# 设置UTF-8环境以解决Windows编码问题并配置模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.name == 'nt':  # Windows系统
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'

sys.path.insert(0, os.path.join(current_dir, 'src'))

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
parser.add_argument('--max_results', type=int, default=1)
parser.add_argument('--aws_region', default=None)
args = parser.parse_args()

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = os.path.join("results", f"{TIMESTAMP}_{args.backend}")
os.makedirs(RESULT_DIR, exist_ok=True)



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
        return info
    except Exception as e:
        print(f"✗ 后端设置失败: {e}")
        from qiskit.primitives import StatevectorEstimator
        return {'backend': None, 'estimator': StatevectorEstimator()}

def run_vqe_iteration(qubit_op, ansatz, optimizer, estimator, backend=None):
    from qiskit_algorithms import VQE
    from qiskit import transpile

    # 【核心修复】使用深拷贝创建干净的Ansatz副本
    # 防止VQE算法在前次运行中添加的测量门污染当前实例
    # 确保每次迭代都有纯净的量子电路用于计算
    working_ansatz = copy.deepcopy(ansatz)
    
    # 彻底移除所有测量门，防止与 Estimator 的自动测量逻辑冲突
    working_ansatz.remove_final_measurements() 

    if backend is not None:
        # 在转译过程中不添加任何测量，保持电路纯净
        # 添加initial_layout参数以更好地处理AWS硬件限制
        working_ansatz = transpile(working_ansatz, backend=backend,
                                  initial_layout=list(range(working_ansatz.num_qubits)) if args.backend not in ['local', 'aws_sv1', 'ibm_simulator'] else None,
                                  optimization_level=3)
        print(f"   VQE电路转译: 逻辑比特数={working_ansatz.num_qubits}, 物理比特数={working_ansatz.width()}")

    convergence = {'counts': [], 'values': [], 'cumulative_shots': []}
    
    # 存储每次评估的实际shots数
    actual_shots_list = []
    
    def record_shots_from_metadata(metadata):
        """从元数据中提取实际执行的shots数"""
        actual_shots = 0
        if "shots_per_circuit" in metadata:
            actual_shots = sum(metadata["shots_per_circuit"])
        elif "execution" in metadata and "circuits" in metadata["execution"]:
            actual_shots = sum(c["shots"] for c in metadata["execution"]["circuits"])
        elif "shots" in metadata:
            shots_per_circuit = metadata["shots"]
            num_circuits = metadata.get("num_circuits", 1)
            actual_shots = shots_per_circuit * num_circuits
        return actual_shots
    
    def callback(eval_count, parameters, mean, std):
        convergence['counts'].append(eval_count)
        convergence['values'].append(mean)
        
        # 在实际应用中，每次评估可能涉及多个Pauli项，所以实际shots数可能高于设定值
        # 这里我们用一个更现实的估算：假设每次评估涉及多个量子电路执行
        # 具体数值取决于哈密顿量的Pauli项数量
        current_step_shots = args.shots  # 这是基础值，实际值会在VQE完成后修正
        actual_shots_list.append(current_step_shots)
        
        # 计算累计shots数
        cumulative_shots = sum(actual_shots_list)
        convergence['cumulative_shots'].append(cumulative_shots)

    # 使用处理后的 working_ansatz，现在使用 VQE 配合 estimator
    vqe = VQE(estimator=estimator, ansatz=working_ansatz, optimizer=optimizer, callback=callback)
    result = vqe.compute_minimum_eigenvalue(qubit_op)
    
    # 为了更准确地记录shots数，我们需要使用一个包装器来捕获estimator的调用
    # 但由于VQE内部的工作方式，我们只能在VQE结束后获取总的元数据
    if hasattr(result, 'metadata') and result.metadata:
        # 尝试从最终结果的元数据中获取总shots数
        total_shots_from_metadata = record_shots_from_metadata(result.metadata)
        if total_shots_from_metadata > 0:
            # 如果能从元数据获取到总shots数，则更新整个收敛过程中的shots数
            # 这是一个近似方法，因为无法知道每次评估的具体shots数
            # 更精确的实现需要自定义estimator或回调机制
            if len(convergence['cumulative_shots']) > 0:
                # 基于评估次数平均分配总shots数
                avg_shots_per_eval = max(1, total_shots_from_metadata // len(convergence['cumulative_shots']))
                convergence['cumulative_shots'] = [i * avg_shots_per_eval for i in range(1, len(convergence['cumulative_shots']) + 1)]
    
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

    print(f"\n正在计算最多 {args.max_results} 个最优结果...")
    print(f"  - 量子采样次数: {args.shots}")
    
    for i in range(args.max_results):
        print(f"  - 正在计算第 {i+1}/{args.max_results} 个结果...")
        algorithm_globals.random_seed = args.random_seed + i
        
        # 显示当前迭代的随机种子
        print(f"    - 当前随机种子: {algorithm_globals.random_seed}")
        
        # 执行VQE迭代计算
        # 将哈密顿量、Ansatz、优化器和估算器传递给VQE算法
        # 返回原始结果和收敛数据
        raw_result, conv_data = run_vqe_iteration(
            qubit_op, base_ansatz, optimizer, 
            backend_info['estimator'], backend_info.get('backend')
        )
        all_conv_data.append(conv_data)
        
        # 结果解析与验证
        # 【核心修复】VQE 使用 Estimator 时不返回 eigenstate (概率分布)
        # 必须根据最优参数还原比特串分布，否则 interpret 无法识别真实的蛋白质形状
        print(f"    - 正在分析最优参数以提取蛋白质结构...")
        try:
            # 1. 产生最优电路的一个干净副本 (此时不含测量门)
            best_circuit = base_ansatz.assign_parameters(raw_result.optimal_point)
            
            # 2. 策略：如果是模拟器 (local 或 aws_sv1)，本地计算 eigenstate 既快又准
            # 这样可以 100% 避免 AWS 后端可能出现的“重复测量”或“网络延迟”问题
            is_simulator = args.backend.lower() in ['local', 'aws_sv1', 'ibm_simulator']
            
            if is_simulator or backend_info.get('backend') is None:
                from qiskit.quantum_info import Statevector
                # 使用 Statevector 直接还原概率分布，结果与 SV1 跑出来的理论概率完全一致
                raw_result.eigenstate = Statevector.from_instruction(best_circuit).probabilities_dict()
                print(f"    - 通过模拟还原概率分布成功")
            else:
                # 3. 如果是真实量子硬件 (如 Garnet)，则必须进行实际采样
                from qiskit.primitives import BackendSamplerV2
                from qiskit import transpile
                
                # 重点：先转译不含测量的干净电路，再添加测量，防止测量冲突
                # 使用更好的转译策略处理硬件限制
                meas_circuit = transpile(best_circuit, backend=backend_info['backend'],
                                       initial_layout=list(range(best_circuit.num_qubits)) if args.backend not in ['local', 'aws_sv1', 'ibm_simulator'] else None,
                                       optimization_level=3)
                print(f"   结果电路转译: 逻辑比特数={meas_circuit.num_qubits}, 物理比特数={meas_circuit.width()}")
                meas_circuit.measure_all()
                
                sampler = BackendSamplerV2(backend=backend_info['backend'])
                job = sampler.run([meas_circuit], shots=args.shots)
                sampler_res = job.result()[0]
                
                # 提取计数 (适配 BackendSamplerV2 的 DataBin 格式)
                reg_name = list(sampler_res.data.keys())[0]
                counts = getattr(sampler_res.data, reg_name).get_counts()
                total_shots = sum(counts.values())
                raw_result.eigenstate = {k: v/total_shots for k, v in counts.items()}
                print(f"    - 通过硬件采样还原概率分布成功")
                
        except Exception as e:
            print(f"⚠ 提取蛋白质结构时出错: {e}")
            if not hasattr(raw_result, 'eigenstate'):
                raw_result.eigenstate = None
            
        if not hasattr(raw_result, 'eigenvalue'):
            raise ValueError("raw_result 缺少 eigenvalue 属性")
        
        result = problem.interpret(raw_result=raw_result)
        energy = float(raw_result.eigenvalue.real)
        print(f"    - 第 {i+1} 个结果能量: {energy:.6f}")
        
        # 解释结果
        print(f"正在处理第 {i+1} 个结果 (能量: {energy:.6f})...")
        print(f"✓ 第 {i+1} 个蛋白质形状解码完成")
        print(f"  - 折叠蛋白的主链转向序列: {result.protein_shape_decoder.main_turns}")
        print(f"  - 侧链转向序列: {result.protein_shape_decoder.side_turns}")
        print(f"  - 代表蛋白质形状的比特串: {result.turn_sequence}")
        
        # 提取坐标与保存数据
        print(f"\n正在获取第 {i+1} 个结果的蛋白质的笛卡尔坐标...")
        xyz_data = None
        try:
            xyz_data = result.protein_shape_file_gen.get_xyz_data()
            print(f"✓ 第 {i+1} 个结果的坐标数据获取完成")
            print("前几行坐标数据:")
            for j, row in enumerate(xyz_data[:5]):  # 只显示前5行
                line = ' '.join(map(str, row))
                if line.strip():
                    print(f"  {line}")
        except Exception as e:
            print(f"⚠ 获取坐标数据时出错: {e}")

        result_data = {
            "result_index": i + 1,
            "energy": energy,
            "main_turns": result.protein_shape_decoder.main_turns,
            "side_turns": result.protein_shape_decoder.side_turns,
            "turn_sequence": result.turn_sequence,
            "main_chain_sequence": args.main_chain,
            "sequence": args.main_chain,
            "shots_requested": args.shots,
            "optimization_convergence": {
                "evaluation_counts": conv_data['counts'],
                "energy_values": conv_data['values'],
                "cumulative_shots": conv_data['cumulative_shots']
            },
            "xyz_coordinates": [list(row) for row in xyz_data] if xyz_data is not None and len(xyz_data) > 0 else []
        }
        
        json_path = os.path.join(RESULT_DIR, f'result_{i+1}_energy_{energy:.4f}_parameters.json')
        with open(json_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        print(f"✓ 第 {i+1} 个结果的核心参数已保存为 {json_path}")

        # 保存 PDB 文件
        if xyz_data is not None:
            # 保存新版详细PDB文件（包含H、N等完整原子坐标）
            from src.protein_folding.utils.detailed_pdb_generator import convert_xyz_to_detailed_pdb
            detailed_pdb_path = os.path.join(RESULT_DIR, f'structure_{i+1}_energy_{energy:.4f}.pdb')
            convert_xyz_to_detailed_pdb(xyz_data, detailed_pdb_path, f"Detailed Structure {i+1} (E={energy:.4f})")
            print(f"✓ 第 {i+1} 个结果的详细PDB文件已保存为 {detailed_pdb_path}")

        # 绘图：单次结构
        try:
            print(f"\n正在生成第 {i+1} 个结果的蛋白质结构的3D图形...")
            fig_struct = result.get_figure(title=f"Result {i+1} (E={energy:.4f})")
            png_path = os.path.join(RESULT_DIR, f"structure_{i+1}_energy_{energy:.4f}.png")
            fig_struct.savefig(png_path)
            plt.close(fig_struct)
            print(f"✓ 第 {i+1} 个结果的3D图形已保存为 {png_path}")
        except Exception as e:
            print(f"⚠ 生成第 {i+1} 个结果的3D图形时出错: {e}")

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
            plt.plot(data['counts'], data['values'], marker='o', label=f'Run {idx+1}')
        plt.xlabel("Evaluation Counts")
        plt.ylabel("Energy")
        plt.title(f"VQE Convergence Comparison ({args.main_chain})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(RESULT_DIR, "vqe_optimization_summary.png"))
        plt.close()
        print(f"✓ VQE优化过程图已保存为 {os.path.join(RESULT_DIR, 'vqe_optimization_summary.png')}")
            
        # 双y轴图：展示能量和shots的关系（改进版）
        if all('cumulative_shots' in data and len(data['cumulative_shots']) > 0 for data in all_conv_data):
            fig, ax1 = plt.subplots(figsize=(12, 8))
            
            # 定义颜色映射，为每个运行结果使用相同颜色的不同样式
            colors = plt.cm.tab10(np.linspace(0, 1, len(all_conv_data)))
            
            # 绘制能量曲线
            energy_lines = []
            for idx, data in enumerate(all_conv_data):
                color = colors[idx]
                line, = ax1.plot(data['counts'], data['values'], marker='o', label=f'Run {idx+1}', 
                         linewidth=3, color=color)  # 增加线宽使能量曲线更突出
                energy_lines.append(line)
            ax1.set_xlabel('Evaluation Counts')
            ax1.set_ylabel('Energy', color='black')
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.grid(True, alpha=0.3)
            
            # 创建第二个y轴用于shots
            shots_bars = []
            ax2 = ax1.twinx()
            for idx, data in enumerate(all_conv_data):
                color = colors[idx]
                # 使用较浅的颜色和边框来减少柱状图的视觉冲击
                bars = ax2.bar(data['counts'], data['cumulative_shots'], alpha=0.95, width=0.3, 
                       color=color, edgecolor=color, linewidth=0.5, 
                       label=f'Run {idx+1} Cumulative Shots')
                shots_bars.append(bars)
            ax2.set_ylabel('Cumulative Shots', color='black')
            ax2.tick_params(axis='y', labelcolor='black')
            
            # 只使用能量曲线的图例，避免重复
            ax1.legend(energy_lines, [f'Run {idx+1}' for idx in range(len(energy_lines))], loc='upper left')
            
            plt.title(f"VQE Convergence with Cumulative Shots ({args.main_chain})")
            fig.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, "vqe_optimization_with_shots.png"))
            plt.close()
            print(f"✓ 带shots信息的VQE优化图已保存为 {os.path.join(RESULT_DIR, 'vqe_optimization_with_shots.png')}")
            
    except Exception as e:
        print(f"⚠ 生成VQE优化过程图时出错: {e}")
    

    print(f"\n✓ 蛋白质折叠计算完成！")
    print(f"🎉 任务完成。结果目录: {RESULT_DIR}")
    
    return True


if __name__ == "__main__":

    success = main()
    if success:
        print("\n🎉 蛋白质折叠模拟运行成功！")
    else:
        print("\n❌ 蛋白质折叠模拟运行失败。")
        sys.exit(1)