#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从protein_folding.ipynb提取的蛋白质折叠算法脚本
"""

import argparse
import os
import sys
import warnings
import datetime
import json

# 设置UTF-8环境以解决Windows编码问题
if os.name == 'nt':  # Windows系统
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
else:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

warnings.filterwarnings('ignore')

# ====================
# 可控参数配置区域
# ====================
parser = argparse.ArgumentParser(description='量子计算参数')
parser.add_argument('--backend', default='local', help='量子后端名称 (默认: local)')
parser.add_argument('--random_seed', type=int, default=23, help='随机种子 (默认: 23)')
parser.add_argument('--max_optimization_iterations', type=int, default=50, help='最大优化迭代次数 (默认: 50)')
parser.add_argument('--ansatz_reps', type=int, default=1, help='变分电路重复次数 (默认: 1)')
parser.add_argument('--main_chain', default='APRLRFY', help='主链序列 (默认: APRLRFY)')
parser.add_argument('--penalty_back', type=float, default=10, help='几何约束惩罚参数 (默认: 10)')
parser.add_argument('--penalty_chiral', type=float, default=10, help='手性约束惩罚参数 (默认: 10)')
parser.add_argument('--penalty_local_overlap', type=float, default=10, help='局部重叠惩罚参数 (默认: 10)')
parser.add_argument('--save_plot_png', type=bool, default=True, help='是否保存PNG格式图形 (默认: True)')
parser.add_argument('--save_plot_pdf', type=bool, default=False, help='是否保存PDF格式图形 (默认: True)')
parser.add_argument('--plot_dpi', type=int, default=300, help='PNG图形分辨率 (默认: 300)')
parser.add_argument('--shots', type=int, default=1000, help='量子电路采样次数 (默认: 1000)')
parser.add_argument('--max_results', type=int, default=1, help='输出最优结果的数量 (默认: 1, 最大值: 5)')
args = parser.parse_args()

# 量子计算参数
#QUANTUM_BACKEND = os.getenv('QUANTUM_BACKEND', 'local')  # 'local', 'aws_sv1', 'aws_garnet'
QUANTUM_BACKEND = args.backend
RANDOM_SEED = args.random_seed
MAX_OPTIMIZATION_ITERATIONS = args.max_optimization_iterations
ANSATZ_REPS = args.ansatz_reps

# 蛋白质结构参数
MAIN_CHAIN = args.main_chain
SIDE_CHAINS = [""] * len(MAIN_CHAIN)  # 侧链序列（本例中不考虑侧链）

# 物理约束参数
PENALTY_BACK = args.penalty_back
PENALTY_CHIRAL = args.penalty_chiral
PENALTY_LOCAL_OVERLAP = args.penalty_local_overlap

# 输出设置
SAVE_PLOT_PNG = args.save_plot_png
SAVE_PLOT_PDF = args.save_plot_pdf
PLOT_DPI = args.plot_dpi
SHOTS = args.shots
MAX_RESULTS = min(args.max_results, 5)  # 最多输出5个结果

# 创建结果目录
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = os.path.join("results", f"{TIMESTAMP}_{QUANTUM_BACKEND}")
os.makedirs(RESULT_DIR, exist_ok=True)
print(f"✓ 结果将保存在目录: {RESULT_DIR}")

# ====================
# 参数配置结束
# ====================

def main():
    print("正在启动蛋白质折叠算法...")
    
    # 导入必要的模块
    try:
        from src.protein_folding.interactions.random_interaction import RandomInteraction
        from src.protein_folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
        from src.protein_folding.peptide.peptide import Peptide
        from src.protein_folding.protein_folding_problem import ProteinFoldingProblem
        from src.protein_folding.penalty_parameters import PenaltyParameters
        from qiskit_algorithms.utils import algorithm_globals
        from qiskit.circuit.library import RealAmplitudes
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit_algorithms.minimum_eigensolvers import SamplingVQE
        from qiskit_aer.primitives import Sampler, SamplerV2
        print("✓ 成功导入蛋白质折叠模块")
    except ImportError as e:
        print(f"✗ 导入模块失败: {e}")
        print("请确保已正确安装src.protein_folding包")
        return False
    
    # 设置随机种子
    algorithm_globals.random_seed = RANDOM_SEED
    print("✓ 随机种子设置完成")
    
    # 定义蛋白质主链
    print("\n正在定义蛋白质结构...")
    main_chain = MAIN_CHAIN
    print(f"✓ 主链序列: {main_chain}")
    
    # 定义侧链
    side_chains = SIDE_CHAINS  # 本例中不考虑侧链
    print(f"✓ 侧链序列: {side_chains}")
    
    # 创建相互作用模型
    print("\n正在创建相互作用模型...")
    mj_interaction = MiyazawaJerniganInteraction()
    print("✓ Miyazawa-Jernigan相互作用模型创建完成")
    
    # 定义惩罚参数
    print("\n正在设置物理约束参数...")
    penalty_back = PENALTY_BACK
    penalty_chiral = PENALTY_CHIRAL
    penalty_1 = PENALTY_LOCAL_OVERLAP
    penalty_terms = PenaltyParameters(penalty_chiral, penalty_back, penalty_1)
    print(f"✓ 惩罚参数设置完成: chiral={penalty_chiral}, back={penalty_back}, local_overlap={penalty_1}")
    
    # 创建肽对象
    print("\n正在创建肽对象...")
    peptide = Peptide(main_chain, side_chains)
    print("✓ 肽对象创建完成")
    
    # 创建蛋白质折叠问题
    print("\n正在构建蛋白质折叠问题...")
    protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)
    qubit_op = protein_folding_problem.qubit_op()
    print(f"✓ 量子比特算子构建完成: {qubit_op}")
    
    # 使用VQE算法求解
    print("\n正在使用VQE算法求解...")
    try:
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit.circuit.library import RealAmplitudes
        from qiskit_algorithms.minimum_eigensolvers import SamplingVQE
        from qiskit_aer.primitives import Sampler, SamplerV2
        
        # 设置经典优化器
        optimizer = COBYLA(maxiter=MAX_OPTIMIZATION_ITERATIONS)
        print("✓ 优化器设置完成")
        
        # 设置变分试验波函数
        ansatz = RealAmplitudes(num_qubits = qubit_op.num_qubits, reps=ANSATZ_REPS)
        ansatz_reps = ansatz.reps

        # 关键：将蓝图电路转换为具体电路
        ansatz = ansatz.decompose()
        print("✓ 变分波函数设置完成")
        
        # 存储中间结果
        counts = []
        values = []
        
        def store_intermediate_result(eval_count, parameters, mean, std):
            counts.append(eval_count)
            values.append(mean)
        
        # 根据环境变量选择后端
        quantum_backend = QUANTUM_BACKEND  # 使用配置的量子后端
        print(f"当前qb:{quantum_backend}")
        if quantum_backend.lower() == 'aws_sv1':
            # 使用AWS SV1模拟器
            try:
                from qiskit_braket_provider import BraketLocalBackend, BraketProvider, BraketSampler
                #PROVIDER = BraketProvider()
                #backend = PROVIDER.get_backend('SV1')
                sampler = BraketSampler(BraketLocalBackend())
                #sampler = BraketSampler(
                #    backend=backend,
                #    options={"default_shots": 100}  # 可以设置 shots 数量
                #)
            except Exception as e:
                print(f"  - AWS连接失败，使用本地模拟器: {e}")
                return
        elif quantum_backend.lower() == 'ibm':
            # 使用IBM服务
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
                service = QiskitRuntimeService()
                backend = service.least_busy(operational=True, simulator=False)
                sampler = Sampler(mode=backend)
            except Exception as e:
                print(f"  - IBM连接失败，使用本地模拟器: {e}")
                return
        else:
            from qiskit_aer import Aer 
            backend = Aer.get_backend('qasm_simulator')
            # 尝试使用BackendSamplerV2，如果不可用则使用Sampler
            try:
                from qiskit.primitives import BackendSamplerV2
                sampler = BackendSamplerV2(
                    backend=backend,
                    options={"default_shots": SHOTS}  # 使用命令行参数设置 shots 数量
                )
            except ImportError:
                from qiskit.primitives import Sampler
                sampler = Sampler(
                    backend=backend,
                    options={"default_shots": SHOTS}  # 使用命令行参数设置 shots 数量
                )
            print("  - 使用本地Qiskit模拟器")

        print(f"Ansatz type: {type(ansatz)}")
        print(f"Ansatz num_qubits: {ansatz.num_qubits}")
        print(f"Ansatz num_parameters: {ansatz.num_parameters}")
        print(f"Sampler type: {type(sampler)}") 

        # 初始化VQE
        vqe = SamplingVQE(
            sampler=sampler,
            ansatz=ansatz,
            optimizer=optimizer,
            aggregation=0.1,
            callback=store_intermediate_result,
        )
        print("✓ VQE初始化完成")
        
        # 计算多个最优结果
        print(f"正在计算最多 {MAX_RESULTS} 个最优结果...")
        print(f"  - 使用优化器: {type(optimizer).__name__}")
        print(f"  - 最大迭代次数: {MAX_OPTIMIZATION_ITERATIONS}")
        print(f"  - 量子电路重复次数: {ansatz_reps}")
        print(f"  - 量子采样次数: {SHOTS}")
        
        # 存储多个结果
        all_results = []
        all_energies = []
        
        # 运行多次VQE以获取多个结果
        for i in range(MAX_RESULTS):
            print(f"  - 正在计算第 {i+1}/{MAX_RESULTS} 个结果...")
            
            # 为每次运行设置不同的随机种子
            algorithm_globals.random_seed = RANDOM_SEED + i
            
            # 重新初始化VQE以确保每次运行不同
            vqe = SamplingVQE(
                sampler=sampler,
                ansatz=ansatz,
                optimizer=optimizer,
                aggregation=0.1,
                callback=store_intermediate_result,
            )
            
            # 计算最小特征值
            raw_result = vqe.compute_minimum_eigenvalue(qubit_op)
            
            # 解释结果
            result = protein_folding_problem.interpret(raw_result=raw_result)
            
            # 存储结果和能量
            all_results.append(result)
            all_energies.append(raw_result.eigenvalue.real)
            
            print(f"  - 第 {i+1} 个结果能量: {raw_result.eigenvalue.real:.6f}")
        
        print("✓ 多结果计算完成")
        
        # 显示量子计算统计信息
        print(f"  - 量子电路深度: {ansatz.decompose().depth()}")
        print(f"  - 量子比特数量: {ansatz.num_qubits}")
        print(f"  - 量子门数量: {ansatz.decompose().size()}")
        print(f"  - 总函数评估次数: {len(counts)}")
        
        # 绘制VQE优化过程的折线图（如果环境支持）
        try:
            import matplotlib.pyplot as plt
            
            print("\n正在生成VQE优化过程的折线图...")
            
            # 创建图形
            fig = plt.figure(figsize=(10, 6))
            plt.plot(counts, values)
            plt.ylabel("Conformation Energy")
            plt.xlabel("VQE Iterations")
            plt.title("VQE Optimization Process")
            
            # 添加子图显示后期迭代的细节
            if len(counts) > 40:
                inset_ax = fig.add_axes([0.44, 0.51, 0.44, 0.32])
                inset_ax.plot(counts[40:], values[40:])
                inset_ax.set_ylabel("Conformation Energy")
                inset_ax.set_xlabel("VQE Iterations")
            
            # 保存优化过程图
            if SAVE_PLOT_PNG:
                plot_path = os.path.join(RESULT_DIR, f'vqe_optimization_process_{QUANTUM_BACKEND}.png')
                plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
                print(f"✓ VQE优化过程图已保存为 {plot_path} (DPI: {PLOT_DPI})")
            
            if SAVE_PLOT_PDF:
                plot_path = os.path.join(RESULT_DIR, f'vqe_optimization_process_{QUANTUM_BACKEND}.pdf')
                plt.savefig(plot_path, bbox_inches='tight')
                print(f"✓ VQE优化过程图已保存为 {plot_path}")
            
            # 在非交互式环境中，我们只保存图片而不显示
            plt.close()  # 关闭图形以释放内存
            print("✓ VQE优化过程折线图生成完成")
            
        except ImportError:
            print("⚠ 无法生成VQE优化过程折线图 (缺少matplotlib依赖)")
        except Exception as e:
            print(f"⚠ 生成VQE优化过程折线图时出错: {e}")
        
        # 处理并保存所有结果
        for i, (result, energy) in enumerate(zip(all_results, all_energies)):
            print(f"\n正在处理第 {i+1} 个结果 (能量: {energy:.6f})...")
            
            # 解释结果
            print(f"✓ 第 {i+1} 个蛋白质形状解码完成")
            print(f"  - 折叠蛋白的主链转向序列: {result.protein_shape_decoder.main_turns}")
            print(f"  - 侧链转向序列: {result.protein_shape_decoder.side_turns}")
            print(f"  - 代表蛋白质形状的比特串: {result.turn_sequence}")
            
            # 获取笛卡尔坐标
            print(f"\n正在获取第 {i+1} 个结果的蛋白质的笛卡尔坐标...")
            xyz_data = result.protein_shape_file_gen.get_xyz_data()
            print(f"✓ 第 {i+1} 个结果的坐标数据获取完成")
            print("前几行坐标数据:")
            for j, row in enumerate(xyz_data[:10]):
                line = ' '.join(map(str, row))
                if line.strip():
                    print(f"  {line}")
            
            # 显示并保存蛋白质结构的3D图形（如果环境支持）
            try:
                print(f"\n正在生成第 {i+1} 个结果的蛋白质结构的3D图形...")
                fig = result.get_figure(title=f"Protein Structure - Result {i+1} (Energy: {energy:.4f})", ticks=False, grid=True)
                
                # 检查Figure对象是否有效
                if hasattr(fig, 'get_axes') and len(fig.get_axes()) > 0:
                    fig.get_axes()[0].view_init(10, 70)
                
                # 保存图形为文件（根据配置）
                import matplotlib.pyplot as plt
                if SAVE_PLOT_PNG:
                    plot_path = os.path.join(RESULT_DIR, f'protein_structure_{i+1}_energy_{energy:.4f}.png')
                    fig.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
                    print(f"✓ 第 {i+1} 个结果的3D图形已保存为 {plot_path} (DPI: {PLOT_DPI})")
                
                if SAVE_PLOT_PDF:
                    plot_path = os.path.join(RESULT_DIR, f'protein_structure_{i+1}_energy_{energy:.4f}.pdf')
                    fig.savefig(plot_path, bbox_inches='tight')
                    print(f"✓ 第 {i+1} 个结果的3D图形已保存为 {plot_path}")
                
                # 关闭图形以释放内存
                plt.close(fig)
                
            except ImportError:
                print(f"⚠ 无法生成第 {i+1} 个结果的3D图形 (缺少matplotlib依赖)")
            except Exception as e:
                print(f"⚠ 保存第 {i+1} 个结果的图形时出错: {e}")
            
            # 保存结果的核心参数为JSON文件
            try:
                result_data = {
                    "result_index": i+1,
                    "energy": energy,
                    "main_turns": result.protein_shape_decoder.main_turns,
                    "side_turns": result.protein_shape_decoder.side_turns,
                    "turn_sequence": result.turn_sequence,
                    "main_chain_sequence": MAIN_CHAIN,
                    "side_chain_sequences": SIDE_CHAINS,
                    "penalty_parameters": {
                        "penalty_chiral": PENALTY_CHIRAL,
                        "penalty_back": PENALTY_BACK,
                        "penalty_local_overlap": PENALTY_LOCAL_OVERLAP
                    },
                    "quantum_parameters": {
                        "backend": QUANTUM_BACKEND,
                        "random_seed": RANDOM_SEED + i,
                        "max_optimization_iterations": MAX_OPTIMIZATION_ITERATIONS,
                        "ansatz_reps": ANSATZ_REPS,
                        "shots": SHOTS
                    },
                    "xyz_coordinates": [list(row) for row in result.protein_shape_file_gen.get_xyz_data()]
                }
                
                json_path = os.path.join(RESULT_DIR, f'result_{i+1}_energy_{energy:.4f}_parameters.json')
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, ensure_ascii=False, indent=2)
                print(f"✓ 第 {i+1} 个结果的核心参数已保存为 {json_path}")
                
            except Exception as e:
                print(f"⚠ 保存第 {i+1} 个结果的JSON参数时出错: {e}")
        
        print("\n✓ 蛋白质折叠计算完成！")
        return True
        
    except Exception as e:
        print(f"✗ VQE计算过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 设置环境变量以解决编码问题
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    success = main()
    if success:
        print("\n🎉 蛋白质折叠模拟运行成功！")
    else:
        print("\n❌ 蛋白质折叠模拟运行失败。")
        sys.exit(1)
