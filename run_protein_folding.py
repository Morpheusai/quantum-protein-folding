#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从protein_folding.ipynb提取的蛋白质折叠算法脚本
"""

import argparse
import os
import sys
import warnings

# 设置UTF-8环境以解决Windows编码问题
if os.name == 'nt':  # Windows系统
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
else:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

warnings.filterwarnings('ignore')

# ====================
# 可控参数配置区域
# ====================
parser = argparse.ArgumentParser(description='量子计算参数')
parser.add_argument('--backend', default='local', help='量子后端名称 (默认: local)')
args = parser.parse_args()

# 量子计算参数
#QUANTUM_BACKEND = os.getenv('QUANTUM_BACKEND', 'local')  # 'local', 'aws_sv1', 'aws_garnet'
QUANTUM_BACKEND = args.backend
RANDOM_SEED = 23  # 随机种子
MAX_OPTIMIZATION_ITERATIONS = 50  # 最大优化迭代次数
ANSATZ_REPS = 1  # 变分电路重复次数

# 蛋白质结构参数
MAIN_CHAIN = "APRLRFY"  # 主链序列
SIDE_CHAINS = [""] * 7  # 侧链序列（本例中不考虑侧链）

# 物理约束参数
PENALTY_BACK = 10  # 几何约束惩罚参数
PENALTY_CHIRAL = 10  # 手性约束惩罚参数
PENALTY_LOCAL_OVERLAP = 10  # 局部重叠惩罚参数

# 输出设置
SAVE_PLOT_PNG = True  # 是否保存PNG格式图形
SAVE_PLOT_PDF = True  # 是否保存PDF格式图形
PLOT_DPI = 300  # PNG图形分辨率

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
        from qiskit.primitives import BackendSamplerV2
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
            sampler = BackendSamplerV2(
                backend=backend,
                options={"default_shots": 1000}  # 可以设置 shots 数量
            )
            #sampler = SamplerV2()
            #sampler = Sampler()
            print("  - 使用本地Qiskit模拟器")

        print(f"Ansatz type: {type(ansatz)}")
        print(f"Ansatz num_qubits: {ansatz.num_qubits}")
        print(f"Ansatz num_parameters: {ansatz.num_parameters}")
        print(f"Sampler type: {type(sampler)}") 

        # 初始化VQE
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
        vqe = SamplingVQE(
            sampler=sampler,
            ansatz=ansatz,
            optimizer=optimizer,
            aggregation=0.1,
            callback=store_intermediate_result,
        )
        print("✓ VQE初始化完成")
        
        # 计算最小特征值
        print("正在计算最小特征值...")
        print(f"  - 使用优化器: {type(optimizer).__name__}")
        print(f"  - 最大迭代次数: 50")  # 代码中设置的值
        print(f"  - 量子电路重复次数: {ansatz_reps}")
        raw_result = vqe.compute_minimum_eigenvalue(qubit_op)
        #raw_result = MinimumEigenOptimizer(vqe)
        print("✓ 计算完成")
        
        # 显示量子计算统计信息
        print(f"  - 量子电路深度: {ansatz.decompose().depth()}")
        print(f"  - 量子比特数量: {ansatz.num_qubits}")
        print(f"  - 量子门数量: {ansatz.decompose().size()}")
        print(f"  - 总函数评估次数: {len(counts)}")
        if len(counts) > 0:
            print(f"  - 最终能量: {values[-1]:.6f}")
        
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
                plt.savefig('vqe_optimization_process.png', dpi=PLOT_DPI, bbox_inches='tight')
                print(f"✓ VQE优化过程图已保存为 vqe_optimization_process.png (DPI: {PLOT_DPI})")
            
            if SAVE_PLOT_PDF:
                plt.savefig('vqe_optimization_process.pdf', bbox_inches='tight')
                print("✓ VQE优化过程图已保存为 vqe_optimization_process.pdf")
            
            # 在非交互式环境中，我们只保存图片而不显示
            plt.close()  # 关闭图形以释放内存
            print("✓ VQE优化过程折线图生成完成")
            
        except ImportError:
            print("⚠ 无法生成VQE优化过程折线图 (缺少matplotlib依赖)")
        except Exception as e:
            print(f"⚠ 生成VQE优化过程折线图时出错: {e}")
        
        # 解释结果
        print("\n正在解释结果...")
        result = protein_folding_problem.interpret(raw_result=raw_result)
        print(f"✓ 蛋白质形状解码完成")
        print(f"  - 折叠蛋白的主链转向序列: {result.protein_shape_decoder.main_turns}")
        print(f"  - 侧链转向序列: {result.protein_shape_decoder.side_turns}")
        print(f"  - 代表蛋白质形状的比特串: {result.turn_sequence}")
        
        # 获取笛卡尔坐标
        print("\n正在获取蛋白质的笛卡尔坐标...")
        xyz_data = result.protein_shape_file_gen.get_xyz_data()
        print("✓ 坐标数据获取完成")
        print("前几行坐标数据:")
        for i, row in enumerate(xyz_data[:10]):
            line = ' '.join(map(str, row))
            if line.strip():
                print(f"  {line}")
        
        # 显示并保存蛋白质结构的3D图形（如果环境支持）
        try:
            print("\n正在生成蛋白质结构的3D图形...")
            fig = result.get_figure(title="Protein Structure", ticks=False, grid=True)
            fig.get_axes()[0].view_init(10, 70)
            
            # 保存图形为文件（根据配置）
            import matplotlib.pyplot as plt
            if SAVE_PLOT_PNG:
                fig.savefig('protein_structure.png', dpi=PLOT_DPI, bbox_inches='tight')
                print(f"✓ 3D图形已保存为 protein_structure.png (DPI: {PLOT_DPI})")
            
            if SAVE_PLOT_PDF:
                fig.savefig('protein_structure.pdf', bbox_inches='tight')
                print("✓ 3D图形已保存为 protein_structure.pdf")
            
        except ImportError:
            print("⚠ 无法生成3D图形 (缺少matplotlib依赖)")
        except Exception as e:
            print(f"⚠ 保存图形时出错: {e}")
        
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
