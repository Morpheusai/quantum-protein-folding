#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量子优化器模块

功能：
- 封装VQE和Sampler模式的优化逻辑
- 提供模拟量子结果类
- 统一的优化接口

主要类：
- QuantumOptimizer: 量子优化器类，提供VQE和Sampler优化方法
- MockQuantumResult: 模拟量子结果类

主要方法：
VQE优化：
- create_vqe_optimizer(): 创建VQE优化器
- run_vqe_optimization(): 运行VQE优化

Sampler优化：
- create_sampler_optimizer(): 创建Sampler优化器
- run_sampler_optimization(): 运行Sampler优化

模拟结果：
- MockQuantumResult: 模拟量子测量结果

使用示例：
    >>> from lib import QuantumOptimizer
    >>> vqe_result, conv_data = QuantumOptimizer.create_vqe_optimizer(qubit_op, ansatz, optimizer, estimator, backend)
    >>> sampler_result, conv_data = QuantumOptimizer.create_sampler_optimizer(qubit_op, ansatz, optimizer, sampler, backend)

依赖：
- qiskit_algorithms: VQE算法
- qiskit: 量子电路转译
- scipy.optimize: 优化器（COBYLA等）
- numpy: 数值计算
- lib.energy_calculator: 能量计算
"""

import copy
from qiskit_algorithms import VQE
from qiskit import transpile
import numpy as np
from scipy.optimize import minimize
import os
import json

from lib.energy_calculator import EnergyCalculator


class MockQuantumResult:
    """模拟量子结果类"""
    
    def __init__(self, eigenstate, eigenvalue, counts, total_shots):
        """
        初始化模拟量子结果
        
        Args:
            eigenstate (str or dict): 本征态（bitstring或概率字典）
            eigenvalue (float): 本征值（能量）
            counts (dict): 测量结果计数 {bitstring: count}
            total_shots (int): 总采样次数
        """
        self.eigenstate = eigenstate
        self.eigenvalue = eigenvalue
        self.counts = counts
        self.total_shots = total_shots
    
    def get_eigenstate(self):
        """
        获取本征态
        
        Returns:
            str or dict: 本征态（bitstring或概率字典）
        """
        return self.eigenstate
    
    def get_eigenvalue(self):
        """
        获取本征值
        
        Returns:
            float: 本征值（能量）
        """
        return self.eigenvalue
    
    def get_counts(self):
        """
        获取测量结果计数
        
        Returns:
            dict: 测量结果计数字典 {bitstring: count}
        """
        return self.counts
    
    def get_total_shots(self):
        """
        获取总采样次数
        
        Returns:
            int: 总采样次数
        """
        return self.total_shots
    
    def get_most_probable_state(self):
        """
        获取最可能的状态
        
        Returns:
            str: 最可能的bitstring
        """
        if not self.counts:
            return self.eigenstate
        
        max_count = 0
        most_probable = self.eigenstate
        
        for state, count in self.counts.items():
            if count > max_count:
                max_count = count
                most_probable = state
        
        return most_probable
    
    def get_probability(self, state):
        """
        获取特定状态的概率
        
        Args:
            state (str): 要查询的状态bitstring
            
        Returns:
            float: 状态概率（0到1之间）
        """
        if not self.counts or self.total_shots == 0:
            return 0.0
        
        return self.counts.get(state, 0) / self.total_shots
    
    def __str__(self):
        """字符串表示"""
        return f"MockQuantumResult(eigenstate={self.eigenstate}, eigenvalue={self.eigenvalue:.4f})"
    
    def __repr__(self):
        """详细字符串表示"""
        return (f"MockQuantumResult(eigenstate='{self.eigenstate}', "
                f"eigenvalue={self.eigenvalue:.4f}, "
                f"total_shots={self.total_shots})")


class QuantumOptimizer:
    """
    量子优化器基类
    
    提供统一的优化接口，支持VQE和Sampler两种模式
    """
    
    @staticmethod
    def create_vqe_optimizer(qubit_op, ansatz, optimizer, estimator, backend=None, args=None):
        """
        创建VQE优化器
        
        Args:
            qubit_op: 量子比特哈密顿量算子
            ansatz: 变分量子线路
            optimizer: 优化器
            estimator: 量子估算器
            backend: 量子后端（可选）
            args: 命令行参数对象
            
        Returns:
            tuple: (VQE结果, 收敛数据字典)
        """
        working_ansatz = copy.deepcopy(ansatz)
        working_ansatz.remove_final_measurements()

        if backend is not None and args is not None:
            working_ansatz = transpile(
                working_ansatz, 
                backend=backend,
                initial_layout=list(range(working_ansatz.num_qubits)) 
                    if args.backend not in ['local', 'aws_sv1', 'ibm_simulator'] else None,
                optimization_level=3
            )
            print(f"   VQE电路转译: 逻辑比特数={working_ansatz.num_qubits}, 物理比特数={working_ansatz.width()}")

        convergence = {'counts': [], 'values': [], 'cumulative_shots': [], 'iteration_shots': []}
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
            
            current_step_shots = args.shots if args else 100
            actual_shots_list.append(current_step_shots)
            convergence['iteration_shots'].append(current_step_shots)
            
            cumulative_shots = sum(actual_shots_list)
            convergence['cumulative_shots'].append(cumulative_shots)

        vqe = VQE(estimator=estimator, ansatz=working_ansatz, optimizer=optimizer, callback=callback)
        result = vqe.compute_minimum_eigenvalue(qubit_op)
        
        if hasattr(result, 'metadata') and result.metadata:
            total_shots_from_metadata = record_shots_from_metadata(result.metadata)
            if total_shots_from_metadata > 0:
                if len(convergence['cumulative_shots']) > 0:
                    avg_shots_per_eval = max(1, total_shots_from_metadata // len(convergence['cumulative_shots']))
                    convergence['cumulative_shots'] = [i * avg_shots_per_eval 
                                                       for i in range(1, len(convergence['cumulative_shots']) + 1)]
        
        convergence['label'] = 'VQE Energy'
        return result, convergence
    
    @staticmethod
    def create_sampler_optimizer(transpiled_circuit, qubit_op, backend, args, result_dir=None, problem=None):
        """
        创建Sampler优化器
        
        Args:
            transpiled_circuit: 转译后的量子电路
            qubit_op: 量子比特哈密顿量算子
            backend: 量子后端
            args: 命令行参数对象
            result_dir: 结果目录路径（用于保存迭代结果）
            
        Returns:
            tuple: (优化结果, 收敛历史, 迭代结果列表, 所有top能量列表, 累积shots历史)
        """
        convergence_history = []
        cumulative_shots_history = []
        iteration_shots_history = []
        iteration_results = []
        all_top_energies = []

        # 如果提供了结果目录，则创建iter_all_results子目录
        iteration_result_dir = None
        if result_dir:
            iteration_result_dir = os.path.join(result_dir, "iter_all_results")
            os.makedirs(iteration_result_dir, exist_ok=True)

        def objective_function(params):
            """
            优化目标函数
            
            该函数接受参数，执行量子电路，使用CVaR策略计算能量，并返回用于优化的值
            
            Args:
                params: 变分参数数组
                
            Returns:
                float: CVaR能量值（优化目标）
            """
            bound_circ = transpiled_circuit.assign_parameters(params)
            
            job = backend.run(bound_circ, shots=args.shots)
            
            counts = job.result().get_counts()
            
            actual_shots = sum(counts.values())
            
            energy = EnergyCalculator.calculate_cvar_energy(counts, qubit_op, args.alpha)
            
            top_results = EnergyCalculator.extract_top_results(counts, qubit_op, args.max_results)
            top_energies = [result[1] for result in top_results]
            
            convergence_history.append(energy)
            iteration_shots_history.append(actual_shots)
            
            if cumulative_shots_history:
                cumulative_shots_history.append(cumulative_shots_history[-1] + actual_shots)
            else:
                cumulative_shots_history.append(actual_shots)
            
            all_top_energies.append(top_energies)
            
            if len(convergence_history) % 2 == 0:
                print(f"    迭代 {len(convergence_history)}: CVaR能量 = {energy:.4f}, 实际shots = {actual_shots}")
                print(f"    各最优结果能量: {[round(e, 4) for e in top_energies]}")
            
            iteration_data = {
                "iteration": len(convergence_history),
                "cvar_energy": energy,
                "top_energies": top_energies,
                "top_results": [(result[0], float(result[1]), result[2]) for result in top_results],
                "total_counts": len(counts),
                "actual_shots": actual_shots,
                "raw_counts": counts  # 添加原始计数用于后续蛋白质结构解析
            }
            iteration_results.append(iteration_data)
            
            # 如果指定了迭代结果目录，则保存当前迭代结果到单独的文件
            if iteration_result_dir:
                # 准备蛋白质结构信息
                protein_structure_info = {}
                top_results_for_struct = iteration_data.get("top_results", [])
                
                if problem and top_results_for_struct:
                    best_bitstring, best_energy, best_count = top_results_for_struct[0]
                    
                    # 创建模拟结果对象用于解析蛋白质结构
                    class MockResult:
                        def __init__(self, bs, val, counts, total):
                            # 设置最高概率比特串的振幅为1（理想情况）
                            self.eigenstate = {bs: 1.0}
                            # 设置优化得到的能量值
                            self.eigenvalue = val
                            # 提供完整的概率分布用于后续分析
                            self.probabilities = {k[::-1]: v/total for k, v in counts.items()}
                    
                    # 创建模拟结果并解析为蛋白质结构
                    raw_res = MockResult(best_bitstring, best_energy, counts, actual_shots)
                    try:
                        result = problem.interpret(raw_res)
                        
                        # 获取XYZ坐标数据
                        xyz_data = result.protein_shape_file_gen.get_xyz_data()
                        
                        protein_structure_info = {
                            "turn_sequence": result.turn_sequence if hasattr(result, 'turn_sequence') else "",
                            "xyz_coordinates": [list(row) for row in xyz_data] if xyz_data is not None else [],
                            "best_bitstring": best_bitstring
                        }
                    except Exception as e:
                        print(f"    ⚠ 迭代 {iteration_data['iteration']} 蛋白质结构解析失败: {e}")
                        protein_structure_info = {
                            "turn_sequence": "",
                            "xyz_coordinates": [],
                            "best_bitstring": best_bitstring
                        }
                
                iteration_result_path = os.path.join(iteration_result_dir, f'iteration_{len(convergence_history)}_result.json')
                with open(iteration_result_path, 'w') as f:
                    serializable_data = {
                        "iteration": iteration_data["iteration"],
                        "cvar_energy": iteration_data["cvar_energy"],
                        "top_energies": [float(e) for e in iteration_data["top_energies"]],
                        "total_counts": iteration_data["total_counts"],
                        "actual_shots": iteration_data["actual_shots"],
                        "protein_structure": protein_structure_info
                    }
                    json.dump(serializable_data, f, indent=2)
            
            return energy

        print(f"\n正在开始优化 (使用 COBYLA)...")
        num_parameters = transpiled_circuit.num_parameters
        initial_params = np.random.uniform(-np.pi, np.pi, num_parameters)
        res = minimize(objective_function, initial_params, method='COBYLA', 
                       options={'maxiter': args.max_optimization_iterations})
        
        return res, convergence_history, iteration_results, all_top_energies, cumulative_shots_history, iteration_shots_history
