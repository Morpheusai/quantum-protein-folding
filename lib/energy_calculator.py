#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
能量计算器模块

功能：
- 从量子测量结果中提取能量值
- 计算CVaR（条件风险价值）能量
- 提取前N个最优结果

主要类：
- EnergyCalculator: 能量计算器类，提供静态方法进行能量计算

主要方法：
- estimate_energy_from_bitstring(): 计算单个比特串的能量
- extract_top_n_results(): 提取前N个最优结果
- calculate_cvar_energy(): 计算CVaR能量

使用示例：
    >>> from lib import EnergyCalculator
    >>> energy = EnergyCalculator.estimate_energy_from_bitstring('0101', qubit_op)
    >>> top_results = EnergyCalculator.extract_top_n_results(counts, qubit_op, n=5)
    >>> cvar_energy = EnergyCalculator.calculate_cvar_energy(counts, qubit_op, alpha=0.1)

依赖：
- numpy: 数值计算
- qiskit.quantum_info: Pauli算子处理
"""

import numpy as np


class EnergyCalculator:
    """能量计算器类"""
    
    @staticmethod
    def estimate_energy_from_bitstring(bitstring, qubit_op):
        """
        计算单个采样比特串对应的哈密顿量能量值
        
        Args:
            bitstring (str): 量子测量得到的比特串，如 '010110'
            qubit_op: 量子比特哈密顿量算子（SparsePauliOp或Pauli）
            
        Returns:
            float: 对应的能量值
            
        Note:
            通过遍历哈密顿量的所有Pauli项，计算在给定比特串下的期望值
        """
        energy = 0.0
        # 将比特串反转，使其与哈密顿量的索引顺序对应
        bit_list = [int(b) for b in reversed(bitstring)]
        
        if len(bitstring) != qubit_op.num_qubits:
             raise ValueError(f"比特串长度 ({len(bitstring)}) 与哈密顿量量子比特数 ({qubit_op.num_qubits}) 不匹配")

        # 遍历哈密顿量的每一项 (Pauli算子及其系数)
        for pauli_str, coeff in qubit_op.to_list():
            val = 1.0
            # 对于每个Pauli项，计算其在当前比特态下的期望值
            for i, char in enumerate(reversed(pauli_str)):
                if char == 'Z' and bit_list[i] == 1:
                    # Z算子在|1>态下贡献-1，在|0>态下贡献+1
                    val *= -1.0
                elif char == 'X' or char == 'Y':
                    # X,Y算符在计算基态下的期望值为0
                    val = 0.0 
                    break
            energy += coeff.real * val
        return energy
    
    @staticmethod
    def calculate_cvar_energy(counts, qubit_op, alpha=0.1):
        """
        计算CVaR (Conditional Value at Risk) 能量
        
        CVaR是一种风险度量方法，只考虑能量最低的一部分样本
        
        Args:
            counts (dict): 量子测量结果的字典 {bitstring: count}
            qubit_op: 量子比特哈密顿量算子
            alpha (float, optional): CVaR参数，取值[0,1]，表示使用最低能量样本的比例，默认为0.1
            
        Returns:
            float: CVaR能量值（最低alpha比例样本的平均能量）
            
        Note:
            - 按总shots数计算，保持正确的样本权重
            - alpha=0.1表示使用能量最低的10%样本
        """
        energies = []
        
        # 计算每个比特串的能量，并根据出现次数复制
        for bitstring, count in counts.items():
            e = EnergyCalculator.estimate_energy_from_bitstring(bitstring, qubit_op)
            # 根据出现次数复制能量值，保持正确的权重
            energies.extend([e] * count)
        
        # 按能量值升序排列
        energies.sort()
        
        # 计算需要保留的样本数量（基于总shots数）
        total_shots = sum(counts.values())
        num_keep = max(1, int(total_shots * alpha))
        
        # 返回最低能量样本的平均值
        return np.mean(energies[:num_keep])
    
    @staticmethod
    def calculate_energy_std(counts, qubit_op):
        """
        计算能量分布的标准差
        
        Args:
            counts (dict): 量子测量结果的字典 {bitstring: count}
            qubit_op: 量子比特哈密顿量算子
            
        Returns:
            float: 能量分布的标准差
        """
        energies = []
        weights = []
        
        # 计算每个比特串的能量和权重
        for bitstring, count in counts.items():
            e = EnergyCalculator.estimate_energy_from_bitstring(bitstring, qubit_op)
            energies.append(e)
            weights.append(count)
        
        # 计算加权标准差
        # 使用numpy的cov或手动计算var
        # var = sum(w * (x - mean)^2) / sum(w)
        if not energies:
            return 0.0
            
        mean_energy = np.average(energies, weights=weights)
        variance = np.average((np.array(energies) - mean_energy)**2, weights=weights)
        
        return np.sqrt(variance)
    
    @staticmethod
    def extract_top_results(counts, qubit_op, max_results=1):
        """
        提取前N个最优结果
        
        Args:
            counts (dict): 量子测量结果的字典 {bitstring: count}
            qubit_op: 量子比特哈密顿量算子
            max_results (int, optional): 要提取的最优结果数量，默认为1
            
        Returns:
            list: 包含前max_results个最优结果的列表，每个元素为(bitstring, energy, count)元组
                - bitstring (str): 比特串
                - energy (float): 能量值
                - count (int): 出现次数
                
        Note:
            结果按能量值升序排列（能量越低越好）
        """
        # 计算每个比特串的能量
        bitstring_energies = []
        for bitstring, count in counts.items():
            energy = EnergyCalculator.estimate_energy_from_bitstring(bitstring, qubit_op)
            bitstring_energies.append((bitstring, energy, count))
        
        # 按能量值升序排列
        bitstring_energies.sort(key=lambda x: x[1])
        
        # 提取能量最低的top_n个结果
        top_results = bitstring_energies[:max_results]
        
        return top_results
    
    @staticmethod
    def calculate_cvar_energy_precise(counts, protein_sequence, interaction_matrix, turn2qubit, alpha=0.1):
        """
        使用精确哈密顿量计算 CVaR 能量
        
        Args:
            counts (dict): 量子测量结果 {bitstring: count}
            protein_sequence (str): 蛋白质序列
            interaction_matrix (np.ndarray): 相互作用矩阵
            turn2qubit (str): 转向到量子比特映射
            alpha (float, optional): CVaR 参数，默认为 0.1
            
        Returns:
            float: CVaR 能量值
        """
        from lib.precise_energy_calculator import PreciseEnergyCalculator
        
        calc = PreciseEnergyCalculator(protein_sequence, interaction_matrix)
        
        energies = []
        for bitstring, count in counts.items():
            e = calc.calculate_energy_from_bitstring(bitstring, turn2qubit)
            energies.extend([e] * count)
        
        energies.sort()
        total_shots = sum(counts.values())
        num_keep = max(1, int(total_shots * alpha))
        
        return np.mean(energies[:num_keep])
    
    @staticmethod
    def extract_top_results_precise(counts, protein_sequence, interaction_matrix, turn2qubit, max_results=1):
        """
        使用精确哈密顿量提取前 N 个最优结果
        
        Args:
            counts (dict): 量子测量结果 {bitstring: count}
            protein_sequence (str): 蛋白质序列
            interaction_matrix (np.ndarray): 相互作用矩阵
            turn2qubit (str): 转向到量子比特映射
            max_results (int, optional): 要提取的最优结果数量，默认为 1
            
        Returns:
            list: [(bitstring, energy, count), ...]
        """
        from lib.precise_energy_calculator import PreciseEnergyCalculator
        
        calc = PreciseEnergyCalculator(protein_sequence, interaction_matrix)
        
        bitstring_energies = []
        for bitstring, count in counts.items():
            e = calc.calculate_energy_from_bitstring(bitstring, turn2qubit)
            bitstring_energies.append((bitstring, e, count))
        
        bitstring_energies.sort(key=lambda x: x[1])
        return bitstring_energies[:max_results]