#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
蛋白质折叠问题构建器模块

功能：
- 构建蛋白质折叠的量子哈密顿量
- 管理约束惩罚参数
- 提供问题构建的统一接口

主要类：
- ProteinFoldingBuilder: 蛋白质折叠问题构建器类

主要方法：
- build(): 构建蛋白质折叠问题，返回问题对象和量子比特算子
- get_qubit_operator(): 获取量子比特算子
- get_problem(): 获取蛋白质折叠问题对象

约束惩罚参数：
- penalty_chiral: 手性约束惩罚系数（默认10）
- penalty_back: 几何约束惩罚系数（默认10）
- penalty_local_overlap: 局部重叠惩罚系数（默认10）

使用示例：
    >>> from lib import ProteinFoldingBuilder
    >>> builder = ProteinFoldingBuilder('APRLRFY', 
    ...                              penalty_chiral=10, 
    ...                              penalty_back=10, 
    ...                              penalty_local_overlap=10)
    >>> problem, qubit_op = builder.build()

依赖：
- src.protein_folding.protein_folding_problem: 蛋白质折叠问题类
- src.protein_folding.qubit_op_builder: 量子比特算子构建器
- src.protein_folding.penalty_parameters: 惩罚参数类
- src.protein_folding.peptide: 肽链类
- src.protein_folding.interactions.miyazawa_jernigan_interaction: MJ相互作用矩阵
"""

from src.protein_folding.protein_folding_problem import ProteinFoldingProblem
from src.protein_folding.qubit_op_builder import QubitOpBuilder
from src.protein_folding.penalty_parameters import PenaltyParameters
from src.protein_folding.peptide.peptide import Peptide
from src.protein_folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction


class ProteinFoldingBuilder:
    """蛋白质折叠问题构建器类"""
    
    def __init__(self, main_chain, penalty_chiral=10, penalty_back=10, penalty_local_overlap=10):
        """
        初始化蛋白质折叠构建器
        
        Args:
            main_chain: 主链氨基酸序列
            penalty_chiral: 手性约束惩罚系数
            penalty_back: 几何约束惩罚系数
            penalty_local_overlap: 局部重叠惩罚系数
        """
        self.main_chain = main_chain
        self.penalty_chiral = penalty_chiral
        self.penalty_back = penalty_back
        self.penalty_local_overlap = penalty_local_overlap
        
        # 创建必要的组件
        # 对于侧链，假设为空（无侧链）
        side_chain_residue_sequences = [""] * len(main_chain)
        peptide = Peptide(main_chain, side_chain_residue_sequences)
        interaction = MiyazawaJerniganInteraction()
        penalty_parameters = PenaltyParameters(
            penalty_chiral=penalty_chiral,
            penalty_back=penalty_back,
            penalty_1=penalty_local_overlap
        )
        
        # 构建蛋白质折叠问题
        self.problem = ProteinFoldingProblem(
            peptide=peptide,
            interaction=interaction,
            penalty_parameters=penalty_parameters
        )
        
        # 使用问题对象内部的量子比特构建器
        self.qubit_op_builder = self.problem._qubit_op_builder
    
    def get_qubit_operator(self):
        """
        获取量子比特哈密顿量
        
        Returns:
            QubitOperator: 量子比特哈密顿量
        """
        return self.problem.qubit_op()
    
    def get_problem(self):
        """
        获取蛋白质折叠问题
        
        Returns:
            ProteinFoldingProblem: 蛋白质折叠问题实例
        """
        return self.problem
    
    def get_num_qubits(self):
        """
        获取量子比特数量
        
        Returns:
            int: 优化后的量子比特数量
        """
        # 从优化后的量子比特算子中获取量子比特数量
        # qubit_op() 方法会自动调用 qubit_number_reducer.remove_unused_qubits() 来优化量子比特数量
        qubit_op = self.problem.qubit_op()
        return qubit_op.num_qubits
    
    def get_parameters(self):
        """
        获取构建器参数
        
        Returns:
            dict: 参数字典
        """
        return {
            'main_chain': self.main_chain,
            'penalty_chiral': self.penalty_chiral,
            'penalty_back': self.penalty_back,
            'penalty_local_overlap': self.penalty_local_overlap,
            'num_qubits': self.get_num_qubits()
        }