#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lib模块包初始化文件

导出lib模块中的所有主要类，方便导入使用

模块列表：
1. EnergyCalculator (energy_calculator.py)
   - 功能：能量计算、CVaR计算、结果提取
   - 用途：从量子测量结果中计算能量值

2. QuantumBackendManager (quantum_backend_manager.py)
   - 功能：量子后端配置和管理
   - 用途：设置本地/AWS/IBM量子后端

3. ProteinFoldingBuilder (protein_folding_builder.py)
   - 功能：构建蛋白质折叠量子问题
   - 用途：创建哈密顿量和问题对象

4. ResultHandler (result_handler.py)
   - 功能：结果文件管理、可视化、PDB生成
   - 用途：保存结果、生成图表和PDB文件

5. QuantumOptimizer (quantum_optimizer.py)
   - 功能：VQE和Sampler优化
   - 用途：执行量子优化算法

6. MockQuantumResult (quantum_optimizer.py)
   - 功能：模拟量子结果类
   - 用途：包装量子测量结果

7. JobMetadataLogger (job_metadata_logger.py)
   - 功能：作业元数据记录
   - 用途：记录作业执行信息到CSV

使用示例：
    >>> from lib import (
    ...     ProteinFoldingBuilder, 
    ...     QuantumBackendManager, 
    ...     EnergyCalculator, 
    ...     ResultHandler, 
    ...     QuantumOptimizer,
    ...     JobMetadataLogger
    ... )
    >>> 
    >>> # 构建问题
    >>> builder = ProteinFoldingBuilder('APRLRFY')
    >>> 
    >>> # 设置后端
    >>> backend_info = QuantumBackendManager.setup_backend('local')
    >>> 
    >>> # 计算能量
    >>> energy = EnergyCalculator.estimate_energy_from_bitstring('0101', qubit_op)
    >>> 
    >>> # 创建结果目录
    >>> result_dir = ResultHandler.create_result_directory('./results', 'qiskit', 'estimator')
    >>> 
    >>> # 运行优化
    >>> optimizer = QuantumOptimizer.create_vqe_optimizer(qubit_op, ansatz, optimizer, estimator)
    >>> 
    >>> # 记录元数据
    >>> metadata_logger = JobMetadataLogger()
    >>> @metadata_logger
    >>> def main():
    ...     pass

版本：3.2.0
"""

from .energy_calculator import EnergyCalculator
from .quantum_backend_manager import QuantumBackendManager
from .protein_folding_builder import ProteinFoldingBuilder
from .result_handler import ResultHandler
from .quantum_optimizer import QuantumOptimizer, MockQuantumResult
from .job_metadata_logger import JobMetadataLogger

# 导出列表
__all__ = [
    'EnergyCalculator',
    'QuantumBackendManager',
    'ProteinFoldingBuilder',
    'ResultHandler',
    'QuantumOptimizer',
    'MockQuantumResult',
    'JobMetadataLogger',
    'convert_xyz_to_detailed_pdb'
]

# 为了向后兼容，提供 convert_xyz_to_detailed_pdb 的快捷访问
convert_xyz_to_detailed_pdb = ResultHandler.convert_xyz_to_detailed_pdb

# 版本信息
__version__ = '3.2.0'