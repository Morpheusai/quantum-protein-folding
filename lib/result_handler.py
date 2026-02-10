#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用结果处理模块

功能：
- 结果文件管理（创建、保存、加载）
- 结果目录结构管理
- 3D蛋白质结构图生成
- VQE收敛曲线图（带shots信息）
- 结果摘要文件生成
- 详细PDB文件生成

主要类：
- ResultHandler: 通用结果处理类，提供静态方法进行结果管理

主要方法：
结果文件管理：
- create_result_directory(): 创建带时间戳的结果目录
- save_result(): 保存结果到JSON文件
- load_result(): 从JSON文件加载结果
- generate_result_filename(): 生成结果文件名
- get_result_summary(): 获取结果目录摘要

可视化功能：
- plot_protein_structure_3d(): 生成3D蛋白质结构图
- plot_vqe_optimization_summary(): 生成VQE优化摘要图
- plot_vqe_convergence_with_shots(): 生成VQE收敛曲线图（带shots信息）
- generate_output_summary(): 生成输出摘要

PDB文件生成：
- convert_xyz_to_detailed_pdb(): 将XYZ坐标转换为详细PDB文件

内部类：
- DetailedPDBGenerator: 详细PDB文件生成器

使用示例：
    >>> from lib import ResultHandler
    >>> result_dir = ResultHandler.create_result_directory('./results', 'local', 'sampler')
    >>> ResultHandler.save_result(data, 'result.json')
    >>> ResultHandler.plot_protein_structure_3d(result, 'structure.png')
    >>> ResultHandler.convert_xyz_to_detailed_pdb(xyz_data, 'output.pdb')

依赖：
- os, json, datetime: 文件和日期处理
- numpy: 数值计算
- matplotlib: 图表生成
"""

import os
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import List, Tuple, Dict, Optional


class ResultHandler:
    """通用结果处理类"""
    
    # ==================== 结果文件管理 ====================
    
    @staticmethod
    def create_result_directory(base_dir, backend_name, mode='estimator'):
        """
        创建结果目录，生成带时间戳的唯一目录名
        
        根据当前时间戳、后端名称和运行模式创建结果目录，目录命名格式为：
        {timestamp}_{backend_name}_{mode}
        
        Args:
            base_dir (str): 基础目录路径，结果目录将在此目录下创建
            backend_name (str): 后端名称，用于标识使用的计算后端
            mode (str, optional): 运行模式，默认为'estimator'，可选值为'estimator'或'sampler'
            
        Returns:
            str: 创建的结果目录完整路径
            
        Example:
            >>> result_dir = ResultHandler.create_result_directory('./results', 'qiskit', 'estimator')
            >>> print(result_dir)
            ./results/20240115_143022_qiskit_estimator
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        dir_name = f"{timestamp}_{backend_name}_{mode}"
        result_dir = os.path.join(base_dir, dir_name)
        
        os.makedirs(result_dir, exist_ok=True)
        
        return result_dir
    
    @staticmethod
    def save_result(result_data, filename):
        """
        保存结果到JSON文件
        
        Args:
            result_data (dict): 要保存的数据字典
            filename (str): 文件名（包含路径）
            
        Note:
            如果文件所在目录不存在，会自动创建
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_result(filename):
        """
        从JSON文件加载结果
        
        Args:
            filename (str): 文件名（包含路径）
            
        Returns:
            dict: 加载的数据字典
            
        Raises:
            FileNotFoundError: 如果文件不存在
            json.JSONDecodeError: 如果文件格式错误
        """
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def generate_result_filename(result_dir, rank, experiment_num, energy, file_type='json'):
        """
        生成结果文件名
        
        Args:
            result_dir (str): 结果目录路径
            rank (int): 结果排名（从1开始）
            experiment_num (int): 实验编号
            energy (float): 能量值
            file_type (str, optional): 文件类型，默认为'json'，可选值为'json', 'pdb', 'png'
            
        Returns:
            str: 生成的完整文件路径
            
        Example:
            >>> filename = ResultHandler.generate_result_filename('./results', 1, 5, 1568.68, 'pdb')
            >>> print(filename)
            ./results/structure_rank_1_exp_5_energy_1568.6800.pdb
        """
        energy_str = f"{abs(energy):.4f}"
        
        if file_type == 'json':
            filename = f"result_rank_{rank}_exp_{experiment_num}_energy_{energy_str}.json"
        elif file_type == 'pdb':
            filename = f"structure_rank_{rank}_exp_{experiment_num}_energy_{energy_str}.pdb"
        elif file_type == 'png':
            filename = f"structure_rank_{rank}_exp_{experiment_num}_energy_{energy_str}.png"
        else:
            filename = f"result_rank_{rank}_exp_{experiment_num}_energy_{energy_str}.{file_type}"
        
        return os.path.join(result_dir, filename)
    
    @staticmethod
    def get_result_summary(result_dir):
        """
        获取结果目录的摘要信息
        
        Args:
            result_dir (str): 结果目录路径
            
        Returns:
            dict: 摘要信息字典，包含以下键：
                - directory (str): 目录路径
                - files (list): 文件名列表
                - total_files (int): 总文件数
                - json_files (int): JSON文件数
                - pdb_files (int): PDB文件数
                - png_files (int): PNG文件数
                
        Example:
            >>> summary = ResultHandler.get_result_summary('./results/20240115_143022_local_estimator')
            >>> print(summary['total_files'])
            15
        """
        summary = {
            'directory': result_dir,
            'files': [],
            'total_files': 0,
            'json_files': 0,
            'pdb_files': 0,
            'png_files': 0
        }
        
        if os.path.exists(result_dir):
            for filename in os.listdir(result_dir):
                filepath = os.path.join(result_dir, filename)
                if os.path.isfile(filepath):
                    summary['files'].append(filename)
                    summary['total_files'] += 1
                    
                    if filename.endswith('.json'):
                        summary['json_files'] += 1
                    elif filename.endswith('.pdb'):
                        summary['pdb_files'] += 1
                    elif filename.endswith('.png'):
                        summary['png_files'] += 1
        
        return summary
    
    # ==================== 可视化功能 ====================
    
    @staticmethod
    def plot_protein_structure_3d(result, filename, title="Protein Structure", figsize=(10, 8)):
        """
        绘制3D蛋白质结构图（使用ProteinFoldingResult的get_figure方法）
        
        Args:
            result: ProteinFoldingResult对象
            filename (str): 保存的文件路径
            title (str, optional): 图表标题，默认为"Protein Structure"
            figsize (tuple, optional): 图表大小，默认为(10, 8)
        """
        fig = result.get_figure(title=title, ticks=True, grid=True)
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    @staticmethod
    def plot_vqe_optimization_summary(all_conv_data, filename, main_chain, figsize=(12, 8)):
        """
        绘制VQE优化摘要图（仅能量收敛曲线）
        
        Args:
            all_conv_data (list): 收敛数据列表
            filename (str): 保存的文件路径
            main_chain (str): 主链序列
            figsize (tuple, optional): 图表大小，默认为(12, 8)
        """
        plt.figure(figsize=figsize)
        for idx, data in enumerate(all_conv_data):
            label = data.get('label', f'Run {idx+1}')
            x_values = range(len(data['values']))
            if 'CVaR' in label:
                linestyle = '--'
            else:
                linestyle = '-'
            plt.plot(x_values, data['values'], marker='o', label=label, 
                    linewidth=2, linestyle=linestyle)
        
        plt.xlabel("Evaluation Counts")
        plt.ylabel("Energy")
        plt.title(f"VQE Convergence Comparison ({main_chain})")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_vqe_convergence_with_shots(all_conv_data, iteration_shots, filename, main_chain, figsize=(12, 8)):
        """
        绘制VQE收敛曲线图（带shots信息和误差带）
        
        Args:
            all_conv_data (list): 收敛数据列表
            iteration_shots (list): 每次迭代的实际shots数列表
            filename (str): 保存的文件路径
            main_chain (str): 主链序列
            figsize (tuple, optional): 图表大小，默认为(12, 8)
        """
        fig, ax1 = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_conv_data)))
        
        energy_lines = []
        labels = []
        for idx, data in enumerate(all_conv_data):
            color = colors[idx]
            label = data.get('label', f'Run {idx+1}')
            values = np.array(data['values'])
            stds = np.array(data.get('stds', []))
            x_values = range(len(values))
            
            if 'CVaR' in label:
                linestyle = '--'
            else:
                linestyle = '-'
            
            # 绘制主线
            line, = ax1.plot(x_values, values, marker='o', label=label, 
                            linewidth=3, color=color, linestyle=linestyle)
            
            # 如果有标准差数据，绘制误差带 (mean ± std)
            if len(stds) == len(values) and len(stds) > 0:
                try:
                    # 确保 stds 是数值类型
                    stds_float = stds.astype(float)
                    ax1.fill_between(x_values, values - stds_float, values + stds_float, 
                                     color=color, alpha=0.2, label=f'{label} Std Dev')
                except Exception:
                    pass # 如果无法转换，跳过误差带绘制

            energy_lines.append(line)
            labels.append(label)
        
        ax1.set_xlabel('Evaluation Counts')
        ax1.set_ylabel('Energy', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3)
        
        ax2 = ax1.twinx()
        bars = ax2.bar(range(len(iteration_shots)), iteration_shots, alpha=0.3, width=0.5, 
                       color='gray', edgecolor='gray', linewidth=0.5, 
                       label='Shots per Iteration')
        ax2.set_ylabel('Shots per Iteration', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        
        ax1.legend(energy_lines, labels, loc='upper right')
        
        plt.title(f"VQE Convergence with Shots per Iteration ({main_chain})")
        fig.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def generate_output_summary(result_dict, filename):
        """
        生成输出摘要文件
        
        Args:
            result_dict (dict): 结果字典
            filename (str): 保存的文件路径
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("--- Quantum Protein Folding Summary (Sampler Mode) ---\n")
            f.write(f"Protein Sequence: {result_dict.get('main_chain_sequence', 'N/A')}\n")
            f.write(f"Backend:        {result_dict.get('backend', 'N/A')}\n")
            f.write(f"Shots Used:     {result_dict.get('shots_requested', 'N/A')}\n")
            if 'alpha' in result_dict:
                f.write(f"CVaR alpha:     {result_dict.get('alpha', 'N/A')}\n")
            f.write(f"Tries:          {result_dict.get('tries', 'N/A')}\n")
            f.write(f"Optimization Algorithm: {result_dict.get('optimizer', 'N/A')}\n")
            f.write(f"Penalty Back:   {result_dict.get('penalty_back', 'N/A')}\n")
            f.write(f"Penalty Chiral: {result_dict.get('penalty_chiral', 'N/A')}\n")
            f.write(f"Penalty Local Overlap: {result_dict.get('penalty_local_overlap', 'N/A')}\n")
            f.write(f"Minimum CVaR Energy: {result_dict.get('energy', 'N/A'):.6f}\n")
            f.write(f"Best Turn Sequence: {result_dict.get('turn_sequence', 'N/A')}\n")
            if 'original_experiment' in result_dict:
                f.write(f"Rank in Top N:  {result_dict.get('rank', 'N/A')} (from original experiment {result_dict.get('original_experiment', 'N/A')})\n")
            else:
                f.write(f"Rank in Top N: {result_dict.get('rank', 'N/A')}\n")

    # ==================== PDB文件生成 ====================
    
    class DetailedPDBGenerator:
        """详细PDB文件生成器类"""
        
        # 标准氨基酸1字母到3字母转换
        AA_1TO3 = {
            'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
            'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
            'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
            'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
        }

        # 定义每个氨基酸的主链原子
        BACKBONE_ATOMS = ['N', 'CA', 'C', 'O']

        # 定义每个氨基酸的侧链原子（简化表示）
        SIDE_CHAIN_ATOMS = {
            'ALA': ['CB'],
            'CYS': ['CB', 'SG'],
            'ASP': ['CB', 'CG', 'OD1', 'OD2'],
            'GLU': ['CB', 'CG', 'CD', 'OE1', 'OE2'],
            'PHE': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
            'GLY': [],
            'HIS': ['CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
            'ILE': ['CB', 'CG1', 'CG2', 'CD1'],
            'LYS': ['CB', 'CG', 'CD', 'CE', 'NZ'],
            'LEU': ['CB', 'CG', 'CD1', 'CD2'],
            'MET': ['CB', 'CG', 'SD', 'CE'],
            'ASN': ['CB', 'CG', 'OD1', 'ND2'],
            'PRO': ['CB', 'CG', 'CD'],
            'GLN': ['CB', 'CG', 'CD', 'OE1', 'NE2'],
            'ARG': ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
            'SER': ['CB', 'OG'],
            'THR': ['CB', 'OG1', 'CG2'],
            'VAL': ['CB', 'CG1', 'CG2'],
            'TRP': ['CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
            'TYR': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH']
        }

        def __init__(self, main_chain_positions: np.ndarray, side_chain_positions: List[Optional[np.ndarray]], 
                     main_chain_sequence: str, side_chain_sequences: List[Optional[str]] = None):
            """
            初始化详细PDB文件生成器
            
            Args:
                main_chain_positions: 主链α碳的位置
                side_chain_positions: 侧链原子的位置
                main_chain_sequence: 主链氨基酸序列
                side_chain_sequences: 侧链氨基酸序列
            """
            self.main_chain_positions = main_chain_positions
            self.side_chain_positions = side_chain_positions
            self.main_chain_sequence = main_chain_sequence
            self.side_chain_sequences = side_chain_sequences or [None] * len(main_chain_sequence)

        def generate_detailed_coordinates(self) -> List[Tuple[str, str, int, str, int, float, float, float, float, float, str]]:
            """
            生成PDB文件的详细原子坐标
            
            Returns:
                包含ATOM记录的元组列表，字段包括：
                (record_type, atom_num, atom_name, res_name, chain_id, res_seq, x, y, z, occupancy, temp_factor, element)
            """
            atoms = []
            atom_counter = 1
            
            for i, (residue_1, pos) in enumerate(zip(self.main_chain_sequence, self.main_chain_positions)):
                residue_3 = self.AA_1TO3.get(residue_1, 'UNK')
                
                backbone_offsets = {
                    'N': (-0.55, -0.20, -0.15),
                    'CA': (0.00, 0.00, 0.00),
                    'C': (0.50, -0.05, 0.15),
                    'O': (0.90, 0.05, 0.45)
                }
                
                for j, atom_name in enumerate(self.BACKBONE_ATOMS):
                    x, y, z = pos
                    
                    if atom_name in backbone_offsets:
                        offset_x, offset_y, offset_z = backbone_offsets[atom_name]
                        adjusted_x = x + offset_x
                        adjusted_y = y + offset_y
                        adjusted_z = z + offset_z
                    else:
                        adjusted_x = x + j * 0.1
                        adjusted_y = y + j * 0.05
                        adjusted_z = z + j * 0.02
                    
                    element = atom_name[0]
                    occupancy = 1.00
                    temp_factor = round(20.0 + (i % 5) * 2.0, 2)
                    
                    atoms.append((
                        'ATOM', atom_counter, atom_name, residue_3, 'A', i+1, 
                        adjusted_x, adjusted_y, adjusted_z, occupancy, temp_factor, element
                    ))
                    atom_counter += 1
                
                side_atoms = self.SIDE_CHAIN_ATOMS.get(residue_3, [])
                ca_x, ca_y, ca_z = pos
                
                for k, atom_name in enumerate(side_atoms):
                    angle = k * 1.2
                    radius = 1.0 + (k * 0.3)
                    
                    offset_x = radius * np.cos(angle)
                    offset_y = radius * np.sin(angle)
                    offset_z = k * 0.5
                    
                    adjusted_x = ca_x + offset_x
                    adjusted_y = ca_y + offset_y
                    adjusted_z = ca_z + offset_z
                    
                    element = atom_name[0] if len(atom_name) > 0 else 'C'
                    occupancy = 1.00
                    temp_factor = round(20.0 + (i % 5) * 2.0 + k * 0.5, 2)
                    
                    atoms.append((
                        'ATOM', atom_counter, atom_name, residue_3, 'A', i+1,
                        adjusted_x, adjusted_y, adjusted_z, occupancy, temp_factor, element
                    ))
                    atom_counter += 1

            return atoms

        def save_pdb_file(self, filename: str, title: str = "Detailed Protein Structure"):
            """
            保存详细原子坐标到PDB文件
            
            Args:
                filename: PDB文件名
                title: PDB文件标题
            """
            atoms = self.generate_detailed_coordinates()
            
            with open(filename, 'w') as f:
                f.write(f"HEADER    {title}\n")
                f.write(f"TITLE     {title}\n")
                f.write("EXPDTA    THEORETICAL MODEL\n")
                f.write("CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1\n")
                
                sequence_str = ''.join([res for res in self.main_chain_sequence])
                f.write(f"COMPND    MOL_ID: 1; MOLECULE: PEPTIDE; CHAIN: A; SEQRES: {len(self.main_chain_sequence)} {sequence_str};\n")
                f.write(f"SOURCE    SYNTHETIC PEPTIDE\n")
                
                for atom in atoms:
                    record_type, atom_num, atom_name, res_name, chain_id, res_seq, x, y, z, occupancy, temp_factor, element = atom
                    line = f"{record_type:<6}{atom_num:>5} {atom_name:>4} {res_name:>3} {chain_id}{res_seq:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}{occupancy:>6.2f}{temp_factor:>6.2f}           {element:>2}"
                    f.write(line + '\n')
                
                num_main_chain_residues = len(self.main_chain_sequence)
                
                cumulative_atom_counts = [0]
                current_count = 0
                for j in range(num_main_chain_residues):
                    residue_3 = self.AA_1TO3.get(self.main_chain_sequence[j], 'UNK')
                    total_atoms_in_residue = len(self.BACKBONE_ATOMS) + len(self.SIDE_CHAIN_ATOMS.get(residue_3, []))
                    current_count += total_atoms_in_residue
                    cumulative_atom_counts.append(current_count)
                
                for i in range(num_main_chain_residues - 1):
                    c_atom_idx = cumulative_atom_counts[i] + 2 + 1
                    n_atom_idx = cumulative_atom_counts[i+1] + 0 + 1
                    f.write(f"CONECT{c_atom_idx:>5}{n_atom_idx:>5}\n")
                
                f.write(f"TER   {len(atoms)+1:>5}      {self.AA_1TO3.get(self.main_chain_sequence[-1], 'UNK'):>3} A{num_main_chain_residues:>4}\n")
                f.write("END\n")
    
    @staticmethod
    def convert_xyz_to_detailed_pdb(xyz_data: np.ndarray, pdb_filename: str, title: str = "Detailed Protein Structure"):
        """
        将XYZ坐标数据转换为详细PDB文件（包含H、N、C、O和侧链原子）
        
        Args:
            xyz_data: 包含氨基酸类型和坐标的数组
            pdb_filename: 输出PDB文件名
            title: PDB文件标题
            
        Example:
            >>> import numpy as np
            >>> xyz_data = np.array([['A', 0.0, 0.0, 0.0], ['P', 1.0, 1.0, 1.0]])
            >>> ResultHandler.convert_xyz_to_detailed_pdb(xyz_data, 'output.pdb', 'My Protein')
        """
        if len(xyz_data) == 0:
            return
        
        residues = [row[0] for row in xyz_data]
        coords = np.array([[float(row[1]), float(row[2]), float(row[3])] for row in xyz_data])
        
        generator = ResultHandler.DetailedPDBGenerator(
            main_chain_positions=coords,
            side_chain_positions=[None] * len(coords),
            main_chain_sequence=''.join(residues)
        )
        
        generator.save_pdb_file(pdb_filename, title)
    
    @staticmethod
    def plot_protein_structure_3d_precise(atoms: List[Dict], sequence: str,
                                      filename: str, title: str = "Protein Structure",
                                      figsize=(12, 10)):
        """
        使用精确原子坐标绘制 3D 蛋白质结构图
        
        Args:
            atoms: 原子坐标列表 [{"name": "N", "coords": (x, y, z)}, ...]
            sequence: 氨基酸序列
            filename: 输出文件名
            title: 图表标题
            figsize: 图表大小
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        color_map = {
            'N': '#1f77b4',
            'CA': '#d62728',
            'C': '#2ca02c',
            'O': '#ff7f0e',
            'CB': '#9467bd'
        }
        
        x_coords = []
        y_coords = []
        z_coords = []
        colors = []
        sizes = []
        
        for atom in atoms:
            x, y, z = atom['coords']
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)
            colors.append(color_map.get(atom['name'], 'gray'))
            sizes.append(200 if atom['name'] == 'CA' else 120)
        
        scatter = ax.scatter(x_coords, y_coords, z_coords, c=colors, s=sizes, alpha=0.9,
                           edgecolors='black', linewidths=1.5)
        
        for i, atom in enumerate(atoms):
            name = atom['name']
            coords = atom['coords']
            
            if i + 1 < len(atoms):
                next_atom = atoms[i + 1]
                next_name = next_atom['name']
                next_coords = next_atom['coords']
                
                if name == 'N' and next_name == 'CA':
                    ax.plot([coords[0], next_coords[0]],
                           [coords[1], next_coords[1]],
                           [coords[2], next_coords[2]],
                           color='black', linewidth=4, alpha=0.9, zorder=1)
                
                elif name == 'CA' and next_name == 'CB':
                    ax.plot([coords[0], next_coords[0]],
                           [coords[1], next_coords[1]],
                           [coords[2], next_coords[2]],
                           color='#9467bd', linewidth=3, alpha=0.8, zorder=1)
                
                elif name == 'CA' and next_name == 'C':
                    ax.plot([coords[0], next_coords[0]],
                           [coords[1], next_coords[1]],
                           [coords[2], next_coords[2]],
                           color='black', linewidth=4, alpha=0.9, zorder=1)
                
                elif name == 'C' and next_name == 'O':
                    ax.plot([coords[0], next_coords[0]],
                           [coords[1], next_coords[1]],
                           [coords[2], next_coords[2]],
                           color='#ff7f0e', linewidth=3, alpha=0.8, zorder=1)
        
        c_indices = [i for i, atom in enumerate(atoms) if atom['name'] == 'C']
        n_indices = [i for i, atom in enumerate(atoms) if atom['name'] == 'N']
        
        for i in range(len(c_indices) - 1):
            c_idx = c_indices[i]
            n_idx = n_indices[i + 1]
            if c_idx < len(atoms) and n_idx < len(atoms):
                c_atom = atoms[c_idx]
                n_atom = atoms[n_idx]
                ax.plot([c_atom['coords'][0], n_atom['coords'][0]],
                       [c_atom['coords'][1], n_atom['coords'][1]],
                       [c_atom['coords'][2], n_atom['coords'][2]],
                       color='#d62728', linewidth=5, alpha=1.0, zorder=0)
        
        aa_colors = plt.cm.tab20(np.linspace(0, 1, len(sequence)))
        ca_indices = [i for i, atom in enumerate(atoms) if atom['name'] == 'CA']
        for idx, ca_idx in enumerate(ca_indices):
            if idx < len(sequence):
                x, y, z = x_coords[ca_idx], y_coords[ca_idx], z_coords[ca_idx]
                ax.text(x, y, z, f' {sequence[idx]}', fontsize=14, fontweight='bold',
                       color='black', bbox=dict(boxstyle='round,pad=0.3',
                                          facecolor='white',
                                          edgecolor='black',
                                          alpha=0.8))
        
        ax.set_xlabel('X (Å)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y (Å)', fontsize=14, fontweight='bold')
        ax.set_zlabel('Z (Å)', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.grid(True, linestyle='--', alpha=0.3)
        
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=color, markersize=10, label=name)
                      for name, color in color_map.items()]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        max_range = np.array([max(x_coords)-min(x_coords),
                             max(y_coords)-min(y_coords),
                             max(z_coords)-min(z_coords)]).max() / 2.0
        mid_x = (max(x_coords)+min(x_coords)) * 0.5
        mid_y = (max(y_coords)+min(y_coords)) * 0.5
        mid_z = (max(z_coords)+min(z_coords)) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()