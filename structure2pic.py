#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
蛋白质结构转图片工具 - 支持xyz格式和json格式输入
"""

import json
import sys
import os
import numpy as np
import argparse
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.protein_folding.utils.protein_plotter import ProteinPlotter


def load_xyz_coordinates(xyz_file_path):
    """
    从标准xyz格式文件加载坐标数据
    """
    coordinates = []
    labels = []
    
    with open(xyz_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # 跳过第一行（原子数）和第二行（注释）
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) >= 4:
            atom_type = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            coordinates.append([x, y, z])
            labels.append(atom_type)
    
    return np.array(coordinates), labels


def load_json_coordinates(json_file_path):
    """
    从JSON文件加载坐标数据
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    xyz_data = data['xyz_coordinates']
    coordinates = []
    labels = []
    
    for row in xyz_data:
        amino_acid = row[0]
        x, y, z = float(row[1]), float(row[2]), float(row[3])
        coordinates.append([x, y, z])
        labels.append(amino_acid)
    
    return np.array(coordinates), labels


def create_visualization_from_coordinates(coordinates, labels, title="Protein Structure", output_path=None):
    """
    从坐标数据创建可视化
    """
    # 创建一个简化的ProteinShapeFileGen模拟对象
    class MockProteinShapeFileGen:
        def __init__(self, positions, amino_acids):
            self.main_positions = positions
            self.main_chain_aminoacid_list = amino_acids
            # 侧链位置设为None（因为我们没有侧链数据）
            self.side_positions = [None] * len(amino_acids)
            self.side_chain_aminoacid_list = [''] * len(amino_acids)
        
        def get_xyz_data(self):
            return self.main_positions.tolist()
    
    # 创建模拟对象
    mock_shape_gen = MockProteinShapeFileGen(coordinates, labels)
    
    # 创建ProteinPlotter并生成图形
    plotter = ProteinPlotter(mock_shape_gen)
    
    # 获取图形
    fig = plotter.get_figure(title=title, ticks=False, grid=True)
    
    # 设置视角（与run_protein_folding.py中相同）
    if hasattr(fig, 'get_axes') and len(fig.get_axes()) > 0:
        fig.get_axes()[0].view_init(10, 70)
    
    # 保存图片
    import matplotlib.pyplot as plt
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 蛋白质结构图已保存到: {output_path}")
    
    # 关闭图形以释放内存
    plt.close(fig)
    
    return output_path


def process_input_file(input_path, output_path=None):
    """
    处理输入文件（xyz或json格式）
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    # 根据文件扩展名确定格式
    if input_path.suffix.lower() == '.xyz':
        # 处理xyz文件
        coordinates, labels = load_xyz_coordinates(input_path)
        title = f"Protein Structure - {input_path.stem}"
        
    elif input_path.suffix.lower() == '.json':
        # 处理json文件
        coordinates, labels = load_json_coordinates(input_path)
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        energy = data.get('energy', 0.0)
        title = f"Protein Structure - Result Energy: {energy:.4f}"
        
    else:
        raise ValueError(f"不支持的文件格式: {input_path.suffix}. 支持的格式: .xyz, .json")
    
    # 如果没有指定输出路径，根据输入文件生成输出路径
    if output_path is None:
        # 生成输出路径，保持在相同目录下
        output_path = input_path.parent / f"{input_path.stem}.png"
    
    # 创建可视化
    return create_visualization_from_coordinates(
        coordinates=coordinates,
        labels=labels,
        title=title,
        output_path=str(output_path)
    )


def main():
    parser = argparse.ArgumentParser(description='蛋白质结构转图片工具 - 支持xyz格式和json格式输入')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入文件路径 (支持.xyz和.json格式)')
    parser.add_argument('--output', '-o', type=str,
                        help='输出图片路径 (默认: 同目录下同名.png)')
    
    args = parser.parse_args()
    
    try:
        result_path = process_input_file(args.input, args.output)
        print(f"完成: {result_path}")
    except Exception as e:
        print(f"处理文件时出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()