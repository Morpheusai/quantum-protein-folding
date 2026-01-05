#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
比较多次量子计算结果的构象差异，评估算法鲁棒性
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import glob
import datetime
from scipy.spatial.distance import pdist, squareform


def load_result_data(result_dir: str) -> List[Dict]:
    """
    从指定目录加载所有结果的JSON数据
    """
    if not os.path.exists(result_dir):
        print(f"错误: 目录 {result_dir} 不存在")
        return []
    
    json_files = glob.glob(os.path.join(result_dir, "result_*_parameters.json"))
    results = []
    
    if not json_files:
        print(f"警告: 在目录 {result_dir} 中未找到结果文件")
        return []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 验证必需的字段
                required_fields = ['energy', 'main_turns', 'side_turns', 'turn_sequence', 'xyz_coordinates']
                for field in required_fields:
                    if field not in data:
                        print(f"警告: 文件 {json_file} 缺少字段 {field}")
                        continue
                # 添加目录信息到结果中
                data['source_dir'] = result_dir
                results.append(data)
        except json.JSONDecodeError as e:
            print(f"警告: 无法解析JSON文件 {json_file}: {e}")
            continue
        except Exception as e:
            print(f"警告: 无法加载文件 {json_file}: {e}")
            continue
    
    # 按能量排序
    results.sort(key=lambda x: x['energy'])
    return results


def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    计算两个坐标集之间的RMSD（均方根偏差）
    """
    if coords1.shape != coords2.shape:
        raise ValueError(f"坐标形状不匹配: {coords1.shape} vs {coords2.shape}")
    
    diff = coords1 - coords2
    diff_squared = diff ** 2
    sum_squared_diff = np.sum(diff_squared)
    rmsd = np.sqrt(sum_squared_diff / len(coords1))
    return rmsd


def calculate_coordination_matrix(coords: np.ndarray) -> np.ndarray:
    """
    计算坐标矩阵，用于比较构象相似性
    """
    distances = squareform(pdist(coords))
    return distances


def compare_conformations(results: List[Dict]) -> Dict:
    """
    比较多个构象结果，返回差异分析
    """
    if len(results) < 2:
        raise ValueError("至少需要2个结果进行比较")
    
    # 提取坐标数据
    all_coords = []
    all_energies = []
    
    for result in results:
        coords = np.array([[float(x), float(y), float(z)] for _, x, y, z in result['xyz_coordinates']])
        all_coords.append(coords)
        all_energies.append(result['energy'])
    
    # 计算RMSD矩阵
    n_results = len(all_coords)
    rmsd_matrix = np.zeros((n_results, n_results))
    
    # 显示进度
    print(f"  - Calculating RMSD matrix for {n_results} results ({n_results*n_results} comparisons)")
    for i in range(n_results):
        for j in range(n_results):
            if i != j:
                rmsd_matrix[i, j] = calculate_rmsd(all_coords[i], all_coords[j])
            else:
                rmsd_matrix[i, j] = 0.0
        # 显示进度
        if (i + 1) % max(1, n_results // 10) == 0:  # 每10%显示一次进度
            print(f"    - Processed {i+1}/{n_results} results")
    
    # 计算构象差异统计
    avg_rmsd = np.mean(rmsd_matrix)
    std_rmsd = np.std(rmsd_matrix)
    min_rmsd = np.min(rmsd_matrix[rmsd_matrix != 0]) if np.any(rmsd_matrix != 0) else 0
    max_rmsd = np.max(rmsd_matrix)
    
    # 计算能量差异统计
    energy_diffs = []
    for i in range(len(all_energies)):
        for j in range(i+1, len(all_energies)):
            energy_diffs.append(abs(all_energies[i] - all_energies[j]))
    
    avg_energy_diff = np.mean(energy_diffs) if energy_diffs else 0
    std_energy_diff = np.std(energy_diffs) if energy_diffs else 0
    
    # 返回比较结果
    comparison_result = {
        'rmsd_matrix': rmsd_matrix.tolist(),
        'avg_rmsd': avg_rmsd,
        'std_rmsd': std_rmsd,
        'min_rmsd': min_rmsd,
        'max_rmsd': max_rmsd,
        'energies': all_energies,
        'avg_energy_diff': avg_energy_diff,
        'std_energy_diff': std_energy_diff,
        'conformations_count': n_results,
        'energy_range': max(all_energies) - min(all_energies)
    }
    
    return comparison_result


def evaluate_robustness(comparison_result: Dict) -> Dict:
    """
    评估算法鲁棒性
    """
    avg_rmsd = comparison_result['avg_rmsd']
    std_rmsd = comparison_result['std_rmsd']
    energy_range = comparison_result['energy_range']
    avg_energy_diff = comparison_result['avg_energy_diff']
    
    # 鲁棒性评估标准（可根据实际情况调整）
    rmsd_threshold = 2.0  # RMSD阈值，低于此值认为构象相似
    energy_threshold = 0.1  # 能量差异阈值
    
    robustness_score = 0
    robustness_indicators = []
    
    if avg_rmsd < rmsd_threshold:
        robustness_score += 25
        robustness_indicators.append(f"平均RMSD较低 ({avg_rmsd:.3f} < {rmsd_threshold}) - 构象一致性好")
    else:
        robustness_indicators.append(f"平均RMSD较高 ({avg_rmsd:.3f} >= {rmsd_threshold}) - 构象差异较大")
    
    if std_rmsd < rmsd_threshold / 2:
        robustness_score += 25
        robustness_indicators.append(f"RMSD标准差较小 ({std_rmsd:.3f}) - 结果稳定性好")
    else:
        robustness_indicators.append(f"RMSD标准差较大 ({std_rmsd:.3f}) - 结果波动性大")
    
    if energy_range < energy_threshold:
        robustness_score += 25
        robustness_indicators.append(f"能量范围较小 ({energy_range:.6f} < {energy_threshold}) - 能量收敛性好")
    else:
        robustness_indicators.append(f"能量范围较大 ({energy_range:.6f} >= {energy_threshold}) - 能量波动性大")
    
    if avg_energy_diff < energy_threshold:
        robustness_score += 25
        robustness_indicators.append(f"平均能量差异较小 ({avg_energy_diff:.6f}) - 结果一致性好")
    else:
        robustness_indicators.append(f"平均能量差异较大 ({avg_energy_diff:.6f}) - 结果差异性大")
    
    # 鲁棒性等级
    if robustness_score >= 75:
        robustness_level = "高"
    elif robustness_score >= 50:
        robustness_level = "中等"
    elif robustness_score >= 25:
        robustness_level = "低"
    else:
        robustness_level = "很差"
    
    return {
        'robustness_score': robustness_score,
        'robustness_level': robustness_level,
        'indicators': robustness_indicators,
        'avg_rmsd': avg_rmsd,
        'std_rmsd': std_rmsd,
        'energy_range': energy_range,
        'avg_energy_diff': avg_energy_diff
    }


def visualize_comparison(results: List[Dict], comparison_result: Dict, output_dir: str):
    """
    可视化比较结果
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. RMSD矩阵热图
    plt.figure(figsize=(10, 8))
    rmsd_matrix = np.array(comparison_result['rmsd_matrix'])
    plt.imshow(rmsd_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='RMSD')
    plt.title('RMSD Matrix of Conformation Differences')
    plt.xlabel('Result Index')
    plt.ylabel('Result Index')
    
    # 添加数值标签
    for i in range(len(rmsd_matrix)):
        for j in range(len(rmsd_matrix)):
            plt.text(j, i, f'{rmsd_matrix[i, j]:.2f}', 
                     ha='center', va='center', color='white', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rmsd_matrix_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 能量分布图
    energies = comparison_result['energies']
    plt.figure(figsize=(10, 6))
    
    # 使用不同颜色和标记来标识每个结果
    for i, energy in enumerate(energies):
        plt.scatter(i, energy, s=100, alpha=0.7, label=f'E={energy:.4f}')
    plt.plot(range(len(energies)), energies, 'b--', alpha=0.5, label='Energy Trend')
    
    plt.xlabel('Result Index')
    plt.ylabel('Energy')
    plt.title('Energy Distribution of Different Results')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    

    
    # 3. RMSD分布直方图
    rmsd_values = []
    for i in range(len(rmsd_matrix)):
        for j in range(i+1, len(rmsd_matrix)):
            rmsd_values.append(rmsd_matrix[i, j])
    
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(rmsd_values, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('RMSD')
    plt.ylabel('Frequency')
    plt.title('RMSD Distribution Histogram')
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息到图上
    plt.text(0.7, 0.9, f'Mean: {np.mean(rmsd_values):.3f}\nStd: {np.std(rmsd_values):.3f}\nMin: {np.min(rmsd_values):.3f}\nMax: {np.max(rmsd_values):.3f}', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rmsd_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3b. 结果能量分布（按RMSD排序）
    energies = comparison_result['energies']
    sorted_indices = sorted(range(len(energies)), key=lambda i: energies[i])
    sorted_energies = [energies[i] for i in sorted_indices]
    
    plt.figure(figsize=(max(10, len(energies)*0.5), 6))
    x_pos = range(len(sorted_energies))
    bars = plt.bar(x_pos, sorted_energies, alpha=0.7, color='lightcoral', edgecolor='black')
    
    # 在每个柱子上显示能量值
    for i, energy in enumerate(sorted_energies):
        plt.text(i, energy + max(sorted_energies) * 0.01, f'{energy:.4f}', 
                 ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Result Rank (by Energy)')
    plt.ylabel('Energy')
    plt.title('Energy Distribution (Ranked by Energy)')
    plt.xticks(x_pos, [f'{i+1}' for i in x_pos], rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_ranked.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 能量 vs RMSD 散点图
    plt.figure(figsize=(10, 6))
    avg_rmsds = []
    for i in range(len(rmsd_matrix)):
        avg_rmsd = np.mean(rmsd_matrix[i])
        avg_rmsds.append(avg_rmsd)
    
    energies = comparison_result['energies']
    plt.scatter(energies, avg_rmsds, s=100, alpha=0.7, c='red', edgecolors='black')
    
    # 在每个点旁边添加结果标签，显示能量值
    for i, (energy, avg_rmsd) in enumerate(zip(energies, avg_rmsds)):
        plt.annotate(f'E={energy:.4f}', (energy, avg_rmsd), 
                     textcoords="offset points", xytext=(5,5), ha='left', fontsize=9)
    
    plt.xlabel('Energy')
    plt.ylabel('Average RMSD')
    plt.title('Energy vs Average RMSD')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_vs_rmsd.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_report(results: List[Dict], comparison_result: Dict, robustness_result: Dict, output_dir: str, top_n: int = 3):
    """
    生成比较报告
    """
    report_path = os.path.join(output_dir, 'comparison_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Quantum Computing Results Conformation Difference Comparison Report\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Number of analyzed results: {comparison_result['conformations_count']}\n")
        f.write(f"Energy range: {comparison_result['energy_range']:.6f}\n")
        f.write(f"Average energy difference: {comparison_result['avg_energy_diff']:.6f} ± {comparison_result['std_energy_diff']:.6f}\n\n")
        
        f.write("Conformation difference analysis:\n")
        f.write(f"  - Average RMSD: {comparison_result['avg_rmsd']:.4f}\n")
        f.write(f"  - RMSD std: {comparison_result['std_rmsd']:.4f}\n")
        f.write(f"  - Min RMSD: {comparison_result['min_rmsd']:.4f}\n")
        f.write(f"  - Max RMSD: {comparison_result['max_rmsd']:.4f}\n\n")
        
        f.write("Robustness evaluation:\n")
        f.write(f"  - Robustness score: {robustness_result['robustness_score']}/100\n")
        f.write(f"  - Robustness level: {robustness_result['robustness_level']}\n\n")
        
        f.write("Robustness indicators analysis:\n")
        for indicator in robustness_result['indicators']:
            f.write(f"  - {indicator}\n")
        
        # 添加最佳结果排名
        f.write("\n\nTop N Best Results (by energy):\n")
        f.write("-" * 40 + "\n")
        sorted_results = sorted(enumerate(results), key=lambda x: x[1]['energy'])
        for rank, (idx, result) in enumerate(sorted_results[:top_n]):  # 使用top_n参数
            f.write(f"Rank {rank+1}: Energy {result['energy']:.6f} from {os.path.basename(result['source_dir'])}\n")
            f.write(f"  - Result Index: {idx+1}\n")
            f.write(f"  - Main chain turns: {result['main_turns']}\n")
            f.write(f"  - Side chain turns: {result['side_turns']}\n")
            f.write(f"  - Turn sequence: {result['turn_sequence']}\n")
            f.write("\n")
        
        f.write("\n\nAll Results Details:\n")
        f.write("-" * 30 + "\n")
        for i, result in enumerate(results):
            f.write(f"Result {i+1} from {os.path.basename(result['source_dir'])}:\n")
            f.write(f"  - Energy: {result['energy']:.6f}\n")
            f.write(f"  - Main chain turns: {result['main_turns']}\n")
            f.write(f"  - Side chain turns: {result['side_turns']}\n")
            f.write(f"  - Turn sequence: {result['turn_sequence']}\n")
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare conformation differences of quantum computing results and evaluate algorithm robustness',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s --result_dirs results/20260106_103729_local results/20260106_103830_local
  %(prog)s --top_n 5  # Compare all result directories in results/ with top 5 results
  %(prog)s  # Compare all result directories in results/
        '''
    )
    parser.add_argument('--result_dirs', nargs='*', 
                        help='List of result directory paths to compare (default: all subdirectories in results directory)')
    parser.add_argument('--top_n', type=int, default=3, 
                        help='Number of top results to display (default: 3)')
    args = parser.parse_args()
    
    print("Comparing conformation differences of quantum computing results...")
    
    # 如果没有指定目录，则使用results目录下的所有子目录
    if not args.result_dirs:
        results_dir = 'results'
        if os.path.exists(results_dir):
            args.result_dirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) 
                                if os.path.isdir(os.path.join(results_dir, d))]
            print(f"No directories specified, analyzing {len(args.result_dirs)} subdirectories in '{results_dir}'")
        else:
            print(f"Error: Directory '{results_dir}' does not exist")
            return False
    
    # 加载所有结果
    all_results = []
    for result_dir in args.result_dirs:
        print(f"Loading results from directory {result_dir}...")
        results = load_result_data(result_dir)
        all_results.extend(results)
        print(f"  - Loaded {len(results)} results from {result_dir}")
    
    if len(all_results) < 2:
        print("Error: Need at least 2 results for comparison")
        return False
    
    print(f"Total loaded {len(all_results)} results")
    
    # 按能量排序
    all_results.sort(key=lambda x: x['energy'])
    
    # 比较构象
    print("Calculating conformation differences...")
    try:
        comparison_result = compare_conformations(all_results)
    except Exception as e:
        print(f"Error calculating conformation differences: {e}")
        return False
    
    # 评估鲁棒性
    print("Evaluating algorithm robustness...")
    try:
        robustness_result = evaluate_robustness(comparison_result)
    except Exception as e:
        print(f"Error evaluating robustness: {e}")
        return False
    
    # 创建输出目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('results', f'comparison_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 可视化
    print("Generating visualization charts...")
    try:
        visualize_comparison(all_results, comparison_result, output_dir)
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        return False
    
    # 生成报告
    print("Generating comparison report...")
    try:
        generate_report(all_results, comparison_result, robustness_result, output_dir, args.top_n)
    except Exception as e:
        print(f"Error generating report: {e}")
        return False
    
    # 输出摘要
    print("\nComparison Summary:")
    print(f"  - Number of analyzed results: {comparison_result['conformations_count']}")
    print(f"  - Average RMSD: {comparison_result['avg_rmsd']:.4f}")
    print(f"  - Energy range: {comparison_result['energy_range']:.6f}")
    print(f"  - Robustness score: {robustness_result['robustness_score']}/100 ({robustness_result['robustness_level']})")
    
    # 输出前N个最优结果
    print(f"\nTop {args.top_n} Best Results (by energy):")
    sorted_results = sorted(enumerate(all_results), key=lambda x: x[1]['energy'])
    for rank, (idx, result) in enumerate(sorted_results[:args.top_n]):
        print(f"  Rank {rank+1}: Energy {result['energy']:.6f} from {os.path.basename(result['source_dir'])}")
        print(f"    - Result Index: {idx+1}")
        print(f"    - Main chain turns: {result['main_turns']}")
        print(f"    - Side chain turns: {result['side_turns']}")
        print(f"    - Turn sequence: {result['turn_sequence']}")
        print()
    
    print(f"\nFull report and charts saved to: {output_dir}")
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Conformation difference comparison completed!")
    else:
        print("\n❌ Conformation difference comparison failed.")
        exit(1)