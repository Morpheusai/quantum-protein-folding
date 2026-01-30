import os
import sys
import csv
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 添加 QuPepFold 模块到 Python 路径
qupepfold_path = Path(__file__).parent / "QuPepFold" / "QuPepFold"
sys.path.insert(0, str(qupepfold_path))

from qupepfold.qupepfold import (
    generate_turn2qubit,
    count_interaction_qubits,
    build_mj_interactions,
    optimize_cvar_multistart,
    build_scalable_ansatz,
    statevector_fold_probs,
    exact_hamiltonian,
    turns_from_cfg_bits,
    dihedrals_from_turns,
    build_backbone_3d,
    write_pdb_with_conect,
    plot_energy_breakdown_for_most_negative,
)


def plot_protein_3d(atoms, seq, title, output_path):
    """
    绘制蛋白质 3D 结构图
    
    Args:
        atoms: 原子坐标列表 [{"name": "N", "coords": (x, y, z)}, ...]
        seq: 氨基酸序列
        title: 图表标题
        output_path: 输出文件路径
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 颜色映射
    color_map = {
        'N': '#1f77b4',      # 蓝色
        'CA': '#d62728',     # 红色
        'C': '#2ca02c',      # 绿色
        'O': '#ff7f0e',      # 橙色
        'CB': '#9467bd'      # 紫色
    }
    
    # 提取坐标
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
        # 增大原子尺寸，CA 原子更大
        sizes.append(200 if atom['name'] == 'CA' else 120)
    
    # 绘制原子（使用更大的尺寸和更高的透明度）
    scatter = ax.scatter(x_coords, y_coords, z_coords, c=colors, s=sizes, alpha=0.9, 
                       edgecolors='black', linewidths=1.5)
    
    # 绘制蛋白质骨架连接线（按照正确的化学键连接）
    # atoms 结构：对于甘氨酸G：N, CA, C, O；对于其他氨基酸：N, CA, CB, C, O
    
    # 首先构建原子索引映射
    atom_indices = {}
    for idx, atom in enumerate(atoms):
        name = atom['name']
        if name not in atom_indices:
            atom_indices[name] = []
        atom_indices[name].append(idx)
    
    # 连接规则：
    # 1. N -> CA (骨架) - 黑色粗线
    # 2. CA -> CB (侧链) - 紫色细线
    # 3. CA -> C (骨架) - 黑色粗线
    # 4. C -> O (羰基) - 橙色细线
    # 5. C -> 下一个氨基酸的 N (肽键) - 红色粗线
    
    # 遍历所有原子，建立连接
    for i, atom in enumerate(atoms):
        name = atom['name']
        coords = atom['coords']
        
        # 找到下一个原子
        if i + 1 < len(atoms):
            next_atom = atoms[i + 1]
            next_name = next_atom['name']
            next_coords = next_atom['coords']
            
            # 连接 N -> CA (骨架键)
            if name == 'N' and next_name == 'CA':
                ax.plot([coords[0], next_coords[0]], 
                       [coords[1], next_coords[1]], 
                       [coords[2], next_coords[2]], 
                       color='black', linewidth=4, alpha=0.9, zorder=1)
            
            # 连接 CA -> CB (侧链)
            elif name == 'CA' and next_name == 'CB':
                ax.plot([coords[0], next_coords[0]], 
                       [coords[1], next_coords[1]], 
                       [coords[2], next_coords[2]], 
                       color='#9467bd', linewidth=3, alpha=0.8, zorder=1)
            
            # 连接 CA -> C (骨架键)
            elif name == 'CA' and next_name == 'C':
                ax.plot([coords[0], next_coords[0]], 
                       [coords[1], next_coords[1]], 
                       [coords[2], next_coords[2]], 
                       color='black', linewidth=4, alpha=0.9, zorder=1)
            
            # 连接 C -> O (羰基)
            elif name == 'C' and next_name == 'O':
                ax.plot([coords[0], next_coords[0]], 
                       [coords[1], next_coords[1]], 
                       [coords[2], next_coords[2]], 
                       color='#ff7f0e', linewidth=3, alpha=0.8, zorder=1)
    
    # 连接肽键：C -> 下一个氨基酸的 N (红色粗线强调)
    # 找到所有 C 原子和 N 原子的索引
    c_indices = [i for i, atom in enumerate(atoms) if atom['name'] == 'C']
    n_indices = [i for i, atom in enumerate(atoms) if atom['name'] == 'N']
    
    # 连接每个 C 到下一个 N（除了最后一个 C）
    for i in range(len(c_indices) - 1):
        c_idx = c_indices[i]
        n_idx = n_indices[i + 1]  # 下一个氨基酸的 N
        if c_idx < len(atoms) and n_idx < len(atoms):
            c_atom = atoms[c_idx]
            n_atom = atoms[n_idx]
            ax.plot([c_atom['coords'][0], n_atom['coords'][0]], 
                   [c_atom['coords'][1], n_atom['coords'][1]], 
                   [c_atom['coords'][2], n_atom['coords'][2]], 
                   color='#d62728', linewidth=5, alpha=1.0, zorder=0)  # 肽键用红色粗线强调
    
    # 标记每个氨基酸的 CA 原子（使用更明显的标签）
    aa_colors = plt.cm.tab20(np.linspace(0, 1, len(seq)))
    ca_indices = [i for i, atom in enumerate(atoms) if atom['name'] == 'CA']
    for idx, ca_idx in enumerate(ca_indices):
        if idx < len(seq):
            x, y, z = x_coords[ca_idx], y_coords[ca_idx], z_coords[ca_idx]
            # 使用更大的字体和更明显的颜色
            ax.text(x, y, z, f' {seq[idx]}', fontsize=14, fontweight='bold', 
                   color='black', bbox=dict(boxstyle='round,pad=0.3', 
                                          facecolor='white', 
                                          edgecolor='black',
                                          alpha=0.8))
    
    # 设置标签和标题
    ax.set_xlabel('X (Å)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (Å)', fontsize=14, fontweight='bold')
    ax.set_zlabel('Z (Å)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 设置背景为白色，提高可读性
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 添加图例
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=color, markersize=10, label=name)
                      for name, color in color_map.items()]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # 设置相等的比例
    max_range = np.array([max(x_coords)-min(x_coords), 
                         max(y_coords)-min(y_coords), 
                         max(z_coords)-min(z_coords)]).max() / 2.0
    mid_x = (max(x_coords)+min(x_coords)) * 0.5
    mid_y = (max(y_coords)+min(y_coords)) * 0.5
    mid_z = (max(z_coords)+min(z_coords)) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 调整视角
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"3D structure plot -> {output_path}")


def main():
    parser = argparse.ArgumentParser(prog="run_qupepfold", description="Quantum Protein Folding Simulation")
    parser.add_argument("--seq", type=str,default='APRLRFY' , help="Protein sequence (2-10 aa, e.g., APRLRFY)")
    parser.add_argument("--tries", type=int, default=20, help="Number of CVaR multi-start attempts")
    parser.add_argument("--alpha", type=float, default=0.025, help="CVaR tail mass (0<alpha<1)")
    parser.add_argument("--shots", type=int, default=1024, help="(Informational) shots to report")
    parser.add_argument("--backend", default="local", choices=["local", "aws_sv1", "aws_garnet", "aws_forte"], help="Quantum backend (default: local)")
    args = parser.parse_args()

    seq = args.seq.upper()
    if not (2 <= len(seq) <= 10) or any(c not in "ARNDCEQGHILKMFPSTWYV" for c in seq):
        raise SystemExit("ERROR: --seq must be 2-10 amino acids using standard one-letter codes.")

    # Create output directory with timestamp and backend
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backend_name = args.backend
    output_dir = Path("./results") / f"{timestamp}_{backend_name}_qupepfold"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Build mapping & hyper (aligned with the core module)
    turn2qubit, fixed_bits, variable_bits = generate_turn2qubit(seq)
    num_q_cfg = turn2qubit.count("q")
    num_q_int = count_interaction_qubits(seq)
    hyper = {
        "protein": seq,
        "turn2qubit": turn2qubit,
        "numQubitsConfig": num_q_cfg,
        "numQubitsInteraction": num_q_int,
        "interactionEnergy": build_mj_interactions(seq),
        "numShots": int(args.shots),
    }

    print("=== Qubit mapping ===")
    print("turn2qubit:", turn2qubit)
    print("fixed bits:", fixed_bits)
    print("var bits:  ", variable_bits)
    print(f"cfg qubits: {num_q_cfg}  |  int qubits: {num_q_int}  |  total (incl. ancilla): {num_q_cfg+num_q_int+1}")

    # Optimize CVaR (multi-start)
    print(f"\n[CVaR-VQE] alpha={args.alpha}, tries={args.tries}")
    best_x, best_cvar, trace = optimize_cvar_multistart(hyper, args.tries, args.alpha)
    print(f"[CVaR-VQE] best CVaR energy: {best_cvar:.6f}")

    # Distribution at optimum (statevector)
    qc = build_scalable_ansatz(best_x, hyper, measure=False)
    probs = statevector_fold_probs(qc, hyper)
    states = list(probs.keys())
    energies = exact_hamiltonian(states, hyper)

    # Report: most probable & most negative-energy bitstrings
    s_most_prob = max(states, key=lambda s: probs[s])
    s_min_idx = int(min(range(len(states)), key=lambda i: energies[i]))
    s_min_energy = states[s_min_idx]

    print("\n=== Results at optimum ===")
    print(f"Most probable bitstring : {s_most_prob} (P={probs[s_most_prob]:.6f})")
    print(f"Lowest-energy bitstring : {s_min_energy} (E={energies[s_min_idx]:.6f})")

    # CSV dump (always enabled)
    csv_path = output_dir / "bitstring_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bitstring", "cfg_bits", "probability", "energy"])
        for s, e in zip(states, energies):
            w.writerow([s, s[:num_q_cfg], float(probs[s]), float(e)])
    print(f"\nWrote CSV -> {csv_path}")

    # Generate PDB files and visualizations for most probable and lowest-energy bitstrings
    print("\n=== Generating PDB files and visualizations ===")
    
    # Create pdb3d subdirectory
    pdb_dir = output_dir / "pdb3d"
    os.makedirs(pdb_dir, exist_ok=True)
    
    # Generate PDB for most probable bitstring
    cfg_bits_most_prob = s_most_prob[:num_q_cfg]
    turns_most_prob = turns_from_cfg_bits(cfg_bits_most_prob, turn2qubit)
    phis_most_prob, psis_most_prob = dihedrals_from_turns(turns_most_prob, len(seq))
    atoms_most_prob = build_backbone_3d(seq, phis_most_prob, psis_most_prob)
    pdb_path_most_prob = pdb_dir / f"fold3d_most_probable_{cfg_bits_most_prob}.pdb"
    write_pdb_with_conect(cfg_bits_most_prob, seq, atoms_most_prob, str(pdb_path_most_prob))
    print(f"Most probable PDB -> {pdb_path_most_prob}")
    
    # Generate 3D structure plot for most probable bitstring
    plot_path_most_prob = output_dir / f"3d_structure_most_probable_{cfg_bits_most_prob}.png"
    plot_protein_3d(atoms_most_prob, seq, 
                   f"Most Probable Structure - {seq} (P={probs[s_most_prob]:.6f})", 
                   str(plot_path_most_prob))
    
    # Generate PDB for lowest-energy bitstring
    cfg_bits_min_energy = s_min_energy[:num_q_cfg]
    turns_min_energy = turns_from_cfg_bits(cfg_bits_min_energy, turn2qubit)
    phis_min_energy, psis_min_energy = dihedrals_from_turns(turns_min_energy, len(seq))
    atoms_min_energy = build_backbone_3d(seq, phis_min_energy, psis_min_energy)
    pdb_path_min_energy = pdb_dir / f"fold3d_lowest_energy_{cfg_bits_min_energy}.pdb"
    write_pdb_with_conect(cfg_bits_min_energy, seq, atoms_min_energy, str(pdb_path_min_energy))
    print(f"Lowest-energy PDB -> {pdb_path_min_energy}")
    
    # Generate 3D structure plot for lowest-energy bitstring
    plot_path_min_energy = output_dir / f"3d_structure_lowest_energy_{cfg_bits_min_energy}.png"
    plot_protein_3d(atoms_min_energy, seq, 
                   f"Lowest Energy Structure - {seq} (E={energies[s_min_idx]:.6f})", 
                   str(plot_path_min_energy))
    
    # Generate energy breakdown plot for lowest-energy bitstring
    print("\nGenerating energy breakdown visualization...")
    plot_energy_breakdown_for_most_negative(probs, hyper, str(output_dir))
    print(f"Energy breakdown plot -> {output_dir / 'most_negative_energy_breakdown.png'}")


if __name__ == "__main__":
    main()
