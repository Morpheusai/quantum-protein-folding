#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
精确能量计算器 - 基于 run_qupepfold.py 的精确计算方法

优势：
1. 精确数学计算，无数值误差
2. 物理意义明确
3. 易于验证和调试
"""

import numpy as np
from typing import Dict, List, Tuple


class PreciseEnergyCalculator:
    """精确能量计算器"""

    _LAM_DIS = 720.0
    _LAM_LOC = 20.0
    _LAM_BACK = 50.0

    _TURN_DIHEDRAL = {
        0: (np.radians(-60),  np.radians(-45)),
        1: (np.radians(-135), np.radians(135)),
        2: (np.radians(-75),  np.radians(145)),
        3: (np.radians(-60),  np.radians(140)),
    }

    def __init__(self, protein_sequence: str, interaction_matrix: np.ndarray):
        """
        初始化精确能量计算器

        Args:
            protein_sequence: 蛋白质序列
            interaction_matrix: 相互作用能量矩阵 (N x N)
        """
        self.protein_sequence = protein_sequence
        self.interaction_matrix = interaction_matrix
        self.N = len(protein_sequence)

    def calculate_energy_from_bitstring(self, bitstring: str, turn2qubit: str) -> float:
        """
        从比特串计算精确能量

        Args:
            bitstring: 完整比特串（配置 + 相互作用）
            turn2qubit: 转向到量子比特映射模板

        Returns:
            float: 精确能量值
        """
        cfg_bits = bitstring[:self._count_config_qubits(turn2qubit)]
        config = self._fill_config_bits(cfg_bits, turn2qubit)
        turns = [int(config[k:k+2], 2) for k in range(0, len(config), 2)]

        energy = 0.0

        energy += self._LAM_BACK * sum(1 for a, b in zip(turns[:-1], turns[1:]) if a == b)

        q = self._count_config_qubits(turn2qubit)
        for i in range(0, self.N - 4):
            for j in range(i + 5, self.N, 2):
                if q >= len(bitstring):
                    continue

                ctrl = bitstring[q]
                q += 1

                if ctrl == '0':
                    continue

                energy += float(self.interaction_matrix[i, j])

                delta_vec = self._compute_delta_vector(turns, i, j)

                dij = np.linalg.norm(delta_vec) ** 2
                energy += self._LAM_DIS * (dij - 1.0)

                if j - 1 > i:
                    dir_vec = self._compute_delta_vector(turns, i, j - 1)
                    energy += self._LAM_LOC * (2.0 - np.linalg.norm(dir_vec) ** 2)

                if j > i + 1:
                    dmj_vec = self._compute_delta_vector(turns, i + 1, j)
                    energy += self._LAM_LOC * (2.0 - np.linalg.norm(dmj_vec) ** 2)

                if i - 1 >= 0:
                    dmj2_vec = self._compute_delta_vector(turns, i - 1, j)
                    energy += self._LAM_LOC * (2.0 - np.linalg.norm(dmj2_vec) ** 2)

                if j + 1 <= self.N - 1:
                    dir2_vec = self._compute_delta_vector(turns, i, j)
                    energy += self._LAM_LOC * (2.0 - np.linalg.norm(dir2_vec) ** 2)

        return energy

    def calculate_energy_breakdown(self, bitstring: str, turn2qubit: str) -> Dict[str, float]:
        """
        计算能量分解（用于分析）

        Returns:
            dict: 包含各项能量的字典
                - backbone: 回溯惩罚
                - mj: MJ 相互作用能量
                - distance: 距离惩罚
                - locality: 局部性惩罚
                - total: 总能量
        """
        cfg_bits = bitstring[:self._count_config_qubits(turn2qubit)]
        config = self._fill_config_bits(cfg_bits, turn2qubit)
        turns = [int(config[k:k+2], 2) for k in range(0, len(config), 2)]

        comp = {
            "backbone": 0.0,
            "mj": 0.0,
            "distance": 0.0,
            "locality": 0.0
        }

        comp["backbone"] = self._LAM_BACK * sum(1 for a, b in zip(turns[:-1], turns[1:]) if a == b)

        q = self._count_config_qubits(turn2qubit)
        for i in range(0, self.N - 4):
            for j in range(i + 5, self.N, 2):
                if q >= len(bitstring):
                    continue

                ctrl = bitstring[q]
                q += 1

                if ctrl == '0':
                    continue

                comp["mj"] += float(self.interaction_matrix[i, j])

                delta_vec = self._compute_delta_vector(turns, i, j)
                dij = np.linalg.norm(delta_vec) ** 2
                comp["distance"] += self._LAM_DIS * (dij - 1.0)

                if j - 1 > i:
                    dir_vec = self._compute_delta_vector(turns, i, j - 1)
                    comp["locality"] += self._LAM_LOC * (2.0 - np.linalg.norm(dir_vec) ** 2)

                if j > i + 1:
                    dmj_vec = self._compute_delta_vector(turns, i + 1, j)
                    comp["locality"] += self._LAM_LOC * (2.0 - np.linalg.norm(dmj_vec) ** 2)

                if i - 1 >= 0:
                    dmj2_vec = self._compute_delta_vector(turns, i - 1, j)
                    comp["locality"] += self._LAM_LOC * (2.0 - np.linalg.norm(dmj2_vec) ** 2)

                if j + 1 <= self.N - 1:
                    dir2_vec = self._compute_delta_vector(turns, i, j)
                    comp["locality"] += self._LAM_LOC * (2.0 - np.linalg.norm(dir2_vec) ** 2)

        comp["total"] = sum(comp.values())
        return comp

    def calculate_cvar_energy_precise(self, counts: dict, turn2qubit: str, alpha: float = 0.1) -> float:
        """
        使用精确哈密顿量计算 CVaR 能量

        Args:
            counts: 量子测量结果 {bitstring: count}
            turn2qubit: 转向到量子比特映射
            alpha: CVaR 参数

        Returns:
            float: CVaR 能量值
        """
        energies = []
        for bitstring, count in counts.items():
            e = self.calculate_energy_from_bitstring(bitstring, turn2qubit)
            energies.extend([e] * count)

        energies.sort()
        total_shots = sum(counts.values())
        num_keep = max(1, int(total_shots * alpha))

        return np.mean(energies[:num_keep])

    def extract_top_results_precise(self, counts: dict, turn2qubit: str, max_results: int = 1) -> list:
        """
        使用精确哈密顿量提取前 N 个最优结果

        Returns:
            list: [(bitstring, energy, count), ...]
        """
        bitstring_energies = []
        for bitstring, count in counts.items():
            e = self.calculate_energy_from_bitstring(bitstring, turn2qubit)
            bitstring_energies.append((bitstring, e, count))

        bitstring_energies.sort(key=lambda x: x[1])
        return bitstring_energies[:max_results]

    def _count_config_qubits(self, turn2qubit: str) -> int:
        """计算配置量子比特数量"""
        return turn2qubit.count('q')

    def _fill_config_bits(self, cfg_bits: str, turn2qubit: str) -> str:
        """填充配置模板"""
        cfg = list(turn2qubit)
        qpos = [i for i, ch in enumerate(cfg) if ch == 'q']

        if len(cfg_bits) != len(qpos):
            raise ValueError(f"cfg_bits length {len(cfg_bits)} != # of 'q' in template {len(qpos)}")

        for i, pos in enumerate(qpos):
            cfg[pos] = cfg_bits[i]

        expanded = ''.join(cfg)
        assert 'q' not in expanded, "Unfilled 'q' in expanded config."
        return expanded

    def _compute_delta_vector(self, turns: List[int], a: int, b: int) -> np.ndarray:
        """
        计算转向向量（距离计算的核心）

        Args:
            turns: 转向序列
            a: 起始索引
            b: 结束索引

        Returns:
            np.ndarray: 4D 转向向量
        """
        seg = turns[a:b]
        vec = np.zeros(4)

        for k in range(4):
            mask = np.array([1 if t == k else 0 for t in seg], float)
            if mask.size:
                vec[k] = np.sum(((-1) ** np.arange(mask.size)) * mask)

        return vec
