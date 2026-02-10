#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标准蛋白质几何生成器 - 基于 run_qupepfold.py 的精确 3D 结构生成

优势：
1. 使用标准键长、键角、二面角
2. 物理准确性高
3. 生成真实的 PDB 文件
"""

import numpy as np
import math
from typing import List, Dict, Tuple


class ProteinGeometryBuilder:
    """标准蛋白质几何构建器"""

    _BOND = {
        "C-N": 1.329,
        "N-CA": 1.458,
        "CA-C": 1.525,
        "C=O": 1.229,
        "CA-CB": 1.53
    }

    _ANGLE = {
        "C-N-CA": math.radians(121.7),
        "N-CA-C": math.radians(110.4),
        "CA-C-N": math.radians(116.2),
        "CA-C-O": math.radians(120.8)
    }

    _OMEGA_TRANS = math.radians(180.0)

    _AA_1TO3 = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
        'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
        'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
        'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
    }

    _TURN_DIHEDRAL = {
        0: (math.radians(-60),  math.radians(-45)),
        1: (math.radians(-135), math.radians(135)),
        2: (math.radians(-75),  math.radians(145)),
        3: (math.radians(-60),  math.radians(140)),
    }

    def __init__(self, protein_sequence: str):
        """
        初始化蛋白质几何构建器

        Args:
            protein_sequence: 蛋白质序列
        """
        self.protein_sequence = protein_sequence
        self.N = len(protein_sequence)

    def build_3d_structure_from_turns(self, turns: List[int]) -> List[Dict]:
        """
        从转向序列构建 3D 结构

        Args:
            turns: 转向序列（长度 N-1）

        Returns:
            list: 原子坐标列表 [{"name": "N", "coords": (x, y, z)}, ...]
        """
        phis, psis = self._dihedrals_from_turns(turns)
        atoms = self._build_backbone_3d(phis, psis)
        return atoms

    def build_3d_structure_from_bitstring(self, bitstring: str, turn2qubit: str) -> List[Dict]:
        """
        从比特串构建 3D 结构

        Args:
            bitstring: 完整比特串
            turn2qubit: 转向到量子比特映射

        Returns:
            list: 原子坐标列表
        """
        cfg_bits = bitstring[:self._count_config_qubits(turn2qubit)]
        config = self._fill_config_bits(cfg_bits, turn2qubit)
        turns = [int(config[k:k+2], 2) for k in range(0, len(config), 2)]
        return self.build_3d_structure_from_turns(turns)

    def _dihedrals_from_turns(self, turns: List[int]) -> Tuple[List[float], List[float]]:
        """
        从转向序列生成二面角

        Returns:
            tuple: (phis, psis) 二面角列表
        """
        phis = [None] * self.N
        psis = [None] * self.N

        for i in range(self.N):
            t_prev = turns[i-1] if i-1 >= 0 and i-1 < len(turns) else 1
            t_next = turns[i] if i < len(turns) else 2

            phis[i] = self._TURN_DIHEDRAL[t_prev][0]
            psis[i] = self._TURN_DIHEDRAL[t_next][1]

        return phis, psis

    def _build_backbone_3d(self, phis: List[float], psis: List[float]) -> List[Dict]:
        """
        构建 3D 蛋白质骨架

        Args:
            phis: φ 二面角列表
            psis: ψ 二面角列表

        Returns:
            list: 原子坐标列表
        """
        N1, CA1, C1, O1 = self._seed_first_residue()
        atoms = [
            {"name": "N", "coords": N1},
            {"name": "CA", "coords": CA1},
            {"name": "C", "coords": C1},
            {"name": "O", "coords": O1}
        ]

        prevA, prevB, prevC = N1, CA1, C1

        for i in range(1, self.N):
            phi, psi = phis[i], psis[i]

            Ni = self._place_atom(prevA, prevB, prevC,
                               self._BOND["C-N"], self._ANGLE["CA-C-N"],
                               self._OMEGA_TRANS)
            CAi = self._place_atom(prevB, prevC, Ni,
                                self._BOND["N-CA"], self._ANGLE["C-N-CA"],
                                phi)
            Ci = self._place_atom(prevC, Ni, CAi,
                               self._BOND["CA-C"], self._ANGLE["N-CA-C"],
                               psi)
            Oi = self._place_atom(Ni, CAi, Ci,
                               self._BOND["C=O"], self._ANGLE["CA-C-O"],
                               0.0)

            atoms.extend([
                {"name": "N", "coords": Ni},
                {"name": "CA", "coords": CAi},
                {"name": "C", "coords": Ci},
                {"name": "O", "coords": Oi}
            ])

            prevA, prevB, prevC = Ni, CAi, Ci

        atoms = self._add_side_chains(atoms)
        return atoms

    def _seed_first_residue(self) -> Tuple[Tuple[float, float, float], ...]:
        """
        初始化第一个残基的坐标

        Returns:
            tuple: (N1, CA1, C1, O1) 坐标
        """
        N1 = (0.0, 0.0, 0.0)
        CA1 = (self._BOND["N-CA"], 0.0, 0.0)

        ang = self._ANGLE["N-CA-C"]
        vx, vy = -math.cos(ang), math.sin(ang)
        C1 = (CA1[0] + self._BOND["CA-C"] * vx,
               CA1[1] + self._BOND["CA-C"] * vy, 0.0)

        O1 = self._place_atom(N1, CA1, C1,
                           self._BOND["C=O"], self._ANGLE["CA-C-O"], 0.0)

        return N1, CA1, C1, O1

    def _place_atom(self, pA: Tuple[float, float, float],
                  pB: Tuple[float, float, float],
                  pC: Tuple[float, float, float],
                  bond_len: float, angle_rad: float,
                  dihedral_rad: float) -> Tuple[float, float, float]:
        """
        使用标准方法放置原子

        Args:
            pA, pB, pC: 三个参考原子的坐标
            bond_len: 键长
            angle_rad: 键角（弧度）
            dihedral_rad: 二面角（弧度）

        Returns:
            tuple: 新原子的坐标 (x, y, z)
        """
        m, n, cb = self._orthonormal_frame(pA, pB, pC)

        x = -bond_len * math.cos(angle_rad)
        y = bond_len * math.cos(dihedral_rad) * math.sin(angle_rad)
        z = bond_len * math.sin(dihedral_rad) * math.sin(angle_rad)

        C = np.asarray(pC, float)
        D = C + x * cb + y * m + z * n

        return tuple(D.tolist())

    def _orthonormal_frame(self, pA: Tuple[float, float, float],
                        pB: Tuple[float, float, float],
                        pC: Tuple[float, float, float]) -> Tuple[np.ndarray, ...]:
        """
        构建正交坐标系

        Returns:
            tuple: (m, n, cb) 三个正交向量
        """
        cb = self._normalize(np.asarray(pB) - np.asarray(pC))
        t = np.asarray(pB) - np.asarray(pA)
        n = np.cross(t, cb)

        if np.linalg.norm(n) < 1e-8:
            tmp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(tmp, cb)) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0])
            n = np.cross(tmp, cb)

        n = self._normalize(n)
        m = self._normalize(np.cross(n, cb))

        return m, n, cb

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        """归一化向量"""
        n = np.linalg.norm(v)
        return v * 0.0 if n < 1e-8 else v / n

    def _add_side_chains(self, atoms: List[Dict]) -> List[Dict]:
        """
        添加侧链原子（简化版）

        Args:
            atoms: 主链原子列表

        Returns:
            list: 包含侧链的完整原子列表
        """
        new_atoms = []
        atom_idx = 0

        for i, aa in enumerate(self.protein_sequence):
            Ni = atoms[atom_idx]["coords"]
            CAi = atoms[atom_idx + 1]["coords"]
            Ci = atoms[atom_idx + 2]["coords"]

            new_atoms.extend(atoms[atom_idx:atom_idx + 4])
            atom_idx += 4

            if aa == 'G':
                continue

            v1 = self._normalize(np.asarray(Ni) - np.asarray(CAi))
            v2 = self._normalize(np.asarray(Ci) - np.asarray(CAi))
            u = v1 + v2

            if np.linalg.norm(u) < 1e-8:
                tmp = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(tmp, v1)) > 0.9:
                    tmp = np.array([0.0, 1.0, 0.0])
                u = self._normalize(tmp)
            else:
                u = self._normalize(u)

            n = np.cross(v1, v2)
            if np.linalg.norm(n) < 1e-8:
                n = np.array([0.0, 0.0, 1.0])
            n = self._normalize(n)

            dir_cb = self._normalize(0.943 * u + 0.333 * n)
            CB = (np.asarray(CAi) + 1.53 * dir_cb).tolist()

            new_atoms.append({"name": "CB", "coords": tuple(CB)})

        return new_atoms

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

    def write_pdb_file(self, atoms: List[Dict], bitstring: str, filename: str):
        """
        写入 PDB 文件

        Args:
            atoms: 原子坐标列表
            bitstring: 比特串（用于标记）
            filename: 输出文件名
        """
        lines = []
        serial = 1
        resi = 1
        serial_map = []
        i = 0

        while resi <= self.N:
            aa = self.protein_sequence[resi - 1]
            resn = self._AA_1TO3.get(aa, "GLY")
            want = ["N", "CA", "CB", "C", "O"] if aa != "G" else ["N", "CA", "C", "O"]

            k = i
            res_serials = {"N": None, "CA": None, "CB": None, "C": None, "O": None}

            for name in want:
                found = None
                for j in range(k, min(k + 12, len(atoms))):
                    if atoms[j]["name"] == name:
                        found = atoms[j]
                        k = j + 1
                        break

                if found is None:
                    continue

                x, y, z = found["coords"]
                lines.append(f"ATOM  {serial:5d} {name:>3s}  {resn} A{resi:4d}    "
                           f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C")
                res_serials[name] = serial
                serial += 1

            serial_map.append(res_serials)
            i = max(i + 1, k)
            resi += 1

        nres = len(serial_map)
        for r in range(nres):
            s = serial_map[r]

            def add_conect(a, b):
                if a is not None and b is not None:
                    lines.append(f"CONECT{a:5d}{b:5d}")

            add_conect(s["N"], s["CA"])
            if s["CB"] is not None:
                add_conect(s["CA"], s["CB"])
            add_conect(s["CA"], s["C"])
            add_conect(s["C"], s["O"])

            if r < nres - 1:
                t = serial_map[r + 1]
                add_conect(s["C"], t["N"])

        lines.append(f"REMARK BITSTRING {bitstring}")
        lines.append("END")

        with open(filename, "w") as f:
            f.write("\n".join(lines))
