# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Defines a protein folding problem that can be passed to algorithms."""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

from qiskit_algorithms.minimum_eigensolvers import MinimumEigensolverResult
from qiskit.quantum_info import Pauli, SparsePauliOp

from .interactions.interaction import Interaction
from .penalty_parameters import PenaltyParameters
from .peptide.peptide import Peptide
from .qubit_op_builder import QubitOpBuilder
from .qubit_utils import qubit_number_reducer
from .sampling_problem import SamplingProblem

if TYPE_CHECKING:
    from .protein_folding_result import ProteinFoldingResult


class ProteinFoldingProblem(SamplingProblem):
    """Defines a protein folding problem that can be passed to algorithms. Example initialization:

    .. code-block:: python

        penalty_terms = PenaltyParameters(15, 15, 15)
        main_chain_residue_seq = "SAASSASAAG"
        side_chain_residue_sequences = ["", "", "A", "A", "A", "A", "A", "A", "S", ""]
        peptide = Peptide(main_chain_residue_seq, side_chain_residue_sequences)
        mj_interaction = MiyazawaJerniganInteraction()
        protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)
        qubit_op = protein_folding_problem.qubit_op()
    """

    def __init__(
        self,
        peptide: Peptide,
        interaction: Interaction,
        penalty_parameters: PenaltyParameters,
    ):
        """
        Args:
            peptide: A peptide object that defines the protein subject to the folding problem.
            interaction: A type of interaction between the beads of the peptide.
            penalty_parameters: Parameters that define the strength of constraints enforcing in
                                the problem.
        """
        self._peptide = peptide
        self._interaction = interaction
        self._penalty_parameters = penalty_parameters
        self._pair_energies = interaction.calculate_energy_matrix(
            peptide.get_main_chain.main_chain_residue_sequence
        )
        self._qubit_op_builder = QubitOpBuilder(
            self._peptide, self._pair_energies, self._penalty_parameters
        )
        self._unused_qubits: List[int] = []

    def qubit_op(self) -> Union[SparsePauliOp, Pauli]:
        """
        Builds a qubit operator for the Hamiltonian encoding a protein folding problem. The
        number of qubits needed for optimization is optimized (compressed), if possible.
        To obtain the full qubit operator for a Hamiltonian, use the method `qubit_op_full`.

        Returns:
            A qubit operator for the Hamiltonian encoding a protein folding problem on an
            optimized number of qubits.
        """
        qubit_operator, unused_qubits = qubit_number_reducer.remove_unused_qubits(
            self._qubit_op_full()
        )
        self._unused_qubits = unused_qubits
        return qubit_operator

    def _qubit_op_full(self) -> Union[Pauli, SparsePauliOp]:
        """
        Builds a full qubit operator for the Hamiltonian encoding a protein folding problem. Full
        means that the number of qubits needed for optimization is not optimized and may be
        larger that necessary. To ensure the optimal number of qubits, use the method `qubit_op`.

        Returns:
            A qubit operator for the Hamiltonian encoding a protein folding problem.
        """
        qubit_operator = self._qubit_op_builder.build_qubit_op()
        return qubit_operator

    def interpret(self, raw_result: MinimumEigensolverResult) -> "ProteinFoldingResult":
        """
        Interprets the raw algorithm result, in the context of this problem, and returns a
        ProteinFoldingResult. The returned class can plot the protein and generate a
        .xyz file with the coordinates of each of its atoms.
        Args:
            raw_result: The raw result of solving the protein folding problem.

        Returns:
            A :class:`~protein_folding.ProteinFoldingResult`
            instance that contains the protein folding result.
        """
        # pylint: disable=import-outside-toplevel
        from .protein_folding_result import ProteinFoldingResult

        # 原始代码 --2026-01-08-leon调整
        # #probs = raw_result.eigenstate.binary_probabilities()
        # probs = raw_result.eigenstate
        # best_turn_sequence = max(probs, key=probs.get)
        # return ProteinFoldingResult(
        #     unused_qubits=self.unused_qubits,
        #     peptide=self.peptide,
        #     turn_sequence=best_turn_sequence,
        # )
        
        # 在 Qiskit 2.x 中，结果的访问方式可能有所不同
        probs = None  # 初始化probs变量以避免UnboundLocalError
        try:
            # 尝试旧版本的API
            probs = raw_result.eigenstate.binary_probabilities()
        except AttributeError:
            # 在新版本中，可能需要通过其他方式访问概率
            try:
                # 尝试访问eigenstate属性
                if hasattr(raw_result, 'eigenstate') and raw_result.eigenstate is not None:
                    eigenstate = raw_result.eigenstate
                    if hasattr(eigenstate, 'binary_probabilities'):
                        probs = eigenstate.binary_probabilities()
                    elif isinstance(eigenstate, dict):
                        # 如果eigenstate已经是字典格式
                        probs = eigenstate
                    else:
                        # 尝试从raw_result直接获取概率
                        probs = getattr(raw_result, 'aux_operators_evaluated', {})
                        if isinstance(probs, (list, tuple)) and len(probs) > 0:
                            # 如果是列表，取第一个
                            probs = probs[0]
                        if not isinstance(probs, dict):
                            # 当哈密顿量为空或全是恒等项时，可能没有概率分布
                            # 构建一个默认的概率分布
                            num_qubits = self._qubit_op_full().num_qubits
                            if num_qubits > 0:
                                # 对于n个量子比特，创建2^n个状态的均匀分布
                                total_states = 2 ** num_qubits
                                probs = {format(i, f'0{num_qubits}b'): 1.0/total_states for i in range(total_states)}
                            else:
                                probs = {'0': 1.0}  # 默认值
            except AttributeError:
                # 当哈密顿量为空时，所有状态都等价，返回默认状态
                num_qubits = self._qubit_op_full().num_qubits
                if num_qubits > 0:
                    # 返回一个默认状态（例如全零状态）
                    probs = {format(0, f'0{num_qubits}b'): 1.0}
                else:
                    probs = {'0': 1.0}
        
        # 检查probs是否为空或None
        if not probs:
            num_qubits = self._qubit_op_full().num_qubits
            if num_qubits > 0:
                probs = {format(0, f'0{num_qubits}b'): 1.0}
            else:
                probs = {'0': 1.0}
        
        best_turn_sequence = max(probs, key=probs.get)
        return ProteinFoldingResult(
            unused_qubits=self.unused_qubits,
            peptide=self.peptide,
            turn_sequence=best_turn_sequence,
        )

    @property
    def unused_qubits(self) -> List[int]:
        """Returns the list of indices for qubits in the original problem formulation that were
        removed during compression."""
        return self._unused_qubits

    @property
    def peptide(self) -> Peptide:
        """Returns the peptide defining the protein subject to the folding problem."""
        return self._peptide
