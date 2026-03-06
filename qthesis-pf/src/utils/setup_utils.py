"""Module providing setup utilities for protein folding quantum simulation, including Hamiltonian construction, VQE setup, and result processing."""

import sys
import os
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit.circuit.library import real_amplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import SamplingMinimumEigensolverResult, SamplingVQE
from qiskit_algorithms.optimizers import COBYLA

from backend import get_sampler
from builder import HamiltonianBuilder
from constants import DEFAULT_TIMEZONE, INTERACTION_TYPE, RESULTS_DATA_DIRPATH
from contact import ContactMap
from distance import DistanceMap
from enums import InteractionType
from exceptions import InvalidInteractionTypeError
from interaction import HPInteraction, Interaction, MJInteraction
from logger import get_logger
from protein import Protein
from result.interpreter import ResultInterpreter
from result.visualizer import ResultVisualizer
from utils.qubit_utils import remove_unused_qubits

if TYPE_CHECKING:
    from pathlib import Path

    from qiskit import QuantumCircuit

logger = get_logger()


def setup_folding_system(
    main_chain: str, side_chain: str
) -> tuple[Protein, Interaction, ContactMap, DistanceMap]:
    """Setup the protein folding system components.

    Args:
        main_chain (str): Main chain protein sequence.
        side_chain (str): Side chain protein sequence.

    Returns:
        tuple[Protein, Interaction, ContactMap, DistanceMap]: The protein, interaction model,
        contact map, and distance map.

    Raises:
        InvalidInteractionTypeError: If the interaction type is invalid (class not inheriting from Interaction).

    """
    if INTERACTION_TYPE == InteractionType.MJ:
        interaction: Interaction = MJInteraction()
    elif INTERACTION_TYPE == InteractionType.HP:
        interaction: Interaction = HPInteraction()
    else:
        raise InvalidInteractionTypeError

    protein = Protein(
        main_protein_sequence=main_chain,
        side_protein_sequence=side_chain,
        valid_symbols=interaction.valid_symbols,
    )

    contact_map = ContactMap(protein=protein)
    distance_map = DistanceMap(protein=protein)

    return protein, interaction, contact_map, distance_map


def build_and_compress_hamiltonian(
    protein: Protein,
    interaction: Interaction,
    contact_map: ContactMap,
    distance_map: DistanceMap,
) -> tuple[SparsePauliOp, SparsePauliOp]:
    """Build and compress the final Hamiltonian for the protein folding system.

    Args:
        protein (Protein): The protein instance.
        interaction (Interaction): The interaction model.
        contact_map (ContactMap): The contact map.
        distance_map (DistanceMap): The distance map.

    Returns:
        tuple[SparsePauliOp, SparsePauliOp]: The original and compressed Hamiltonians

    """
    h_builder = HamiltonianBuilder(
        protein=protein,
        interaction=interaction,
        distance_map=distance_map,
        contact_map=contact_map,
    )

    hamiltonian = h_builder.sum_hamiltonians()
    logger.info("Original hamiltonian qubits: %s", hamiltonian.num_qubits)

    compressed_h = remove_unused_qubits(hamiltonian)
    logger.info("Compressed hamiltonian qubits: %s", compressed_h.num_qubits)

    return hamiltonian, compressed_h


def setup_vqe_optimization(
    num_qubits: int,
    shots: int = 100,
    max_iterations: int = 50,
) -> tuple[SamplingVQE, list[int], list[float], dict[str, Any]]:
    """Setup the VQE optimization process.

    Args:
        num_qubits (int): Number of qubits for the ansatz.
        shots (int): Number of shots per iteration.
        max_iterations (int): Maximum number of optimization iterations.

    Returns:
        tuple[SamplingVQE, list[int], list[float], dict[str, Any]]: The VQE instance, evaluation counts (iterations), 
        their respective energy values, and circuit information dictionary.

    """
    optimizer = COBYLA(maxiter=max_iterations)

    ansatz: QuantumCircuit = real_amplitudes(num_qubits=num_qubits, reps=1)

    counts: list[int] = []
    values: list[float] = []
    
    circuit_info: dict[str, Any] = {
        'total_gates': [],
        'single_qubit_gates': [],
        'two_qubit_gates': [],
        'circuit_depth': [],
        'shots': [],
        'cumulative_shots': [],
        '_last_callback_time': time.perf_counter()
    }

    def _store_intermediate_result(
        eval_count: int,
        _parameters: np.ndarray[Any, Any],
        mean: float,
        _std: dict[str, Any],
    ) -> None:
        """Callback to store intermediate VQE results."""
        counts.append(eval_count)
        values.append(mean)
        
        decomposed_circuit = ansatz.decompose()
        total_gates = decomposed_circuit.size()
        single_qubit_gates = sum(1 for op in decomposed_circuit.data if len(op.qubits) == 1)
        two_qubit_gates = sum(1 for op in decomposed_circuit.data if len(op.qubits) == 2)
        circuit_depth = decomposed_circuit.depth()
        
        circuit_info['total_gates'].append(total_gates)
        circuit_info['single_qubit_gates'].append(single_qubit_gates)
        circuit_info['two_qubit_gates'].append(two_qubit_gates)
        circuit_info['circuit_depth'].append(circuit_depth)
        circuit_info['shots'].append(shots)
        if 'timing' not in circuit_info:
            circuit_info['timing'] = []
            
        current_time = time.perf_counter()
        if circuit_info.get('_last_callback_time'):
            iter_time = current_time - circuit_info['_last_callback_time']
            # qthesis does not use a wrapped estimator, so we default to setting the entire callback time interval
            # The actual quantum_time vs queue_time extraction logic would require wrapping the inner BaseSampler.
            # To be safe and avoid breaking qiskit algorithms, we record iter_time as classical_time for the local runs
            timing_info = {
                "quantum_time": 0.0,
                "queue_time": 0.0,
                "classical_time": round(iter_time, 3),
                "total_time": round(iter_time, 3)
            }
            circuit_info['timing'].append(timing_info)
        else:
            circuit_info['timing'].append({
                "quantum_time": 0.0,
                "queue_time": 0.0,
                "classical_time": 0.0,
                "total_time": 0.0
            })
            
        circuit_info['_last_callback_time'] = current_time
        circuit_info['cumulative_shots'].append(shots * len(counts))

    sampler, backend = get_sampler()
    
    # Native timing tracker wrapper
    class LocalQuantumTimeTracker:
        def __init__(self):
            self.submit_time = 0.0
            self.iteration_start = 0.0
            self.result_received = 0.0
            self.iteration_end = 0.0
            self.quantum_time = 0.0
            self.queue_time = 0.0
            self.classical_time = 0.0
            self.total_time = 0.0
            
        def extract_from_job(self, job, backend_name):
            """从任务结果中提取量子时间、队列时间和经典时间。
            
            支持多种后端类型：
            - AWS Braket 模拟器 (SV1, Garnet, Forte)
            - IBM 量子硬件
            - 本地模拟器
            
            Args:
                job: 执行的任务对象
                backend_name: 后端名称字符串
            """
            backend_name_str = str(backend_name).lower()
            
            # 第一步：尝试从 AWS Braket 任务元数据获取精确时间戳
            if "braket" in backend_name_str or "sv1" in backend_name_str or "garnet" in backend_name_str or "forte" in backend_name_str:
                try:
                    result = job.result()
                    # 检查是否有 Braket 任务的元数据
                    metadata = {}
                    if hasattr(result, '_metadata') and result._metadata:
                        metadata = result._metadata
                    elif hasattr(result, 'metadata') and result.metadata:
                        metadata = result.metadata
                    
                    if metadata:
                        # 尝试从 timestamps 提取时间（Braket SDK v0.23+）
                        timestamps = metadata.get('timestamps', {})
                        if isinstance(timestamps, dict) and timestamps:
                            from datetime import datetime as dt_module
                            
                            created = timestamps.get('created')
                            running = timestamps.get('running')
                            ended = timestamps.get('ended') or timestamps.get('completed')
                            
                            # 计算队列时间（created -> running）
                            if created and running:
                                try:
                                    created_dt = dt_module.fromisoformat(created.replace('Z', '+00:00'))
                                    running_dt = dt_module.fromisoformat(running.replace('Z', '+00:00'))
                                    self.queue_time = (running_dt - created_dt).total_seconds()
                                except Exception:
                                    pass
                            
                            # 计算量子时间（running -> ended）
                            if running and ended:
                                try:
                                    running_dt = dt_module.fromisoformat(running.replace('Z', '+00:00'))
                                    ended_dt = dt_module.fromisoformat(ended.replace('Z', '+00:00'))
                                    self.quantum_time = (ended_dt - running_dt).total_seconds()
                                except Exception:
                                    pass
                        
                        # 如果 metadata 直接提供了 quantum_time 或 queue_time
                        if self.quantum_time == 0.0:
                            self.quantum_time = metadata.get('quantum_time', 0.0)
                        if self.queue_time == 0.0:
                            self.queue_time = metadata.get('queue_time', 0.0)
                            
                except Exception:
                    # 静默失败，继续下面的回退逻辑
                    pass
            
            # 第二步：尝试从 IBM Qiskit Runtime 获取时间
            if self.quantum_time == 0.0:
                try:
                    metrics = job.metrics()
                    if metrics:
                        # IBM Qiskit Runtime 格式
                        self.quantum_time = metrics.get('usage', {}).get('quantum_seconds', 0.0)
                        timestamps = metrics.get('timestamps', {})
                        if isinstance(timestamps, dict) and 'running' in timestamps and 'created' in timestamps:
                            running_time = datetime.fromisoformat(timestamps['running'].replace('Z', '+00:00'))
                            created_time = datetime.fromisoformat(timestamps['created'].replace('Z', '+00:00'))
                            self.queue_time = (running_time - created_time).total_seconds()
                except Exception:
                    pass
            
            # 第三步：基于后端类型的回退策略
            if self.quantum_time == 0.0:
                try:
                    # 对于模拟器和云后端，如果没有明确的量子时间元数据
                    # 我们根据后端类型估算时间分配
                    
                    if ("sv1" in backend_name_str or 
                        "garnet" in backend_name_str or 
                        "forte" in backend_name_str or
                        "braket" in backend_name_str or
                        "aws" in backend_name_str):
                        # AWS Braket 后端（包括 SV1 模拟器和其他量子设备）
                        # AWS SV1 是状态向量模拟器，执行时间主要是量子模拟时间
                        # Garnet 和 Forte 是真实量子硬件，但作为模拟器运行时也应类似处理
                        if self.result_received > 0 and self.submit_time > 0:
                            total_execution = self.result_received - self.submit_time
                            # 模拟器的经验法则：95% 算作量子时间，5% 算作经典开销
                            self.quantum_time = total_execution * 0.95
                            self.queue_time = 0.0  # 模拟器通常没有队列时间
                            
                    elif ("local" in backend_name_str or 
                          "statevector" in backend_name_str or
                          "sim" in backend_name_str or
                          "qasm" in backend_name_str):
                        # 本地模拟器（Qiskit Aer 等）
                        # 全部执行时间都算作量子时间
                        if self.result_received > 0 and self.submit_time > 0:
                            self.quantum_time = self.result_received - self.submit_time
                            
                    else:
                        # 其他后端（可能是真实量子硬件）
                        # 尝试从结果元数据提取
                        result = job.result()
                        if hasattr(result, '_metadata') and result._metadata:
                            self.quantum_time = result._metadata.get('quantum_time', 0.0)
                            self.queue_time = result._metadata.get('queue_time', 0.0)
                        elif hasattr(result, 'metadata') and result.metadata:
                            self.quantum_time = result.metadata.get('quantum_time', 0.0)
                            self.queue_time = result.metadata.get('queue_time', 0.0)
                            
                except Exception:
                    # 所有尝试都失败，保持默认值 0
                    pass
                    
            # 第四步：计算总时间和经典时间
            if self.iteration_end > 0 and self.iteration_start > 0:
                self.total_time = self.iteration_end - self.iteration_start
                # 经典时间 = 总时间 - 量子时间 - 队列时间
                # 确保不会出现负数
                self.classical_time = max(0.0, self.total_time - self.quantum_time - self.queue_time)
                
        def to_dict(self):
            return {
                "quantum_time": round(self.quantum_time, 3),
                "queue_time": round(self.queue_time, 3),
                "classical_time": round(self.classical_time, 3),
                "total_time": round(self.total_time, 3)
            }

    class TimingSamplerWrapper:
        def __init__(self, base_sampler, backend):
            self.base_sampler = base_sampler
            self.backend = backend
            self.options = getattr(base_sampler, 'options', None)
        
        def run(self, *args, **kwargs):
            self.last_tracker = LocalQuantumTimeTracker()
            self.last_tracker.submit_time = time.perf_counter()
            self.last_tracker.iteration_start = self.last_tracker.submit_time
            
            try:
                job = self.base_sampler.run(*args, **kwargs)
            except AttributeError:
                job = self.base_sampler(*args, **kwargs)
            
            self.last_tracker.result_received = time.perf_counter()
            self.last_tracker.iteration_end = self.last_tracker.result_received
            self.last_tracker.extract_from_job(job, getattr(self.backend, 'name', 'local') if self.backend else 'local')
            
            circuit_info['_last_sampler_timing'] = self.last_tracker.to_dict()
            return job
            
    original_run = sampler.run
    def wrapped_run(*args, **kwargs):
        tracker = LocalQuantumTimeTracker()
        tracker.submit_time = time.perf_counter()
        tracker.iteration_start = tracker.submit_time
        job = original_run(*args, **kwargs)
        tracker.result_received = time.perf_counter()
        tracker.iteration_end = tracker.result_received
        tracker.extract_from_job(job, getattr(backend, 'name', 'local') if backend else 'local')
        circuit_info['_last_sampler_timing'] = tracker.to_dict()
        return job
    
    if hasattr(sampler, 'run'):
        sampler.run = wrapped_run
        
    original_store = _store_intermediate_result
    def timing_aware_callback(eval_count, parameters, mean, std):
        original_store(eval_count, parameters, mean, std)
        if '_last_sampler_timing' in circuit_info and len(circuit_info['timing']) > 0:
            est_timing = circuit_info['_last_sampler_timing']
            last_timing_entry = circuit_info['timing'][-1]
            iter_time = last_timing_entry['total_time']
            
            q_time = est_timing.get('quantum_time', 0.0)
            queue_time = est_timing.get('queue_time', 0.0)
            c_time = max(0.0, iter_time - q_time - queue_time)
            
            last_timing_entry['quantum_time'] = round(q_time, 3)
            last_timing_entry['queue_time'] = round(queue_time, 3)
            last_timing_entry['classical_time'] = round(c_time, 3)
            
            circuit_info['_last_sampler_timing'] = None
    if backend is not None:
        logger.debug("Using backend: %s", backend.name)
        logger.debug("Transpilation will be handled by the sampler during execution")

    vqe = SamplingVQE(
        sampler=sampler,
        ansatz=ansatz,
        optimizer=optimizer,
        aggregation=0.1,
        callback=timing_aware_callback,
    )

    return vqe, counts, values, circuit_info


def run_vqe_optimization(
    vqe: SamplingVQE,
    hamiltonian: SparsePauliOp,
) -> SamplingMinimumEigensolverResult:
    """Run the VQE optimization process.

    Args:
        vqe (SamplingVQE): The VQE instance.
        hamiltonian (SparsePauliOp): The Hamiltonian to optimize.

    Returns:
        SamplingMinimumEigensolverResult: The raw results from the VQE optimization.

    """
    logger.debug("Starting VQE optimization")

    start_time: float = time.perf_counter()

    raw_results: SamplingMinimumEigensolverResult = vqe.compute_minimum_eigenvalue(
        hamiltonian
    )
    duration: float = time.perf_counter() - start_time
    minutes, seconds = divmod(duration, 60)

    logger.info("VQE optimization completed in %sm %.2fs", int(minutes), seconds)
    return raw_results


def setup_result_analysis(
    raw_results: SamplingMinimumEigensolverResult,
    protein: Protein,
    vqe_iterations: list[int],
    vqe_energies: list[float],
) -> tuple[ResultInterpreter, ResultVisualizer]:
    """Setup the result analysis components.

    Args:
        raw_results (SamplingMinimumEigensolverResult): The raw results from the VQE optimization.
        protein (Protein): The protein instance.
        vqe_iterations (list[int]): The VQE evaluation counts (iterations).
        vqe_energies (list[float]): The VQE energy values.

    Returns:
        tuple[ResultInterpreter, ResultVisualizer]: The result interpreter and visualizer instances.

    """
    # 使用外部配置提供的基础结果目录，不再创建嵌套的时间戳-链序列子目录
    RESULTS_DATA_DIRPATH.mkdir(parents=True, exist_ok=True)
    dirpath: Path = RESULTS_DATA_DIRPATH

    result_interpreter: ResultInterpreter = ResultInterpreter(
        dirpath=dirpath,
        raw_vqe_results=raw_results,
        protein=protein,
        vqe_iterations=vqe_iterations,
        vqe_energies=vqe_energies,
    )

    result_visualizer: ResultVisualizer = ResultVisualizer(
        dirpath=dirpath,
        turn_sequence=result_interpreter.turn_sequence,
        coordinates_3d=result_interpreter.coordinates_3d,
        main_main_contacts_detected=result_interpreter.main_main_contacts_detected,
    )

    return result_interpreter, result_visualizer
