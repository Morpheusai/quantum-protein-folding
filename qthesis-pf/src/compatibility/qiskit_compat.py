"""
Qiskit 2.x 兼容性模块

这个模块提供了 qiskit-algorithms 0.4.0 与 Qiskit 2.x 之间的兼容性支持。
主要解决 DataBin API 不兼容的问题。
"""

import sys
import importlib
from typing import Any


def patch_diagonal_estimator():
    """修补 diagonal_estimator 模块以兼容 Qiskit 2.x"""

    try:
        # 导入 diagonal_estimator 模块
        diagonal_estimator = importlib.import_module(
            "qiskit_algorithms.minimum_eigensolvers.diagonal_estimator"
        )

        # 存储原始方法
        original_run_pub = diagonal_estimator._DiagonalEstimator._run_pub

        def patched_run_pub(self, pub):
            """修补后的 _run_pub 方法，兼容 Qiskit 2.x"""
            import numpy as np
            from qiskit.quantum_info import SparsePauliOp
            from qiskit.primitives.containers import DataBin

            # 重新实现整个方法，避免使用不兼容的 API
            circuit = pub.circuit
            observables = pub.observables
            parameter_values = pub.parameter_values
            bound_circuits = parameter_values.bind_all(circuit)
            bc_circuits, bc_obs = np.broadcast_arrays(bound_circuits, observables)
            sampler_pubs = []
            evs = np.zeros_like(bc_circuits, dtype=np.float64)
            best_measurements = []

            def _has_measurements(circuit):
                """可靠检测电路是否已有测量指令（兼容 Qiskit 2.x）"""
                if circuit.num_clbits == 0:
                    return False
                ops = circuit.count_ops()
                if "measure" in ops and ops["measure"] > 0:
                    return True
                for instruction in circuit.data:
                    if hasattr(instruction, "operation"):
                        if instruction.operation.name == "measure":
                            return True
                    elif hasattr(instruction, "name") and instruction.name == "measure":
                        return True
                return False

            for index in np.ndindex(*bc_circuits.shape):
                bound_circuit = bc_circuits[index]
                observable = bc_obs[index]
                paulis, coeffs = zip(*observable.items())
                obs = SparsePauliOp(paulis, coeffs)

                diagonal_estimator._check_observable_is_diagonal(obs)

                has_measurements = _has_measurements(bound_circuit)

                if has_measurements:
                    if pub.precision is None:
                        sampler_pubs.append((bound_circuit,))
                    else:
                        sampler_pubs.append(
                            (
                                bound_circuit,
                                None,
                                round((sum(obs.coeffs) / pub.precision) ** 2),
                            )
                        )
                else:
                    if pub.precision is None:
                        sampler_pubs.append((bound_circuit.measure_all(inplace=False),))
                    else:
                        sampler_pubs.append(
                            (
                                bound_circuit.measure_all(inplace=False),
                                None,
                                round((sum(obs.coeffs) / pub.precision) ** 2),
                            )
                        )

            job = self.sampler.run(sampler_pubs)
            sampler_pubs_results = job.result()

            for sampler_pub_result, index in zip(
                sampler_pubs_results, np.ndindex(*bc_circuits.shape)
            ):
                observable = bc_obs[index]
                paulis, coeffs = zip(*observable.items())
                obs = SparsePauliOp(paulis, coeffs)

                # 修复：使用正确的 DataBin 访问方法
                meas_data = None

                # 尝试不同的方法找到测量数据
                if hasattr(sampler_pub_result.data, "meas"):
                    meas_data = sampler_pub_result.data.meas
                elif hasattr(sampler_pub_result.data, "_data"):
                    for key in sampler_pub_result.data._data:
                        value = sampler_pub_result.data._data[key]
                        if hasattr(value, "get_int_counts") and hasattr(
                            value, "num_shots"
                        ):
                            meas_data = value
                            break

                # 如果仍未找到，尝试使用可用数据
                if meas_data is None:
                    available_keys = list(sampler_pub_result.data.keys())
                    if available_keys:
                        # 使用第一个有必需方法的可用数据
                        for key in available_keys:
                            value = sampler_pub_result.data[key]
                            if hasattr(value, "get_int_counts") and hasattr(
                                value, "num_shots"
                            ):
                                meas_data = value
                                break
                        if meas_data is None and available_keys:
                            # 回退：使用第一个可用数据
                            meas_data = sampler_pub_result.data[available_keys[0]]
                    else:
                        raise ValueError("在 DataBin 中未找到测量数据")

                # 确保我们有必需的方法
                if not hasattr(meas_data, "get_int_counts") or not hasattr(
                    meas_data, "num_shots"
                ):
                    raise ValueError("测量数据没有必需的方法")

                sampled = {
                    label: value / meas_data.num_shots
                    for label, value in meas_data.get_int_counts().items()
                }

                # 继续原始逻辑的其余部分
                evaluated = {
                    state: (
                        probability,
                        diagonal_estimator._evaluate_sparsepauli(state, obs),
                    )
                    for state, probability in sampled.items()
                }
                evs[index] = np.real_if_close(self.aggregation(evaluated.values()))
                best_result = min(evaluated.items(), key=lambda x: x[1][1])
                best_measurements.append(
                    {
                        "state": best_result[0],
                        "bitstring": bin(best_result[0])[2:].zfill(
                            pub.circuit.num_qubits
                        ),
                        "value": best_result[1][1],
                        "probability": best_result[1][0],
                    }
                )

            if self.callback is not None:
                self.callback(best_measurements)

            data = DataBin(evs=evs, shape=evs.shape)

            # 使用原始结果类返回结果
            return diagonal_estimator._DiagonalEstimatorResult(
                data,
                metadata={
                    "circuit_metadata": pub.circuit.metadata,
                    "target_precision": pub.precision,
                },
                best_measurements=best_measurements,
            )

        # 应用修补
        diagonal_estimator._DiagonalEstimator._run_pub = patched_run_pub
        return True

    except Exception as e:
        print(f"应用 diagonal_estimator 修补时出错: {e}")
        return False


def patch_transpiling_sampler():
    """修补 transpiling_sampler 模块以保留测量指令"""

    try:
        # 导入 transpiling_sampler 模块
        from backend.transpiling_sampler import TranspilingSampler

        # 存储原始 run 方法
        original_run = TranspilingSampler.run

        def patched_run(self, pubs, *, shots=None):
            """修补后的 run 方法，保留测量指令"""
            from typing import cast
            from qiskit import transpile
            from qiskit.circuit.quantumcircuit import QuantumCircuit
            from qiskit.primitives.containers import SamplerPubLike

            pub_list = list(pubs)
            transpiled_pubs = []

            for pub in pub_list:
                if hasattr(pub, "circuit"):
                    circuit = pub.circuit
                elif isinstance(pub, tuple) and len(pub) > 0:
                    circuit = pub[0]
                else:
                    circuit = cast(QuantumCircuit, pub)

                # 重要：不要移除最终测量指令
                # circuit.remove_final_measurements()  # 注释掉这行

                transpiled_circuit = transpile(
                    circuit,
                    backend=self._backend,
                    optimization_level=3,
                )

                if hasattr(pub, "circuit") or isinstance(pub, tuple):
                    transpiled_pubs.append((transpiled_circuit, *pub[1:]))
                else:
                    transpiled_pubs.append(transpiled_circuit)

            return self._sampler.run(transpiled_pubs, shots=shots)

        # 应用修补
        TranspilingSampler.run = patched_run
        return True

    except Exception as e:
        print(f"应用 transpiling_sampler 修补时出错: {e}")
        return False


def apply_all_patches():
    """应用所有兼容性修补"""
    print("正在应用 Qiskit 2.x 兼容性修补...")

    success_count = 0

    if patch_diagonal_estimator():
        success_count += 1
        print("✓ diagonal_estimator 修补成功")

    if patch_transpiling_sampler():
        success_count += 1
        print("✓ transpiling_sampler 修补成功")

    print(f"兼容性修补完成 ({success_count}/2 个修补成功)")
    return success_count == 2


# 模块导入时自动应用修补
apply_all_patches()
