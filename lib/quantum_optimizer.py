#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量子优化器模块

主要类：
- QuantumOptimizer: 量子优化器，提供VQE和Sampler优化方法
- MockQuantumResult: 模拟量子结果类
"""

import copy
import time
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA, SPSA, SLSQP
from qiskit import transpile, QuantumCircuit
import numpy as np
from scipy.optimize import minimize
import os
import json
import csv
import traceback

from lib.energy_calculator import EnergyCalculator
from lib.protein_folding_builder import ProteinFoldingBuilder


class QuantumTimeTracker:
    """量子计算时间追踪器 - 区分量子/经典计算时间"""

    def __init__(self):
        self.submit_time = None
        self.result_received = None
        self.iteration_start = None
        self.iteration_end = None
        self.quantum_time = 0.0
        self.queue_time = 0.0
        self.classical_time = 0.0
        self.total_time = 0.0
        self._backend_name = None

    def extract_from_job(self, job, backend_name):
        """从 job 对象提取时间信息"""
        self._backend_name = backend_name
        try:
            backend_lower = backend_name.lower()
            if backend_lower.startswith("ibm"):
                self._extract_ibm_time(job)
            elif backend_lower.startswith("aws"):
                self._extract_aws_time(job)
            else:
                self._extract_local_time()
        except Exception as e:
            print(f"    ⚠ 提取时间信息失败: {e}")
            self._extract_local_time()

    def _extract_ibm_time(self, job):
        """从 IBM Quantum job 提取时间"""
        try:
            if hasattr(job, "metrics"):
                metrics = job.metrics()
                self.quantum_time = float(
                    metrics.get("usage", {}).get("quantum_seconds", 0.0)
                )
                classical_from_ibm = float(
                    metrics.get("usage", {}).get("classical_seconds", 0.0)
                )
                timestamps = metrics.get("timestamps", {})
                if timestamps:
                    from datetime import datetime

                    created = timestamps.get("created")
                    running = timestamps.get("running")
                    if created and running:
                        t_created = datetime.fromisoformat(
                            created.replace("Z", "+00:00")
                        )
                        t_running = datetime.fromisoformat(
                            running.replace("Z", "+00:00")
                        )
                        self.queue_time = (t_running - t_created).total_seconds()
                if (
                    self.quantum_time == 0.0
                    and self.submit_time
                    and self.result_received
                ):
                    self.quantum_time = self.result_received - self.submit_time
            else:
                self._extract_local_time()
        except Exception:
            self._extract_local_time()

    def _extract_aws_time(self, job):
        """从 AWS Braket job 提取时间"""
        try:
            metadata = None
            if hasattr(job, "metadata"):
                metadata = job.metadata()
            elif hasattr(job, "_job") and hasattr(job._job, "metadata"):
                metadata = job._job.metadata()
            if metadata:
                from datetime import datetime

                created = metadata.get("createdAt")
                started = metadata.get("startedAt")
                ended = metadata.get("endedAt")
                if created:
                    t_created = datetime.fromisoformat(created.replace("Z", "+00:00"))
                if started:
                    t_started = datetime.fromisoformat(started.replace("Z", "+00:00"))
                else:
                    t_started = t_created
                if ended:
                    t_ended = datetime.fromisoformat(ended.replace("Z", "+00:00"))
                else:
                    t_ended = t_started
                if started and ended:
                    self.quantum_time = (t_ended - t_started).total_seconds()
                if created and started:
                    self.queue_time = (t_started - t_created).total_seconds()
            else:
                self._extract_local_time()
        except Exception:
            self._extract_local_time()

    def _extract_local_time(self):
        """本地模拟器：使用实际测量时间"""
        if self.submit_time and self.result_received:
            self.quantum_time = self.result_received - self.submit_time
        self.queue_time = 0.0

    def finalize(self):
        """计算最终时间"""
        if self.iteration_start and self.iteration_end:
            self.total_time = self.iteration_end - self.iteration_start
        if self.total_time > 0:
            self.classical_time = max(
                0.0, self.total_time - self.quantum_time - self.queue_time
            )

    def to_dict(self):
        """输出时间字典"""
        self.finalize()
        return {
            "quantum_time": round(self.quantum_time, 3),
            "queue_time": round(self.queue_time, 3),
            "classical_time": round(self.classical_time, 3),
            "total_time": round(self.total_time, 3),
        }


def generate_timing_summary(all_timings, backend_name):
    """生成时间汇总报告"""
    if not all_timings:
        return None
    
    normalized_timings = []
    for t in all_timings:
        if isinstance(t, dict):
            normalized_timings.append(t)
        elif isinstance(t, (int, float)):
            normalized_timings.append({
                "quantum_time": round(t, 3),
                "queue_time": 0.0,
                "classical_time": 0.0,
                "total_time": round(t, 3)
            })
        else:
            normalized_timings.append({
                "quantum_time": 0.0,
                "queue_time": 0.0,
                "classical_time": 0.0,
                "total_time": 0.0
            })
    
    all_timings = normalized_timings
    total_quantum = sum(t.get("quantum_time", 0) for t in all_timings)
    total_queue = sum(t.get("queue_time", 0) for t in all_timings)
    total_classical = sum(t.get("classical_time", 0) for t in all_timings)
    total_time = sum(t.get("total_time", 0) for t in all_timings)
    num_iters = len(all_timings)
    return {
        "backend": backend_name,
        "total_quantum_time": round(total_quantum, 3),
        "total_queue_time": round(total_queue, 3),
        "total_classical_time": round(total_classical, 3),
        "total_time": round(total_time, 3),
        "num_iterations": num_iters,
        "average_per_iteration": {
            "quantum_time": round(total_quantum / max(1, num_iters), 3),
            "queue_time": round(total_queue / max(1, num_iters), 3),
            "classical_time": round(total_classical / max(1, num_iters), 3),
            "total_time": round(total_time / max(1, num_iters), 3),
        },
        "per_iteration": all_timings,
    }


class MockQuantumResult:
    """模拟量子结果类"""
    
    def __init__(self, eigenstate, eigenvalue, counts, total_shots):
        self.eigenstate = eigenstate
        self.eigenvalue = eigenvalue
        self.counts = counts
        self.total_shots = total_shots
    
    def get_eigenstate(self):
        """获取本征态"""
        return self.eigenstate
    
    def get_eigenvalue(self):
        """获取本征值（能量）"""
        return self.eigenvalue
    
    def get_counts(self):
        """获取测量结果计数"""
        return self.counts
    
    def get_total_shots(self):
        """获取总采样次数"""
        return self.total_shots
    
    def get_most_probable_state(self):
        """获取最可能的状态"""
        if not self.counts:
            return self.eigenstate
        
        max_count = 0
        most_probable = self.eigenstate
        
        for state, count in self.counts.items():
            if count > max_count:
                max_count = count
                most_probable = state
        
        return most_probable
    
    def get_probability(self, state):
        """获取特定状态的概率"""
        if not self.counts or self.total_shots == 0:
            return 0.0
        
        return self.counts.get(state, 0) / self.total_shots
    
    def __str__(self):
        return f"MockQuantumResult(eigenstate={self.eigenstate}, eigenvalue={self.eigenvalue:.4f})"
    
    def __repr__(self):
        return (f"MockQuantumResult(eigenstate='{self.eigenstate}', "
                f"eigenvalue={self.eigenvalue:.4f}, "
                f"total_shots={self.total_shots})")


class JobRecorder:
    """AWS 任务记录器，用于持久化 Job ID 以防丢失"""
    
    @staticmethod
    def get_job_file(result_dir, label, call_count):
        """生成标准化的任务文件名，用于匹配恢复"""
        if not result_dir: return None
        # 我们使用 label 和 call_count 进行前缀匹配
        return f"{label}_step_{call_count}"

    @staticmethod
    def record_job(job, result_dir, label="job"):
        """记录任务 ID 到文件"""
        if not result_dir:
            return
            
        jobs_dir = os.path.join(result_dir, "jobs")
        os.makedirs(jobs_dir, exist_ok=True)
        
        try:
            job_id = "local_simulation"
            if hasattr(job, 'job_id'):
                job_id = job.job_id()
            
            job_info = {
                "job_id": job_id,
                "timestamp": str(np.datetime64('now')),
                "status": "submitted",
                "label": label
            }
            
            # 使用时间戳和标签生成唯一文件名
            import datetime
            ts = datetime.datetime.now().strftime("%H%M%S_%f")
            job_file = os.path.join(jobs_dir, f"{label}_{ts}.json")
            
            with open(job_file, 'w') as f:
                json.dump(job_info, f, indent=2)
            
            print(f"    [AWS 保护] 任务已记录: {job_id} -> {os.path.basename(job_file)}")
        except Exception as e:
            print(f"    ⚠ 记录任务失败: {e}")

class JobResolver:
    """任务解析器，用于从历史目录恢复 Job"""
    
    @staticmethod
    def resolve_job_id(resume_dir, label_prefix):
        """从恢复目录中寻找匹配的 Job ID"""
        if not resume_dir: return None
        jobs_dir = os.path.join(resume_dir, "jobs")
        if not os.path.exists(jobs_dir): return None
        
        # 查找所有 json 文件，寻找 label 匹配且时间戳最新的
        candidates = []
        for f in os.listdir(jobs_dir):
            if f.startswith(label_prefix) and f.endswith(".json"):
                candidates.append(f)
        
        if not candidates: return None
        
        # 排序取最新（文件名含时间戳）
        latest_file = sorted(candidates)[-1]
        try:
            with open(os.path.join(jobs_dir, latest_file), 'r') as f:
                data = json.load(f)
                return data.get("job_id")
        except:
            return None

class TimingJobWrapper:
    """Job 包装类，用于追踪量子任务时间"""

    def __init__(self, base_job, tracker, backend_name, submit_time):
        self._base_job = base_job
        self._tracker = tracker
        self._backend_name = backend_name
        self._submit_time = submit_time
        self._result_received_time = None

    def result(self, *args, **kwargs):
        result = self._base_job.result(*args, **kwargs)
        self._result_received_time = time.perf_counter()

        timing = QuantumTimeTracker()
        timing.submit_time = self._submit_time
        timing.result_received = self._result_received_time
        timing.iteration_start = self._submit_time
        timing.iteration_end = self._result_received_time
        timing.extract_from_job(self._base_job, self._backend_name)

        self._tracker.last_timing = timing.to_dict()

        print(
            f"    ⏱ 时间分解: 量子={self._tracker.last_timing['quantum_time']:.3f}s, "
            f"队列={self._tracker.last_timing['queue_time']:.3f}s, "
            f"经典={self._tracker.last_timing['classical_time']:.3f}s, "
            f"总计={self._tracker.last_timing['total_time']:.3f}s"
        )

        return result

    def __getattr__(self, name):
        return getattr(self._base_job, name)


class AdaptiveEstimatorV2:
    """EstimatorV2 包装器，支持自适应精度和任务记录"""
    
    def __init__(self, base_estimator, args, result_dir=None):
        self.base_estimator = base_estimator
        self.args = args
        self.result_dir = result_dir
        self.call_count = 0
        self.max_iter = getattr(args, 'max_optimization_iterations', 100)
        self.last_timing = None
        self._backend_name = getattr(args, 'backend', 'local')
    
    def run(self, pubs, precision=None):
        self.call_count += 1
        
        label = f"estimator_step_{self.call_count}"
        
        # 断点恢复逻辑 (Resume)
        if hasattr(self.args, 'resume') and self.args.resume:
            job_id = JobResolver.resolve_job_id(self.args.resume, label)
            if job_id and job_id != "local_simulation":
                print(f"    [Resume] 发现历史任务 ID: {job_id}, 正在尝试拉取结果...")
                try:
                    # 只有真实真机 backend 才有 retrieve_job
                    backend = getattr(self.base_estimator, 'backend', None)
                    if not backend and hasattr(self.base_estimator, 'base_estimator'):
                         # 穿透包装器找原始 backend
                         curr = self.base_estimator
                         while hasattr(curr, 'base_estimator'): curr = curr.base_estimator
                         backend = getattr(curr, 'backend', None)

                    if backend and hasattr(backend, 'retrieve_job'):
                        job = backend.retrieve_job(job_id)
                        print(f"    ✓ 成功恢复任务: {job_id}")
                        return job
                except Exception as e:
                    print(f"    ⚠ 恢复任务失败，回退到重新运行: {e}")

        # 干跑模式 (Dry-Run)
        if hasattr(self.args, 'dry_run') and self.args.dry_run:
            print(f"    [Dry-Run] 拦截 Estimator 调用 (评估点数量: {len(pubs)})")
            # 模拟一个极快的结果或抛出异常进行预检
            pass

        # 如果启用了自适应精度
        if hasattr(self.args, 'adaptive_shots') and self.args.adaptive_shots:
            min_shots = getattr(self.args, 'min_shots', 100)
            max_shots = getattr(self.args, 'max_shots', 1000)
            
            # 使用简单的进度比例计算当前 shots
            ratio = min(self.call_count / (self.max_iter * 2), 1.0) # 估算因子
            current_shots = int(min_shots + (max_shots - min_shots) * ratio)
            
            # 将 shots 转换为 precision: precision = 1 / sqrt(shots)
            adaptive_precision = 1 / (current_shots**0.5)
            
            # 如果外部明确传入了更小的 precision，则保留小的那个以保证质量
            if precision is not None:
                precision = min(precision, adaptive_precision)
            else:
                precision = adaptive_precision
        
        submit_time = time.perf_counter()
        job = self.base_estimator.run(pubs, precision=precision)
        
        # 记录 Job ID
        JobRecorder.record_job(job, self.result_dir, label=f"estimator_step_{self.call_count}")
        
        # 包装 Job 以追踪时间
        wrapped_job = TimingJobWrapper(job, self, self._backend_name, submit_time)
        
        # 对于本地模拟器，job 很快就完成了，我们可以在这里直接设置初始时间信息
        # 这样即使 result() 还没被调用，至少也有个大概的时间
        result_received_time = time.perf_counter()
        timing = QuantumTimeTracker()
        timing.submit_time = submit_time
        timing.result_received = result_received_time
        timing.iteration_start = submit_time
        timing.iteration_end = result_received_time
        timing.extract_from_job(job, self._backend_name)
        self.last_timing = timing.to_dict()
        
        return wrapped_job


class StructuralLoggingEstimator:
    """Estimator 的包装器，拦截 run 调用以获取完整的参数向量并执行结构解析"""
    def __init__(self, base_estimator, structural_callback, result_dir=None):
        self.base_estimator = base_estimator
        self.structural_callback = structural_callback
        self.result_dir = result_dir
        self.call_count = 0
    
    def run(self, pubs, precision=None):
        self.call_count += 1
        # pubs 是一个列表，每个元素通常是 (ansatz, operator, parameters)
        for pub in pubs:
            ansatz, operator, parameters = pub
            # parameters 可能包含多个点 (评估批次)
            if parameters.ndim == 1:
                # 单个评估点
                self.structural_callback(parameters)
            else:
                # 批量评估点
                for p in parameters:
                    self.structural_callback(p)
        
        job = self.base_estimator.run(pubs, precision=precision)
        # 记录 Job ID (如果 base_estimator 不是 AdaptiveEstimatorV2，由这里补齐记录)
        # 如果是嵌套包装，JobRecorder 会生成多个记录文件，这在真机调试时是有好处的
        if not isinstance(self.base_estimator, AdaptiveEstimatorV2):
             JobRecorder.record_job(job, self.result_dir, label=f"struct_step_{self.call_count}")
             
        return job


class QuantumOptimizer:
    """量子优化器基类，提供统一的优化接口，支持VQE和Sampler两种模式"""
    
    @staticmethod
    def create_vqe_optimizer(qubit_op, ansatz, optimizer, estimator, backend=None, args=None, 
                            optimizer_name='COBYLA', result_dir=None, problem=None, sampler=None, trial_idx=1, iteration_csv_path=None):
        """
        创建VQE优化器
        
        Args:
            qubit_op: 量子比特哈密顿量算子
            ansatz: 变分量子线路
            optimizer: 优化器
            estimator: 量子估算器
            backend: 量子后端（可选）
            args: 命令行参数对象（可选）
            
        Returns:
            tuple: (VQE结果, 收敛数据字典)
        """
        working_ansatz = copy.deepcopy(ansatz)
        working_ansatz.remove_final_measurements()

        if backend is not None and args is not None:
            working_ansatz = transpile(
                working_ansatz, 
                backend=backend,
                initial_layout=list(range(working_ansatz.num_qubits)) 
                    if args.backend not in ['local', 'aws_sv1', 'ibm_simulator'] else None,
                optimization_level=3
            )
            print(f"   VQE电路转译: 逻辑比特数={working_ansatz.num_qubits}, 物理比特数={working_ansatz.width()}")

        convergence = {
            'counts': [], 
            'values': [], 
            'stds': [], 
            'cumulative_shots': [], 
            'iteration_shots': [],
            'single_qubit_gates': [],
            'two_qubit_gates': [],
            'circuit_depth': [],
            'best_params': None,
            'best_energy': float('inf'),
            'all_params': [], # 用于后续采样候选结构
            'iteration_times': []
        }
        actual_shots_list = []
        _last_callback_time = [time.perf_counter()]
        
        # 迭代结果保存目录
        iteration_result_dir = None
        if result_dir:
            iteration_result_dir = os.path.join(result_dir, "iter_all_results")
            os.makedirs(iteration_result_dir, exist_ok=True)
        
        # 预先创建 Builder 实例，避免在循环中重复创建
        if problem:
             # 从 problem 中获取必需的属性
             # 注意：有的版本的 problem 可能没有 main_chain 属性，需要兼容性处理
             main_chain = getattr(args, 'main_chain', 'APRLRFY')
             # 如果 problem 有 build_geometry 方法则直接使用，否则创建专用的 geo_builder
             from lib.protein_geometry import ProteinGeometryBuilder
             geo_builder = ProteinGeometryBuilder(main_chain)
             builder = ProteinFoldingBuilder(main_chain)
             turn2qubit = builder.get_turn2qubit()
        else:
             geo_builder = None
             builder = None
             turn2qubit = None

        # 内部状态跟踪
        state = {'best_energy': float('inf'), 'best_params': None}
        
        def record_shots_from_metadata(metadata):
            """从元数据中提取实际执行的shots数"""
            actual_shots = 0
            if "shots_per_circuit" in metadata:
                actual_shots = sum(metadata["shots_per_circuit"])
            elif "execution" in metadata and "circuits" in metadata["execution"]:
                actual_shots = sum(c["shots"] for c in metadata["execution"]["circuits"])
            elif "shots" in metadata:
                shots_per_circuit = metadata["shots"]
                num_circuits = metadata.get("num_circuits", 1)
                actual_shots = shots_per_circuit * num_circuits
            return actual_shots
        
        def callback(eval_count, parameters, mean, std):
            nonlocal _last_callback_time
            convergence['counts'].append(eval_count)
            convergence['values'].append(mean)
            
            # 处理 std 参数，在 EstimatorV2 中通常是元数据字典
            actual_std = 0.0
            if isinstance(std, float):
                actual_std = std
            elif isinstance(std, dict):
                # 尝试从元数据中提取标准差或方差
                if 'standard_error' in std:
                    actual_std = std['standard_error']
                elif 'variance' in std:
                    actual_std = std['variance']**0.5
            elif isinstance(std, (list, np.ndarray)) and len(std) > 0:
                # 某些情况下可能返回一个数组
                actual_std = float(np.mean(std))

            convergence['stds'].append(actual_std)
            convergence['all_params'].append((parameters, mean))
            
            # 跟踪最佳参数
            if mean < state['best_energy']:
                state['best_energy'] = mean
                state['best_params'] = parameters
                # 使用当前正在优化的 ansatz 的参数 view 确保身份一致
                state['best_params_dict'] = {p: v for p, v in zip(working_ansatz.parameters, parameters)}
                
                convergence['best_energy'] = mean
                convergence['best_params'] = parameters
                convergence['best_params_dict'] = state['best_params_dict']

            # 计算电路门数量和深度
            try:
                # 绑定参数到电路
                bound_circuit = working_ansatz.assign_parameters({p: v for p, v in zip(working_ansatz.parameters, parameters)})
                # 计算门数量
                ops = bound_circuit.count_ops()
                # 单比特门数量
                single_qubit_gates = 0
                # 双比特门数量
                two_qubit_gates = 0
                for op, count in ops.items():
                    # 常见的单比特门
                    if op in ['h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg', 'rx', 'ry', 'rz', 'u1', 'u2', 'u3']:
                        single_qubit_gates += count
                    # 常见的双比特门
                    elif op in ['cx', 'cz', 'swap', 'ch', 'cy', 'cs', 'csdg', 'ct', 'ctdg', 'crx', 'cry', 'crz', 'cu1', 'cu2', 'cu3']:
                        two_qubit_gates += count
                # 计算电路深度
                circuit_depth = bound_circuit.depth()
                
                convergence['single_qubit_gates'].append(single_qubit_gates)
                convergence['two_qubit_gates'].append(two_qubit_gates)
                convergence['circuit_depth'].append(circuit_depth)
            except Exception as e:
                print(f"⚠ 计算电路门数量和深度时出错: {e}")
                # 如果出错，添加默认值
                convergence['single_qubit_gates'].append(0)
                convergence['two_qubit_gates'].append(0)
                convergence['circuit_depth'].append(0)

            # 模拟 shots 逻辑 (EstimatorV2 使用 precision，但为了绘图一致性记录 shots)
            current_step_shots = args.shots if args else 100
            
            # 如果启用了自适应 shots，这里也记录预期的 shots 数
            if args and hasattr(args, 'adaptive_shots') and args.adaptive_shots:
                min_shots = getattr(args, 'min_shots', 100)
                max_shots = getattr(args, 'max_shots', 1000)
                max_iter = args.max_optimization_iterations
                # 简单估算当前进度的 shots
                ratio = min(len(convergence['counts']) / max(1, max_iter), 1.0)
                current_step_shots = int(min_shots + (max_shots - min_shots) * ratio)

            actual_shots_list.append(current_step_shots)
            convergence['iteration_shots'].append(current_step_shots)
            
            cumulative_shots = sum(actual_shots_list)
            convergence['cumulative_shots'].append(cumulative_shots)

            current_time = time.perf_counter()
            if _last_callback_time:
                iter_time = current_time - _last_callback_time[0]
                
                est_timing = None
                curr_est = state.get('estimator_ref', estimator)
                while True:
                    if hasattr(curr_est, 'last_timing') and curr_est.last_timing:
                        est_timing = curr_est.last_timing
                        break
                    if hasattr(curr_est, 'base_estimator'):
                        curr_est = curr_est.base_estimator
                    else:
                        break
                
                if est_timing:
                    q_time = est_timing.get("quantum_time", 0.0)
                    queue_time = est_timing.get("queue_time", 0.0)
                    c_time = max(0.0, iter_time - q_time - queue_time)
                else:
                    q_time = 0.0
                    queue_time = 0.0
                    c_time = iter_time

                timing_info = {
                    "quantum_time": round(q_time, 3),
                    "queue_time": round(queue_time, 3),
                    "classical_time": round(c_time, 3),
                    "total_time": round(iter_time, 3)
                }
                convergence['iteration_times'].append(timing_info)
                
                print(
                    f"    ⏱ 时间分解: 量子={timing_info['quantum_time']:.3f}s, "
                    f"队列={timing_info['queue_time']:.3f}s, "
                    f"经典={timing_info['classical_time']:.3f}s, "
                    f"总计={timing_info['total_time']:.3f}s"
                )
            else:
                timing_info = {
                    "quantum_time": 0.0,
                    "queue_time": 0.0,
                    "classical_time": 0.0,
                    "total_time": 0.0
                }
            _last_callback_time[0] = current_time

            # 写入 CSV 记录
            if iteration_csv_path:
                try:
                    single_q = convergence['single_qubit_gates'][-1]
                    two_q = convergence['two_qubit_gates'][-1]
                    total_gates = single_q + two_q
                    
                    with open(iteration_csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            trial_idx,
                            eval_count,
                            args.backend if hasattr(args, 'backend') else 'unknown',
                            total_gates,
                            single_q,
                            two_q,
                            convergence['circuit_depth'][-1],
                            current_step_shots,
                            float(mean),
                            cumulative_shots,
                            float(mean),
                            timing_info.get("quantum_time", 0.0),
                            timing_info.get("queue_time", 0.0),
                            timing_info.get("classical_time", 0.0),
                            timing_info.get("total_time", 0.0)
                        ])
                except Exception as e:
                    print(f"⚠ 写入 CSV 失败: {e}")

            # 保存每步迭代的结果 (如果有 result_dir 和 problem)
            if iteration_result_dir and problem:
                save_path = os.path.join(iteration_result_dir, f'iteration_{eval_count}_result.json')
                
                # 从 state 中获取由 structural_callback 记录的最新结构信息
                protein_info = state.get('last_protein_info', {})

                # 写入 JSON
                with open(save_path, 'w') as f:
                    json.dump({
                        "iteration": eval_count,
                        "energy": float(mean),
                        "std": float(actual_std),
                        "shots": int(current_step_shots),
                        "single_qubit_gates": int(convergence['single_qubit_gates'][-1]),
                        "two_qubit_gates": int(convergence['two_qubit_gates'][-1]),
                        "circuit_depth": int(convergence['circuit_depth'][-1]),
                        "timing": {
                            "quantum_time": timing_info.get("quantum_time", 0.0),
                            "queue_time": timing_info.get("queue_time", 0.0),
                            "classical_time": timing_info.get("classical_time", 0.0),
                            "total_time": timing_info.get("total_time", 0.0)
                        },
                        "protein_structure": protein_info
                    }, f, indent=2)

        def structural_callback(parameters):
            """当 Estimator 收到完整的参数向量时触发，执行蛋白质结构解析"""
            if not (result_dir and problem and sampler):
                return
            
            protein_info = {}
            try:
                # 绑定参数到电路
                sampling_circuit = working_ansatz.copy()
                if not sampling_circuit.get_instructions('measure'):
                     sampling_circuit.measure_all()
                
                param_dict = {p: v for p, v in zip(sampling_circuit.parameters, parameters)}
                bound_circuit = sampling_circuit.assign_parameters(param_dict)
                
                # 执行采样
                current_step_shots = args.shots if args else 100
                if hasattr(sampler, 'run'): # SamplerV2
                    job = sampler.run([bound_circuit], shots=current_step_shots)
                    sampler_result = job.result()
                    pub_result = sampler_result[0]
                    counts = pub_result.data.meas.get_counts()
                else: # Legacy Sampler
                    job = sampler.run(bound_circuit, shots=current_step_shots)
                    sampler_result = job.result()
                    counts = sampler_result.quasi_dists[0].binary_probabilities()
                    
                # 查找最高概率的 bitstring
                best_bs = max(counts, key=counts.get)
                
                # 使用已经创建好的 geo_builder 和 turn2qubit (来自闭包)
                # 这样比在每次回调中重新导入并创建实例快得多
                atoms = geo_builder.build_3d_structure_from_bitstring(best_bs, turn2qubit)
                xyz = [[atom["name"], atom["coords"][0], atom["coords"][1], atom["coords"][2]] for atom in atoms]
                
                # 获取转向序列
                cfg_bits = best_bs[:turn2qubit.count('q')]
                config = geo_builder._fill_config_bits(cfg_bits, turn2qubit)
                turns = [int(config[k:k+2], 2) for k in range(0, len(config), 2)]
                
                protein_info = {
                    "turn_sequence": turns,
                    "xyz_coordinates": xyz,
                    "best_bitstring": best_bs
                }
            except Exception as e:
                print(f"⚠ Warning: Structural interpretation in callback failed: {e}")
                protein_info = {"error": str(e)}
            
            # 将解析出的结构存入 state 供 callback 写入 JSON 时使用
            state['last_protein_info'] = protein_info

        # 确保使用正确的优化器实例
        # 如果传入的是 COBYLA 实例但 optimizer_name 是其他值，这里其实以外部传入的 optimizer 实例为主
        # 但为了逻辑完整，建议外部调用者在调用此方法前就根据 optimizer_name 创建好 optimizer 实例
        
        # 最终包装 Estimator，加入时间追踪、自适应精度支持、结构分析支持和 Job 记录
        final_estimator = AdaptiveEstimatorV2(estimator, args, result_dir=result_dir)
        
        # 包装结构解析逻辑
        final_estimator = StructuralLoggingEstimator(final_estimator, structural_callback, result_dir=result_dir)
        state['estimator_ref'] = final_estimator

        # 干跑模式 (Dry-Run)
        if args and hasattr(args, 'dry_run') and args.dry_run:
            print("\n    [Dry-Run] 正在执行 Estimator 预检...")
            try:
                # 执行一次小规模评估
                test_params = np.random.uniform(-np.pi, np.pi, working_ansatz.num_parameters)
                job = final_estimator.run([(working_ansatz, qubit_op, test_params)], precision=0.3)
                # record_job 已经在包装器中执行
                job.result()
                print("    ✓ Estimator 预检通过 (电路与后端兼容)")
            except Exception as e:
                print(f"    ❌ Estimator 预检失败: {e}")
                raise e

        # 创建 VQE 实例
        # 支持 Warm-Start: 如果 args 中有 initial_point，优先使用
        initial_point = getattr(args, 'initial_point', None)
        vqe = VQE(estimator=final_estimator, ansatz=working_ansatz, optimizer=optimizer, callback=callback, initial_point=initial_point)
        result = vqe.compute_minimum_eigenvalue(qubit_op)
        
        # 使用 result.optimal_point 重建最优参数字典（这是最准确的）
        # 回调函数中的 parameters 可能不完整，所以这里必须重建
        convergence['best_params_dict'] = {p: v for p, v in zip(working_ansatz.parameters, result.optimal_point)}

        # 将使用的 ansatz 存入收敛数据，供后续还原结构使用
        convergence['ansatz'] = working_ansatz

        if hasattr(result, 'metadata') and result.metadata:
            total_shots_from_metadata = record_shots_from_metadata(result.metadata)
            if total_shots_from_metadata > 0:
                if len(convergence['cumulative_shots']) > 0:
                    avg_shots_per_eval = max(1, total_shots_from_metadata // len(convergence['cumulative_shots']))
                    convergence['cumulative_shots'] = [i * avg_shots_per_eval 
                                                       for i in range(1, len(convergence['cumulative_shots']) + 1)]
        
        convergence['label'] = 'VQE Energy'
        
        iteration_results = []
        all_top_energies = []
        iteration_shots_history = convergence.get('iteration_shots', [])
        cumulative_shots_history = convergence.get('cumulative_shots', [])
        all_iteration_timings = convergence.get('iteration_times', [])
        std_history = convergence.get('stds', [0.0]*len(convergence.get('values', [])))
        
        return result, convergence, std_history, iteration_results, all_top_energies, cumulative_shots_history, iteration_shots_history, all_iteration_timings
    
    @staticmethod
    def create_sampler_optimizer(transpiled_circuit, qubit_op, backend, args, result_dir=None, problem=None, sampler_v2=None, trial_idx=1, iteration_csv_path=None):
        """
        创建Sampler优化器
        
        Args:
            transpiled_circuit: 转译后的量子电路
            qubit_op: 量子比特哈密顿量算子
            backend: 量子后端
            args: 命令行参数对象 (需包含 optimizer 属性，默认为 COBYLA)
            result_dir: 结果目录路径
            problem: 蛋白质折叠问题对象
            
        Returns:
            tuple: 优化结果等
        """
        optimizer_name = getattr(args, 'optimizer', 'COBYLA').upper()


        convergence_history = []
        cumulative_shots_history = []
        iteration_shots_history = []
        iteration_results = []
        all_top_energies = []
        all_iteration_timings = []
        _last_objective_time = [time.perf_counter()]

        iteration_result_dir = None
        if result_dir:
            iteration_result_dir = os.path.join(result_dir, "iter_all_results")
            os.makedirs(iteration_result_dir, exist_ok=True)
            
        # 预先提取必需的组件。Sampler 模式也执行此操作。
        from lib.protein_geometry import ProteinGeometryBuilder
        from lib.precise_energy_calculator import PreciseEnergyCalculator
        
        main_chain = args.main_chain
        p_builder = ProteinFoldingBuilder(main_chain)
        interaction_matrix = p_builder.get_interaction_matrix()
        turn2qubit = p_builder.get_turn2qubit()
        geo_builder = ProteinGeometryBuilder(main_chain)
        precise_calc = PreciseEnergyCalculator(main_chain, interaction_matrix)

        # 干跑模式 (Dry-Run)
        if hasattr(args, 'dry_run') and args.dry_run:
            print("\n    [Dry-Run] 正在执行 Sampler 预检...")
            try:
                # 执行一次小规模采样
                dry_shots = 10
                test_params = np.random.uniform(-np.pi, np.pi, transpiled_circuit.num_parameters)
                if sampler_v2:
                    job = sampler_v2.run([(transpiled_circuit, test_params)], shots=dry_shots)
                else:
                    bound_circ = transpiled_circuit.assign_parameters(test_params)
                    job = backend.run(bound_circ, shots=dry_shots)
                
                # 记录 Dry-Run 的 Job
                JobRecorder.record_job(job, result_dir, label="dry_run_sampler")
                res = job.result()
                print("    ✓ Sampler 预检通过 (电路与后端兼容)")
            except Exception as e:
                print(f"    ❌ Sampler 预检失败: {e}")
                raise e

        def objective_function(params):
            """优化目标函数：执行量子电路，使用CVaR策略计算能量"""
            nonlocal _last_objective_time
            
            # 在函数开始时记录迭代开始时间（包含经典处理时间）
            iter_tracker = QuantumTimeTracker()
            iter_tracker.iteration_start = time.perf_counter()
            
            try:
                # 动态计算 shots
                current_shots = args.shots
                if hasattr(args, 'adaptive_shots') and args.adaptive_shots:
                    min_shots = getattr(args, 'min_shots', 100)
                    max_shots = getattr(args, 'max_shots', 2000)
                    max_iter = getattr(args, 'max_optimization_iterations', 100)
                    current_iter = len(convergence_history)
                    
                    # 线性增长策略
                    if max_iter > 1:
                        ratio = min(current_iter / (max_iter - 1), 1.0) # 确保不超过1.0
                        adaptive_shots = int(min_shots + (max_shots - min_shots) * ratio)
                        current_shots = adaptive_shots
                
                label = f"sampler_step_{len(convergence_history)+1}"

                # 断点恢复逻辑 (Resume)
                if hasattr(args, 'resume') and args.resume:
                    job_id = JobResolver.resolve_job_id(args.resume, label)
                    if job_id and job_id != "local_simulation":
                        print(f"    [Resume] 发现历史任务 ID: {job_id}, 正在尝试拉取结果...")
                        try:
                            if backend and hasattr(backend, 'retrieve_job'):
                                job = backend.retrieve_job(job_id)
                                result_recovered = job.result()
                                print(f"    ✓ 成功恢复任务: {job_id}")
                                if sampler_v2:
                                    res_part = result_recovered[0]
                                    data_name = 'meas' if 'meas' in res_part.data else next(iter(res_part.data))
                                    counts = res_part.data[data_name].get_counts()
                                else:
                                    counts = result_recovered.get_counts()
                                # 跳过提交，直接处理恢复的结果
                                actual_shots = sum(counts.values())
                                
                                # 使用已提取的 precise_calc 替代重复创建
                                energy = EnergyCalculator.calculate_cvar_energy_precise(
                                    counts, args.main_chain, interaction_matrix, turn2qubit, args.alpha
                                )
                                energy_std = EnergyCalculator.calculate_energy_std(counts, qubit_op)
                                top_results = EnergyCalculator.extract_top_results_precise(
                                    counts, args.main_chain, interaction_matrix, turn2qubit, args.max_results
                                )
                                top_energies = [res_t[1] for res_t in top_results]
                                convergence_history.append(energy)
                                if not hasattr(objective_function, 'std_history'):
                                    objective_function.std_history = []
                                objective_function.std_history.append(energy_std)
                                iteration_shots_history.append(actual_shots)
                                if cumulative_shots_history:
                                    cumulative_shots_history.append(cumulative_shots_history[-1] + actual_shots)
                                else:
                                    cumulative_shots_history.append(actual_shots)
                                all_top_energies.append(top_energies)
                                if len(convergence_history) % 2 == 0:
                                    print(f"    迭代 {len(convergence_history)}: CVaR能量 = {energy:.4f}, 实际shots = {actual_shots}")
                                return energy
                        except Exception as e:
                            print(f"    ⚠ 恢复任务失败，回退到重新运行: {e}")

                if sampler_v2:
                    # 使用 SamplerV2 (现代化接口)
                    job = sampler_v2.run([(transpiled_circuit, params)], shots=current_shots)
                else:
                    # 使用传统 backend.run 接口
                    bound_circ = transpiled_circuit.assign_parameters(params)
                    job = backend.run(bound_circ, shots=current_shots)
                
                # 记录 Job ID
                JobRecorder.record_job(job, result_dir, label=label)
                
                # 设置提交时间（在量子任务提交后）
                iter_tracker.submit_time = time.perf_counter()
                
                result = job.result()
                
                iter_tracker.result_received = time.perf_counter()
                iter_tracker.iteration_end = iter_tracker.result_received
                iter_tracker.extract_from_job(job, getattr(args, 'backend', 'local'))
                
                timing_info = iter_tracker.to_dict()
                all_iteration_timings.append(timing_info)

                current_time = time.perf_counter()
                if _last_objective_time:
                    iter_time = current_time - _last_objective_time[0]
                    
                    q_time = iter_tracker.quantum_time
                    queue_time = iter_tracker.queue_time
                    c_time = max(0.0, iter_time - q_time - queue_time)
                    
                    timing_info = {
                        "quantum_time": round(q_time, 3),
                        "queue_time": round(queue_time, 3),
                        "classical_time": round(c_time, 3),
                        "total_time": round(iter_time, 3)
                    }
                    print(
                        f"    ⏱ 时间分解: 量子={timing_info['quantum_time']:.3f}s, "
                        f"队列={timing_info['queue_time']:.3f}s, "
                        f"经典={timing_info['classical_time']:.3f}s, "
                        f"总计={timing_info['total_time']:.3f}s"
                    )
                _last_objective_time[0] = current_time
                
                if sampler_v2:
                    result = result[0]
                    # 获取计数 (兼容 standard measure_all 产生的 'meas' 名)
                    data_name = 'meas' if 'meas' in result.data else next(iter(result.data))
                    counts = result.data[data_name].get_counts()
                else:
                    counts = result.get_counts()
                
                actual_shots = sum(counts.values())
                
                # 使用外部 precise_calc 进行优化后的能量计算
                energy = EnergyCalculator.calculate_cvar_energy_precise(
                    counts, args.main_chain, interaction_matrix, turn2qubit, args.alpha
                )
                energy_std = EnergyCalculator.calculate_energy_std(counts, qubit_op)
                
                top_results = EnergyCalculator.extract_top_results_precise(
                    counts, args.main_chain, interaction_matrix, turn2qubit, args.max_results
                )
                top_energies = [res_t[1] for res_t in top_results]
                
                convergence_history.append(energy)
                # 临时存储 std，后续需要添加到 iteration_results 或 convergence_history 中 
                # 但这里的 convergence_history 只是一个 values 列表
                # 我们需要扩展它或创建新的列表
                if not hasattr(objective_function, 'std_history'):
                    objective_function.std_history = []
                objective_function.std_history.append(energy_std)
                
                iteration_shots_history.append(actual_shots)
                
                if cumulative_shots_history:
                    cumulative_shots_history.append(cumulative_shots_history[-1] + actual_shots)
                else:
                    cumulative_shots_history.append(actual_shots)
                
                all_top_energies.append(top_energies)
                
                if len(convergence_history) % 2 == 0:
                    print(f"    迭代 {len(convergence_history)}: CVaR能量 = {energy:.4f}, 实际shots = {actual_shots}")
                    if hasattr(args, 'adaptive_shots') and args.adaptive_shots:
                         print(f"      -> 自适应Shots: {current_shots}")
                    print(f"    各最优结果能量: {[round(e, 4) for e in top_energies]}")
            
                # 计算电路门数量和深度
                try:
                    # 绑定参数到电路
                    bound_circuit = transpiled_circuit.assign_parameters(params)
                    # 计算门数量
                    ops = bound_circuit.count_ops()
                    # 单比特门数量
                    single_qubit_gates = 0
                    # 双比特门数量
                    two_qubit_gates = 0
                    for op, count in ops.items():
                        # 常见的单比特门
                        if op in ['h', 'x', 'y', 'z', 's', 'sdg', 't', 'tdg', 'rx', 'ry', 'rz', 'u1', 'u2', 'u3']:
                            single_qubit_gates += count
                        # 常见的双比特门
                        elif op in ['cx', 'cz', 'swap', 'ch', 'cy', 'cs', 'csdg', 'ct', 'ctdg', 'crx', 'cry', 'crz', 'cu1', 'cu2', 'cu3']:
                            two_qubit_gates += count
                    # 计算电路深度
                    circuit_depth = bound_circuit.depth()
                except Exception as e:
                    print(f"⚠ 计算电路门数量和深度时出错: {e}")
                    # 如果出错，添加默认值
                    single_qubit_gates = 0
                    two_qubit_gates = 0
                    circuit_depth = 0
                
                iteration_data = {
                    "iteration": len(convergence_history),
                    "cvar_energy": energy,
                    "energy_std": energy_std,
                    "top_energies": top_energies,
                    "top_results": [(result[0], float(result[1]), result[2]) for result in top_results],
                    "total_counts": len(counts),
                    "actual_shots": actual_shots,
                    "single_qubit_gates": single_qubit_gates,
                    "two_qubit_gates": two_qubit_gates,
                    "circuit_depth": circuit_depth,
                    "raw_counts": counts,
                    "timing": timing_info if 'timing_info' in dir() else {}
                }
                iteration_results.append(iteration_data)
                
                # 写入 CSV 记录
                if iteration_csv_path:
                    try:
                        single_q = iteration_data.get("single_qubit_gates", 0)
                        two_q = iteration_data.get("two_qubit_gates", 0)
                        total_gates = single_q + two_q
                        with open(iteration_csv_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                trial_idx,
                                iteration_data["iteration"],
                                args.backend if hasattr(args, 'backend') else 'unknown',
                                total_gates,
                                single_q,
                                two_q,
                                iteration_data.get("circuit_depth", 0),
                                iteration_data.get("actual_shots", 0),
                                iteration_data.get("cvar_energy", 0),
                                cumulative_shots_history[-1] if cumulative_shots_history else 0,
                                iteration_data.get("cvar_energy", 0),
                                iteration_data.get("timing", {}).get("quantum_time", 0.0),
                                iteration_data.get("timing", {}).get("queue_time", 0.0),
                                iteration_data.get("timing", {}).get("classical_time", 0.0),
                                iteration_data.get("timing", {}).get("total_time", 0.0)
                            ])
                    except Exception as e:
                        print(f"⚠ 写入 CSV 失败: {e}")
                
                if iteration_result_dir:
                    protein_structure_info = {}
                    top_results_for_struct = iteration_data.get("top_results", [])
                    
                    if top_results_for_struct:
                        best_bitstring, best_energy_val, best_count = top_results_for_struct[0]
                        
                        # 使用已创建的 geo_builder 和 turn2qubit 提取结构
                        atoms = geo_builder.build_3d_structure_from_bitstring(best_bitstring, turn2qubit)
                        
                        # 生成 XYZ 数据
                        xyz_data = [[atom["name"], atom["coords"][0], atom["coords"][1], atom["coords"][2]] 
                                    for atom in atoms]
                        
                        # 生成转向序列
                        cfg_bits = best_bitstring[:turn2qubit.count('q')]
                        config = geo_builder._fill_config_bits(cfg_bits, turn2qubit)
                        turns = [int(config[k:k+2], 2) for k in range(0, len(config), 2)]
                        
                        protein_structure_info = {
                            "turn_sequence": turns,
                            "xyz_coordinates": xyz_data,
                            "best_bitstring": best_bitstring
                        }
                    
                    iteration_result_path = os.path.join(iteration_result_dir, f'iteration_{len(convergence_history)}_result.json')
                    with open(iteration_result_path, 'w') as f:
                        _iter_timing = iteration_data.get("timing", {})
                        serializable_data = {
                            "iteration": iteration_data["iteration"],
                            "cvar_energy": iteration_data["cvar_energy"],
                            "energy_std": iteration_data["energy_std"],
                            "top_energies": [float(e) for e in iteration_data["top_energies"]],
                            "total_counts": iteration_data["total_counts"],
                            "actual_shots": iteration_data["actual_shots"],
                            "single_qubit_gates": int(iteration_data.get("single_qubit_gates", 0)),
                            "two_qubit_gates": int(iteration_data.get("two_qubit_gates", 0)),
                            "circuit_depth": int(iteration_data.get("circuit_depth", 0)),
                            "timing": {
                                "quantum_time": _iter_timing.get("quantum_time", 0.0),
                                "queue_time": _iter_timing.get("queue_time", 0.0),
                                "classical_time": _iter_timing.get("classical_time", 0.0),
                                "total_time": _iter_timing.get("total_time", 0.0)
                            },
                            "protein_structure": protein_structure_info
                        }
                        json.dump(serializable_data, f, indent=2)
                
                return energy
                
            except Exception as e:
                error_msg = str(e)
                print("=" * 80)
                if "DeviceOfflineException" in error_msg or "OFFLINE" in error_msg:
                    print(f"错误: AWS量子设备当前不可用（离线状态）")
                    print(f"详细信息: {e}")
                    print(f"\n完整错误堆栈:")
                    print(traceback.format_exc())
                    print("=" * 80)
                    print(f"建议: 请使用其他可用的AWS设备，如:")
                    print(f"  - aws_sv1 (模拟器，34量子比特)")
                    print(f"  - aws_tn1 (模拟器，50量子比特)")
                    print(f"  - aws_dm1 (模拟器，17量子比特)")
                    print(f"  - aws_forte (IonQ Forte 1，36量子比特)")
                    print(f"  - aws_aria (IonQ Aria 1，25量子比特)")
                    print(f"  - aws_ankaa (Rigetti Ankaa-3，82量子比特)")
                    print(f"  - aws_emerald (IQM Emerald，54量子比特)")
                    print(f"  - aws_ibex (AQT IBEX Q1，1量子比特)")
                    print(f"  - local (本地模拟器)")
                    print(f"\n可以使用以下命令查看所有可用设备:")
                    print(f"  python aws_check.py")
                    print("=" * 80)
                    raise ValueError(f"设备离线，请更换其他设备")
                elif "Unable to translate the operations" in error_msg or ("gpi" in error_msg and "gpi2" in error_msg):
                    print(f"错误: IonQ设备量子门转换失败")
                    print(f"详细信息: {e}")
                    print(f"\n完整错误堆栈:")
                    print(traceback.format_exc())
                    print("=" * 80)
                    print(f"原因: IonQ设备使用特殊的量子门集合 (gpi, gpi2, ms)")
                    print(f"      当前代码使用标准量子门 (H, CNOT, RZ, RX, RY等)")
                    print(f"      Qiskit的transpile无法自动转换到IonQ的基集")
                    print("=" * 80)
                    print(f"解决方案:")
                    print(f"  1. 使用AWS Braket SDK直接构建IonQ兼容的电路")
                    print(f"  2. 添加自定义的量子门转换规则")
                    print(f"  3. 使用其他支持标准量子门的AWS设备:")
                    print(f"     - aws_sv1 (模拟器，34量子比特) ✓ 推荐")
                    print(f"     - aws_tn1 (模拟器，50量子比特) ✓ 推荐")
                    print(f"     - aws_dm1 (模拟器，17量子比特) ✓ 推荐")
                    print(f"     - aws_ankaa (Rigetti Ankaa-3，82量子比特)")
                    print(f"     - aws_emerald (IQM Emerald，54量子比特)")
                    print(f"     - aws_ibex (AQT IBEX Q1，1量子比特)")
                    print(f"     - local (本地模拟器) ✓ 免费，无网络延迟")
                    print("=" * 80)
                    raise ValueError(f"IonQ设备需要特殊的电路转换，请使用其他设备")
                else:
                    print(f"错误: 量子计算执行失败")
                    print(f"原因: {e}")
                    print(f"\n完整错误堆栈:")
                    print(traceback.format_exc())
                    print("=" * 80)
                    raise
            
        print(f"\n正在开始优化 (使用 {optimizer_name})...")
        num_parameters = transpiled_circuit.num_parameters
        
        # 支持 Warm-Start: 如果 args 中有 initial_point，优先使用
        initial_params = getattr(args, 'initial_point', None)
        if initial_params is None:
            initial_params = np.random.uniform(-np.pi, np.pi, num_parameters)
            print(f"   使用随机初始化参数 (维度: {num_parameters})")
        else:
            print(f"   使用 Warm-Start 初始参数 (维度: {num_parameters})")
        
        try:
            if optimizer_name == 'SPSA':
                # SPSA 使用 Qiskit 优化器接口
                # SPSA 需要 bounds，或者我们信任它自动处理
                optimizer = SPSA(maxiter=args.max_optimization_iterations)
                # Qiskit optimizer.minimize 返回的是 OptimizerResult 对象
                result = optimizer.minimize(objective_function, initial_params)
                
                # 构造类似 scipy 的 MinimizeResult
                class ScipyResult:
                    def __init__(self, x, fun):
                        self.x = x
                        self.fun = fun
                res = ScipyResult(result.x, result.fun)
                
            elif optimizer_name == 'SLSQP':
                # 使用 SciPy 的 SLSQP
                res = minimize(objective_function, initial_params, method='SLSQP', 
                            options={'maxiter': args.max_optimization_iterations})
            else:
                # 默认 COBYLA
                res = minimize(objective_function, initial_params, method='COBYLA', 
                            options={'maxiter': args.max_optimization_iterations})
                            
        except Exception as e:
            print(f"⚠ 优化过程发生严重错误: {e}")
            print(traceback.format_exc())
            # 尝试返回已有的最佳结果
            if convergence_history:
                best_idx = np.argmin(convergence_history)
                # 构造一个临时的结果对象
                class EmergencyResult:
                    def __init__(self, x, fun):
                        self.x = initial_params # 这里可能不准确，但也无法获取历史参数
                        self.fun = convergence_history[best_idx]
                res = EmergencyResult(initial_params, convergence_history[best_idx])
            else:
                 raise e 
        
        # 从 objective_function 中提取 std_history
        std_history = getattr(objective_function, 'std_history', [0.0]*len(convergence_history))
        return res, convergence_history, std_history, iteration_results, all_top_energies, cumulative_shots_history, iteration_shots_history, all_iteration_timings
        

