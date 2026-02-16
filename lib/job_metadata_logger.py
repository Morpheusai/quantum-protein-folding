#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
作业元数据记录器模块

功能：
- 记录量子计算作业的元数据到CSV文件
- 使用装饰器模式自动记录函数执行信息
- 支持作业历史查询

主要类：
- JobMetadataLogger: 作业元数据记录器类

主要方法：
- log_job(): 记录作业元数据
- get_job_history(): 获取作业历史记录
- clear_history(): 清除作业历史记录

装饰器使用：
- @metadata_logger: 自动记录被装饰函数的执行信息

CSV格式：
job_id,program_name,backend,result_dir,start_time,end_time,duration_seconds,status,python_version,system_info

字段说明：
- job_id: 任务ID（唯一标识）
- program_name: 程序名称（如 run_sampler.py）
- backend: 量子后端（local/aws/ibm）
- result_dir: 结果目录路径
- start_time: 开始时间（ISO格式）
- end_time: 结束时间（ISO格式）
- duration_seconds: 执行时长（秒）
- status: 状态（success/failed: ...）
- python_version: Python版本
- system_info: 系统信息（JSON格式）

使用示例：
    >>> from lib import JobMetadataLogger
    >>> metadata_logger = JobMetadataLogger("protein_folding_jobs.csv")
    >>> 
    >>> @metadata_logger
    >>> def main():
    >>>     pass
    >>> 
    >>> history = metadata_logger.get_job_history(limit=10)

依赖：
- os, csv, datetime: 文件和日期处理
- platform: 系统信息获取
- functools.wraps: 装饰器支持
"""

import os
import csv
import datetime
import platform
import sys
import math
from functools import wraps
from typing import Optional, Dict, Any


class JobMetadataLogger:
    def __init__(self, csv_filename: str = "protein_folding_jobs_detailed.csv"):
        self.csv_filename = csv_filename
        # 可配置的成本表：可后期修改或通过环境变量覆盖
        # 说明：
        # - AWS 模拟器（SV1/DM1/TN1）按“任务持续时间（分钟）”计费
        # - AWS 真机（如 Garnet/Forte/IonQ 等）按“次/任务/预订小时”计价，不直接与 shots 线性相关
        # 默认不设置具体数值，避免误算；可在部署环境中填写或用环境变量覆盖
        self.COST_CONFIG = {
            "aws": {
                "simulator_per_minute_rate": {
                    "aws_sv1": 0.075,
                    "aws_dm1": 0.075,
                    "aws_tn1": 0.275
                },
                "qpu_per_task_rate": {
                    "aws_garnet": 0.3,
                    "aws_forte": 0.3,
                    "aws_ionq": 0.3,
                    "aws_aria": 0.3,
                    "aws_ankaa": 0.3,
                    "aws_emerald": 0.3,
                    "aws_ibex": 0.3
                },
                "qpu_per_shot_rate": {
                    "aws_garnet": 0.00145,
                    "aws_forte": 0.08,
                    "aws_ionq": 0.03,
                    "aws_aria": 0.03,
                    "aws_ankaa": 0.0009,
                    "aws_emerald": 0.0016,
                    "aws_ibex": 0.0235
                }
            },
            "ibm": {
                "simulator_per_minute_rate": {
                    "ibm_simulator": None
                },
                "qpu_per_shot_rate": {
                    "ibm": None
                }
            },
            "local": {
                "rate": 0.0
            }
        }
        self.fieldnames = [
            'job_id',
            'program_name',
            'backend',
            'iteration_count',
            'shots_requested',
            'shots_actual_total',
            'qubits_used',
            'estimated_cost',
            'environment',
            'shots_consistent',
            'result_dir',
            'start_time',
            'end_time',
            'duration_seconds',
            'status',
            'outcome_summary',
            'python_version',
            'system_info',
            'transpile_metrics',
            'convergence_metrics'
        ]
        self.fieldlabels_cn = {
            'job_id': '任务唯一标识',
            'program_name': '入口脚本名称',
            'backend': '量子后端名称',
            'qubits_used': '实际量子位数',
            'environment': '运行环境标签',
            'shots_requested': '请求的每迭代采样次数',
            'shots_actual_total': '实际累积采样次数',
            'shots_consistent': '采样一致性判断',
            'result_dir': '结果目录路径',
            'start_time': '开始时间(ISO 8601)',
            'end_time': '结束时间(ISO 8601)',
            'duration_seconds': '执行时长(秒)',
            'status': '运行状态',
            'iteration_count': '迭代总次数',
            'outcome_summary': '结果摘要',
            'python_version': 'Python版本',
            'system_info': '系统信息(JSON)',
            'estimated_cost': '预估成本',
            'transpile_metrics': '量子转译指标(JSON)',
            'convergence_metrics': '算法收敛指标(JSON)'
        }
        self.cn_fieldnames = [self.fieldlabels_cn[n] for n in self.fieldnames]
        self._ensure_csv_file_exists()
    
    def _ensure_csv_file_exists(self):
        if not os.path.exists(self.csv_filename):
            with open(self.csv_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.cn_fieldnames)
                writer.writeheader()
        else:
            try:
                # 如果已存在但表头顺序不同，则进行迁移重写为新的中文表头顺序
                with open(self.csv_filename, 'r', newline='', encoding='utf-8') as f_in:
                    reader = csv.reader(f_in)
                    existing_header = next(reader, None)
                    if existing_header is not None and existing_header != self.cn_fieldnames:
                        f_in.seek(0)
                        dict_reader = csv.DictReader(f_in)
                        rows = list(dict_reader)
                        tmp_path = self.csv_filename + ".tmp"
                        with open(tmp_path, 'w', newline='', encoding='utf-8') as f_out:
                            writer = csv.DictWriter(f_out, fieldnames=self.cn_fieldnames)
                            writer.writeheader()
                            for row in rows:
                                out = {}
                                for name in self.fieldnames:
                                    cn = self.fieldlabels_cn[name]
                                    val = row.get(name, '')
                                    if val == '':
                                        val = row.get(cn, '')
                                    out[cn] = val
                                writer.writerow(out)
                        # 原子替换
                        os.replace(tmp_path, self.csv_filename)
            except Exception:
                pass
        schema_path = os.path.join(os.path.dirname(self.csv_filename) or ".", "protein_folding_jobs_detailed_schema.csv")
        if not os.path.exists(schema_path):
            with open(schema_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['field', 'label_cn'])
                for name in self.fieldnames:
                    writer.writerow([name, self.fieldlabels_cn.get(name, '')])
    
    def _get_system_info(self) -> str:
        import json
        system_info = {
            'platform': platform.platform(),
            'platform_release': platform.release(),
            'architecture': platform.architecture()[0],
            'machine': platform.machine(),
            'processor': platform.processor(),
            'working_directory': os.getcwd()
        }
        return json.dumps(system_info)
    
    def _parse_backend_from_result_dir(self, result_dir: str) -> str:
        base = os.path.basename(result_dir) if result_dir else ''
        parts = base.split('_')
        for p in parts[::-1]:
            if p and p.lower() not in ['results', 'qupepfold', 'qthesis', 'stfc', 'sampler', 'estimator']:
                return p
        return ''

    def _collect_metrics_from_result_dir(self, result_dir: str) -> Dict[str, Any]:
        import json
        shots_requested = None
        shots_actual_total = 0
        iteration_count = 0
        outcome_summary = ''
        backend = ''
        environment = ''
        found_iteration = False
        transpile_metrics = None
        convergence_metrics = None
        qubits_used = None
        try:
            metrics_path = os.path.join(result_dir, "metrics.json")
            if result_dir and os.path.exists(metrics_path):
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    m = json.load(f)
                backend = m.get('backend', backend)
                environment = backend
                shots_requested = m.get('shots_requested', shots_requested)
                shots_actual_total = int(m.get('shots_actual_total') or 0)
                iteration_count = int(m.get('iteration_count') or 0)
                outcome_summary = m.get('outcome_summary', outcome_summary)
                transpile_metrics = m.get('transpile_metrics', transpile_metrics)
                convergence_metrics = m.get('convergence_metrics', convergence_metrics)
                qubits_used = m.get('qubits_used', qubits_used)
                return {
                    'backend': backend,
                    'environment': environment,
                    'shots_requested': shots_requested,
                    'shots_actual_total': shots_actual_total,
                    'iteration_count': iteration_count,
                    'outcome_summary': outcome_summary,
                    'transpile_metrics': transpile_metrics,
                    'convergence_metrics': convergence_metrics,
                    'qubits_used': qubits_used
                }
        except Exception:
            pass
        if not result_dir or not os.path.exists(result_dir):
            return {
                'backend': backend,
                'environment': environment,
                'shots_requested': shots_requested,
                'shots_actual_total': shots_actual_total,
                'iteration_count': iteration_count,
                'outcome_summary': outcome_summary,
                'transpile_metrics': transpile_metrics,
                'convergence_metrics': convergence_metrics,
                'qubits_used': qubits_used
            }
        for root, _, files in os.walk(result_dir):
            for name in files:
                path = os.path.join(root, name)
                if name.endswith('.json'):
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if shots_requested is None:
                            if 'shots_requested' in data:
                                shots_requested = int(data.get('shots_requested') or 0)
                            elif 'shots' in data and isinstance(data.get('shots'), int):
                                shots_requested = int(data.get('shots') or 0)
                        if qubits_used is None:
                            if 'qubits_used' in data and isinstance(data.get('qubits_used'), int):
                                qubits_used = int(data.get('qubits_used'))
                            elif 'num_qubits' in data and isinstance(data.get('num_qubits'), int):
                                qubits_used = int(data.get('num_qubits'))
                        # 迭代级实际shots
                        if 'actual_shots' in data and isinstance(data.get('actual_shots'), int):
                            shots_actual_total += int(data.get('actual_shots') or 0)
                            iteration_count += 1
                            found_iteration = True
                        if 'shots' in data and isinstance(data.get('shots'), int) and 'iteration' in data:
                            shots_actual_total += int(data.get('shots') or 0)
                            iteration_count += 1
                            found_iteration = True
                        # 从最终结果摘要中聚合迭代shots（避免遗漏）
                        if 'optimization_convergence' in data:
                            conv = data['optimization_convergence']
                            if isinstance(conv, dict) and 'iteration_shots' in conv and isinstance(conv['iteration_shots'], list):
                                # 若尚未找到迭代文件，则以最终摘要为准
                                if not found_iteration:
                                    try:
                                        shots_actual_total += sum(int(x) for x in conv['iteration_shots'])
                                        iteration_count = max(iteration_count, len(conv['iteration_shots']))
                                        found_iteration = True
                                    except:
                                        pass
                        if not backend and 'backend' in data and isinstance(data.get('backend'), str):
                            backend = data.get('backend')
                        if not outcome_summary:
                            if 'energy' in data:
                                outcome_summary = f"energy={float(data['energy']):.6f}"
                            elif 'cvar_energy' in data:
                                outcome_summary = f"cvar={float(data['cvar_energy']):.6f}"
                        if 'transpile_metrics' in data and transpile_metrics is None:
                            transpile_metrics = data.get('transpile_metrics')
                            try:
                                if isinstance(transpile_metrics, dict) and qubits_used is None:
                                    if 'logical_qubits' in transpile_metrics and isinstance(transpile_metrics['logical_qubits'], int):
                                        qubits_used = int(transpile_metrics['logical_qubits'])
                                    elif 'physical_qubits' in transpile_metrics and isinstance(transpile_metrics['physical_qubits'], int):
                                        qubits_used = int(transpile_metrics['physical_qubits'])
                            except:
                                pass
                        if 'convergence_metrics' in data and convergence_metrics is None:
                            convergence_metrics = data.get('convergence_metrics')
                        if not found_iteration:
                            try:
                                if 'iteration_count' in data and isinstance(data.get('iteration_count'), int):
                                    iteration_count = max(iteration_count, int(data.get('iteration_count')))
                                if shots_actual_total == 0 and 'shots_actual_total' in data and isinstance(data.get('shots_actual_total'), int):
                                    shots_actual_total = int(data.get('shots_actual_total'))
                            except:
                                pass
                    except:
                        pass
                elif name.endswith('.csv'):
                    try:
                        if 'qaoa_res' in name or 'exact_res' in name or 'sa_res' in name:
                            with open(path, 'r', encoding='utf-8') as f:
                                reader = csv.DictReader(f)
                                for row in reader:
                                    if 'shots' in row and row['shots']:
                                        shots_requested = int(float(row['shots']))
                                    if 'iterations' in row and row['iterations']:
                                        iteration_count = int(float(row['iterations']))
                                    if 'optimal_eigenvalue' in row and row['optimal_eigenvalue']:
                                        try:
                                            val = float(row['optimal_eigenvalue'])
                                            outcome_summary = f"energy={val:.6f}"
                                        except:
                                            pass
                                    if 'backend' in row and row['backend']:
                                        backend = row['backend']
                                    break
                    except:
                        pass
        if not backend:
            backend = self._parse_backend_from_result_dir(result_dir)
        environment = backend
        return {
            'backend': backend,
            'environment': environment,
            'shots_requested': shots_requested,
            'shots_actual_total': shots_actual_total,
            'iteration_count': iteration_count,
            'outcome_summary': outcome_summary,
            'transpile_metrics': transpile_metrics,
            'convergence_metrics': convergence_metrics,
            'qubits_used': qubits_used
        }

    def _estimate_cost(self, backend: str, shots_actual_total: int, duration_seconds: Optional[float] = None) -> Optional[float]:
        b = backend.lower() if backend else ''
        if b in ('local', 'local_aer', 'local_qiskit'):
            return 0.0
        if not backend or shots_actual_total <= 0:
            # 对于模拟器，可能基于时长计费
            if duration_seconds and duration_seconds > 0:
                minutes = duration_seconds / 60.0
                # 先尝试环境变量覆盖
                env_key_min = f"QUANTUM_COST_PER_MIN_{backend.upper().replace('-', '_')}"
                val_min = os.environ.get(env_key_min)
                if val_min:
                    try:
                        rate = float(val_min)
                        bill_minutes = max(1, math.ceil(minutes))
                        return bill_minutes * rate
                    except:
                        pass
                # 再尝试成本配置
                if b in ("aws_sv1", "aws_dm1", "aws_tn1"):
                    rate = self.COST_CONFIG["aws"]["simulator_per_minute_rate"].get(b)
                    if isinstance(rate, (int, float)) and rate > 0:
                        bill_minutes = max(1, math.ceil(minutes))
                        return bill_minutes * rate
            return None
        key = backend.upper().replace('-', '_')
        env_key = f"QUANTUM_COST_PER_SHOT_{key}"
        val = os.environ.get(env_key)
        try:
            if val:
                price = float(val)
                return shots_actual_total * price
        except:
            return None
        # 若有QPU按任务定价
        b = backend.lower()
        if b in self.COST_CONFIG["aws"]["qpu_per_task_rate"]:
            env_task_key = f"QUANTUM_COST_PER_TASK_{key}"
            env_shot_key = f"QUANTUM_COST_PER_SHOT_{key}"
            rate_task = None
            rate_shot = None
            try:
                if os.environ.get(env_task_key):
                    rate_task = float(os.environ.get(env_task_key))
                if os.environ.get(env_shot_key):
                    rate_shot = float(os.environ.get(env_shot_key))
            except:
                rate_task = rate_task
                rate_shot = rate_shot
            if rate_task is None:
                rate_task = self.COST_CONFIG["aws"]["qpu_per_task_rate"].get(b)
            if rate_shot is None:
                rate_shot = self.COST_CONFIG["aws"]["qpu_per_shot_rate"].get(b)
            if isinstance(rate_task, (int, float)) and rate_task >= 0:
                if isinstance(rate_shot, (int, float)) and rate_shot >= 0:
                    return rate_task + shots_actual_total * rate_shot
                return rate_task
        return None

    def log_job(self, job_data: Dict[str, Any]):
        with open(self.csv_filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.cn_fieldnames, extrasaction='ignore')
            if f.tell() == 0:
                writer.writeheader()
            out = {}
            for name in self.fieldnames:
                out[self.fieldlabels_cn[name]] = job_data.get(name, '')
            writer.writerow(out)
    
    def _get_latest_result_dir(self) -> str:
        results_dir = "results"
        if not os.path.exists(results_dir):
            return ""
        
        dirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) 
                 if os.path.isdir(os.path.join(results_dir, d))]
        
        if not dirs:
            return ""
        
        latest_dir = max(dirs, key=os.path.getmtime)
        return latest_dir
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.datetime.now()
            start_time_str = start_time.isoformat()
            
            try:
                result = func(*args, **kwargs)
                end_time = datetime.datetime.now()
                end_time_str = end_time.isoformat()
                duration = (end_time - start_time).total_seconds()
                result_dir = self._get_latest_result_dir()
                metrics = self._collect_metrics_from_result_dir(result_dir)
                shots_req = metrics.get('shots_requested')
                shots_act = metrics.get('shots_actual_total', 0)
                iter_count = metrics.get('iteration_count', 0)
                if (shots_act is None or shots_act <= 0) and shots_req is not None:
                    shots_act = (shots_req * iter_count) if iter_count and iter_count > 0 else shots_req
                backend = metrics.get('backend') or kwargs.get('backend', '')
                env_name = metrics.get('environment') or backend
                consistent = None
                if shots_req is not None and iter_count > 0:
                    expected = shots_req * iter_count
                    try:
                        diff = abs(shots_act - expected)
                        consistent = diff / max(1, expected) < 0.1
                    except:
                        consistent = None
                estimated_cost = self._estimate_cost(backend, shots_act, duration_seconds=duration)
                import json as _json
                transpile_metrics = metrics.get('transpile_metrics')
                convergence_metrics = metrics.get('convergence_metrics')
                job_data = {
                    'job_id': f"protein_folding_{start_time.strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}",
                    'program_name': os.path.basename(sys.argv[0]) if sys.argv else func.__name__,
                    'backend': backend,
                    'qubits_used': metrics.get('qubits_used') if metrics.get('qubits_used') is not None else '',
                    'environment': env_name,
                    'shots_requested': shots_req if shots_req is not None else '',
                    'shots_actual_total': shots_act,
                    'shots_consistent': consistent if consistent is not None else '',
                    'result_dir': result_dir,
                    'start_time': start_time_str,
                    'end_time': end_time_str,
                    'duration_seconds': duration,
                    'status': 'success',
                    'iteration_count': iter_count,
                    'outcome_summary': metrics.get('outcome_summary', ''),
                    'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    'system_info': self._get_system_info(),
                    'estimated_cost': estimated_cost if estimated_cost is not None else '',
                    'transpile_metrics': _json.dumps(transpile_metrics) if transpile_metrics is not None else '',
                    'convergence_metrics': _json.dumps(convergence_metrics) if convergence_metrics is not None else ''
                }
                
                self.log_job(job_data)
                return result
                
            except Exception as e:
                end_time = datetime.datetime.now()
                end_time_str = end_time.isoformat()
                duration = (end_time - start_time).total_seconds()
                result_dir = self._get_latest_result_dir()
                metrics = self._collect_metrics_from_result_dir(result_dir)
                backend = metrics.get('backend') or kwargs.get('backend', '')
                import json as _json
                shots_req = metrics.get('shots_requested')
                shots_act = metrics.get('shots_actual_total', 0)
                iter_count = metrics.get('iteration_count', 0)
                if (shots_act is None or shots_act <= 0) and shots_req is not None:
                    shots_act = (shots_req * iter_count) if iter_count and iter_count > 0 else shots_req
                job_data = {
                    'job_id': f"protein_folding_{start_time.strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}",
                    'program_name': os.path.basename(sys.argv[0]) if sys.argv else func.__name__,
                    'backend': backend,
                    'qubits_used': metrics.get('qubits_used') if metrics.get('qubits_used') is not None else '',
                    'environment': metrics.get('environment') or backend,
                    'shots_requested': metrics.get('shots_requested') if metrics.get('shots_requested') is not None else '',
                    'shots_actual_total': shots_act,
                    'shots_consistent': '',
                    'result_dir': result_dir,
                    'start_time': start_time_str,
                    'end_time': end_time_str,
                    'duration_seconds': duration,
                    'status': f"failed: {str(e)}",
                    'iteration_count': metrics.get('iteration_count', 0),
                    'outcome_summary': metrics.get('outcome_summary', ''),
                    'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    'system_info': self._get_system_info(),
                    'estimated_cost': '',
                    'transpile_metrics': _json.dumps(metrics.get('transpile_metrics')) if metrics.get('transpile_metrics') is not None else '',
                    'convergence_metrics': _json.dumps(metrics.get('convergence_metrics')) if metrics.get('convergence_metrics') is not None else ''
                }
                
                self.log_job(job_data)
                raise
        
        return wrapper
    
    def get_job_history(self, limit: Optional[int] = None):
        """
        获取作业历史记录
        
        Args:
            limit: 返回的最大记录数，None表示返回所有记录
            
        Returns:
            包含作业记录的列表
        """
        history = []
        
        if not os.path.exists(self.csv_filename):
            return history
        
        with open(self.csv_filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                history.append(row)
        
        if limit is not None:
            history = history[-limit:]
        
        return history
    
    def clear_history(self):
        """
        清除作业历史记录
        """
        if os.path.exists(self.csv_filename):
            os.remove(self.csv_filename)
            self._ensure_csv_file_exists()
