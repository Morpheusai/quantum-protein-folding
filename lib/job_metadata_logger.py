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
"""

import os
import csv
import datetime
import platform
import sys
import math
import json
from functools import wraps
from typing import Optional, Dict, Any, List


class JobMetadataLogger:
    def __init__(self, csv_filename: str = "protein_folding_jobs_detailed.csv"):
        self.csv_filename = csv_filename
        # 可配置的成本表
        self.COST_CONFIG = {
            "aws": {
                "simulator_per_minute_rate": {"aws_sv1": 0.075, "aws_dm1": 0.075, "aws_tn1": 0.275},
                "qpu_per_task_rate": {k: 0.3 for k in ["aws_garnet", "aws_forte", "aws_ionq", "aws_aria", "aws_ankaa", "aws_emerald", "aws_ibex"]},
                "qpu_per_shot_rate": {
                    "aws_garnet": 0.00145, "aws_forte": 0.08, "aws_ionq": 0.03,
                    "aws_aria": 0.03, "aws_ankaa": 0.0009, "aws_emerald": 0.0016, "aws_ibex": 0.0235
                }
            },
            "ibm": {"simulator_per_minute_rate": {"ibm_simulator": None}, "qpu_per_shot_rate": {"ibm": None}},
            "local": {"rate": 0.0}
        }
        
        self.fields = [
            ('job_id', '任务唯一标识'),
            ('program_name', '入口脚本名称'),
            ('backend', '量子后端名称'),
            ('iteration_count', '迭代总次数'),
            ('total_gates', '总门数'),
            ('single_qubit_gates', '单比特门数'),
            ('two_qubit_gates', '双比特门数'),
            ('circuit_depth', '电路深度'),
            ('shots_requested', '请求的每迭代采样次数'),
            ('shots_actual_total', '实际累积采样次数'),
            ('qubits_used', '实际量子位数'),
            ('qubits_full', '展开的量子位数'),
            ('estimated_cost', '预估成本'),
            ('environment', '运行环境标签'),
            ('shots_consistent', '采样一致性判断'),
            ('result_dir', '结果目录路径'),
            ('start_time', '开始时间(ISO 8601)'),
            ('end_time', '结束时间(ISO 8601)'),
            ('duration_seconds', '执行时长(秒)'),
            ('total_quantum_time', '总量子时间(s)'),
            ('total_queue_time', '总队列时间(s)'),
            ('total_classical_time', '总经典时间(s)'),
            ('status', '运行状态'),
            ('outcome_summary', '结果摘要'),
            ('python_version', 'Python版本'),
            ('system_info', '系统信息(JSON)'),
            ('transpile_metrics', '量子转译指标(JSON)'),
            ('convergence_metrics', '算法收敛指标(JSON)')
        ]
        self.fieldnames = [f[0] for f in self.fields]
        self.fieldlabels_cn = dict(self.fields)
        self.cn_fieldnames = [f[1] for f in self.fields]
        
        self._ensure_csv_file_exists()

    def _ensure_csv_file_exists(self):
        """确保CSV文件存在且表头正确"""
        schema_path = os.path.join(os.path.dirname(self.csv_filename) or ".", "protein_folding_jobs_detailed_schema.csv")
        if os.path.exists(schema_path):
            try: os.remove(schema_path)
            except Exception: pass

        if not os.path.exists(self.csv_filename):
            with open(self.csv_filename, 'w', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=self.cn_fieldnames).writeheader()
            return

        try:
            with open(self.csv_filename, 'r', newline='', encoding='utf-8') as f_in:
                reader = csv.reader(f_in)
                header = next(reader, None)
                if header == self.cn_fieldnames:
                    return
                # 表头不匹配，进行迁移
                f_in.seek(0)
                rows = list(csv.DictReader(f_in))
            
            tmp_path = self.csv_filename + ".tmp"
            with open(tmp_path, 'w', newline='', encoding='utf-8') as f_out:
                writer = csv.DictWriter(f_out, fieldnames=self.cn_fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({cn: (row.get(name) or row.get(cn) or '') for name, cn in self.fields})
            os.replace(tmp_path, self.csv_filename)
        except Exception:
            pass

    def _get_system_info(self) -> str:
        return json.dumps({
            'platform': platform.platform(),
            'platform_release': platform.release(),
            'architecture': platform.architecture()[0],
            'machine': platform.machine(),
            'processor': platform.processor(),
            'working_directory': os.getcwd()
        })

    def _collect_metrics_from_result_dir(self, result_dir: str) -> Dict[str, Any]:
        """从结果目录收集各项指标"""
        metrics = {k: (None if k != 'shots_actual_total' and k != 'iteration_count' else 0) 
                   for k in self.fieldnames if k not in ['job_id', 'program_name', 'start_time', 'end_time', 'duration_seconds', 'status', 'python_version', 'system_info']}
        metrics.update({'backend': '', 'environment': '', 'outcome_summary': ''})

        if not result_dir or not os.path.exists(result_dir):
            return metrics

        def update_metrics(data: dict):
            # 基础字段映射
            mapping = {
                'shots_requested': ['shots_requested', 'shots'],
                'shots_actual_total': ['shots_actual_total'],
                'iteration_count': ['iteration_count', 'iterations'],
                'qubits_used': ['qubits_used', 'num_qubits'],
                'qubits_full': ['qubits_full'],
                'backend': ['backend'],
                'environment': ['environment'],
                'outcome_summary': ['outcome_summary'],
                'transpile_metrics': ['transpile_metrics'],
                'convergence_metrics': ['convergence_metrics']
            }
            for key, alt_keys in mapping.items():
                for alt in alt_keys:
                    if alt in data and data[alt] is not None:
                        is_empty = metrics.get(key) is None or metrics.get(key) == '' or metrics.get(key) == 0
                        if is_empty:
                            metrics[key] = data[alt]
            
            # 时间与门数
            timing_keys = ['total_quantum_time', 'total_queue_time', 'total_classical_time', 'total_gates', 'single_qubit_gates', 'two_qubit_gates', 'circuit_depth']
            for k in timing_keys:
                if k in data and data[k] is not None and not metrics.get(k):
                    metrics[k] = data[k]
                    
            # 能量详情
            if not metrics['outcome_summary']:
                if 'energy' in data: metrics['outcome_summary'] = f"energy={float(data['energy']):.6f}"
                elif 'cvar_energy' in data: metrics['outcome_summary'] = f"cvar={float(data['cvar_energy']):.6f}"

        # 1. 优先尝试标准文件
        for fname in ["metrics.json", "timing_summary.json"]:
            path = os.path.join(result_dir, fname)
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        update_metrics(json.load(f))
                except Exception: pass

        # 2. 只有在关键指标缺失时，才遍历其他文件作为补充（避免双倍计数）
        if metrics['shots_actual_total'] == 0 or metrics['qubits_used'] is None:
            found_iter = False
            for root, _, files in os.walk(result_dir):
                for name in files:
                    if name in ["metrics.json", "timing_summary.json"]: continue
                    path = os.path.join(root, name)
                    try:
                        if name.endswith('.json'):
                            with open(path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                update_metrics(data)
                                # 累加采样数和迭代数
                                shots_val = data.get('actual_shots') or (data.get('shots') if 'iteration' in data else None)
                                if isinstance(shots_val, int):
                                    metrics['shots_actual_total'] += shots_val
                                    if not found_iter: metrics['iteration_count'], found_iter = 0, True
                                    metrics['iteration_count'] += 1
                        elif name.endswith('.csv') and any(k in name for k in ['qaoa_res', 'exact_res', 'sa_res']):
                            with open(path, 'r', encoding='utf-8') as f:
                                row = next(csv.DictReader(f), {})
                                if row.get('shots') and metrics['shots_requested'] is None: metrics['shots_requested'] = int(float(row['shots']))
                                if row.get('iterations') and metrics['iteration_count'] == 0: metrics['iteration_count'] = int(float(row['iterations']))
                                if row.get('backend') and not metrics['backend']: metrics['backend'] = row['backend']
                                if row.get('optimal_eigenvalue') and not metrics['outcome_summary']:
                                    metrics['outcome_summary'] = f"energy={float(row['optimal_eigenvalue']):.6f}"
                    except Exception: pass

        if not metrics['backend']:
            base = os.path.basename(result_dir)
            parts = base.split('_')
            for p in parts[::-1]:
                if p and p.lower() not in ['results', 'qupepfold', 'qthesis', 'stfc', 'sampler', 'estimator']:
                    metrics['backend'] = p; break
        metrics['environment'] = metrics.get('environment') or metrics['backend']
        return metrics

    def _estimate_cost(self, backend: str, shots: int, duration: float) -> Optional[float]:
        b = (backend or '').lower()
        if 'local' in b: return 0.0
        
        # 模拟器按时长计费
        if b in ["aws_sv1", "aws_dm1", "aws_tn1"]:
            rate = os.environ.get(f"QUANTUM_COST_PER_MIN_{b.upper().replace('-','_')}") or self.COST_CONFIG["aws"]["simulator_per_minute_rate"].get(b)
            if rate: return max(1, math.ceil(duration / 60.0)) * float(rate)
            
        # QPU按任务或采样计费
        if shots > 0:
            key = (backend or '').upper().replace('-', '_')
            rate_shot = os.environ.get(f"QUANTUM_COST_PER_SHOT_{key}")
            if rate_shot: return shots * float(rate_shot)
            
            if b in self.COST_CONFIG["aws"]["qpu_per_task_rate"]:
                r_task = os.environ.get(f"QUANTUM_COST_PER_TASK_{key}") or self.COST_CONFIG["aws"]["qpu_per_task_rate"].get(b)
                r_shot = os.environ.get(f"QUANTUM_COST_PER_SHOT_{key}") or self.COST_CONFIG["aws"]["qpu_per_shot_rate"].get(b)
                cost = 0.0
                if r_task: cost += float(r_task)
                if r_shot: cost += shots * float(r_shot)
                return cost if cost > 0 else None
        return None

    def log_job(self, job_data: Dict[str, Any]):
        with open(self.csv_filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.cn_fieldnames, extrasaction='ignore')
            writer.writerow({self.fieldlabels_cn[k]: job_data.get(k, '') for k in self.fieldnames})

    def _get_latest_result_dir(self) -> str:
        if not os.path.exists("results"): return ""
        dirs = [os.path.join("results", d) for d in os.listdir("results") if os.path.isdir(os.path.join("results", d))]
        return max(dirs, key=os.path.getmtime) if dirs else ""

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.datetime.now()
            status, error_msg = "success", ""
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = f"failed: {str(e)}"
                raise
            finally:
                end_time = datetime.datetime.now()
                duration = (end_time - start_time).total_seconds()
                result_dir = self._get_latest_result_dir()
                metrics = self._collect_metrics_from_result_dir(result_dir)
                
                backend = metrics.get('backend') or kwargs.get('backend', '')
                shots_req = metrics.get('shots_requested')
                iter_count = metrics.get('iteration_count', 0)
                shots_act = metrics.get('shots_actual_total') or ((shots_req * iter_count) if shots_req and iter_count else (shots_req or 0))
                
                consistent = None
                if shots_req and iter_count:
                    consistent = abs(shots_act - shots_req * iter_count) / (shots_req * iter_count) < 0.1
                
                job_data = {
                    'job_id': f"protein_folding_{start_time.strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}",
                    'program_name': os.path.basename(sys.argv[0]) if sys.argv else func.__name__,
                    'backend': backend,
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'duration_seconds': duration,
                    'status': status,
                    'shots_actual_total': shots_act,
                    'shots_consistent': consistent if consistent is not None else '',
                    'estimated_cost': self._estimate_cost(backend, shots_act, duration) or '',
                    'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    'system_info': self._get_system_info(),
                    'result_dir': result_dir
                }
                # 填充其他从目录收集到的指标
                for k in self.fieldnames:
                    if k not in job_data:
                        val = metrics.get(k, '')
                        if k in ['transpile_metrics', 'convergence_metrics'] and val:
                            job_data[k] = json.dumps(val)
                        else:
                            job_data[k] = val if val is not None else ''
                
                self.log_job(job_data)
        return wrapper

    def get_job_history(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        if not os.path.exists(self.csv_filename): return []
        with open(self.csv_filename, 'r', encoding='utf-8') as f:
            history = list(csv.DictReader(f))
        return history[-limit:] if limit else history

    def clear_history(self):
        if os.path.exists(self.csv_filename):
            os.remove(self.csv_filename)
        self._ensure_csv_file_exists()
