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
from functools import wraps
from typing import Optional, Dict, Any


class JobMetadataLogger:
    """
    作业元数据记录器
    
    用于记录量子计算作业的元数据信息到CSV文件
    可以作为装饰器使用，自动记录函数执行信息
    """
    
    def __init__(self, csv_filename: str = "protein_folding_jobs.csv"):
        """
        初始化作业元数据记录器
        
        Args:
            csv_filename: CSV文件名，默认为"protein_folding_jobs.csv"
        """
        self.csv_filename = csv_filename
        self._ensure_csv_file_exists()
    
    def _ensure_csv_file_exists(self):
        """
        确保CSV文件存在并包含正确的表头
        """
        if not os.path.exists(self.csv_filename):
            with open(self.csv_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'job_id',
                    'program_name',
                    'backend',
                    'result_dir',
                    'start_time',
                    'end_time',
                    'duration_seconds',
                    'status',
                    'python_version',
                    'system_info'
                ])
    
    def _get_system_info(self) -> str:
        """
        获取系统信息
        
        Returns:
            系统信息的JSON字符串
        """
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
    
    def log_job(self, job_data: Dict[str, Any]):
        """
        记录作业元数据
        
        Args:
            job_data: 包含作业信息的字典，支持以下键：
                - job_id: 作业ID
                - program_name: 程序名称
                - backend: 量子后端
                - result_dir: 结果目录
                - start_time: 开始时间
                - end_time: 结束时间
                - duration_seconds: 执行时长（秒）
                - status: 状态
                - python_version: Python版本（可选，默认使用当前版本）
                - system_info: 系统信息（可选，默认自动获取）
        """
        with open(self.csv_filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                job_data.get('job_id', ''),
                job_data.get('program_name', ''),
                job_data.get('backend', ''),
                job_data.get('result_dir', ''),
                job_data.get('start_time', ''),
                job_data.get('end_time', ''),
                job_data.get('duration_seconds', ''),
                job_data.get('status', ''),
                job_data.get('python_version', f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"),
                job_data.get('system_info', self._get_system_info())
            ])
    
    def _get_latest_result_dir(self) -> str:
        """
        获取results目录下最新的目录
        
        Returns:
            最新结果目录的路径，如果不存在则返回空字符串
        """
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
        """
        装饰器方法
        
        使用方式：
            @metadata_logger
            def my_function():
                pass
        
        Args:
            func: 被装饰的函数
            
        Returns:
            包装后的函数
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.datetime.now()
            start_time_str = start_time.isoformat()
            
            try:
                result = func(*args, **kwargs)
                end_time = datetime.datetime.now()
                end_time_str = end_time.isoformat()
                duration = (end_time - start_time).total_seconds()
                
                job_data = {
                    'job_id': f"protein_folding_{start_time.strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}",
                    'program_name': os.path.basename(sys.argv[0]) if sys.argv else func.__name__,
                    'backend': kwargs.get('backend', 'local'),
                    'result_dir': self._get_latest_result_dir(),
                    'start_time': start_time_str,
                    'end_time': end_time_str,
                    'duration_seconds': duration,
                    'status': 'success'
                }
                
                self.log_job(job_data)
                return result
                
            except Exception as e:
                end_time = datetime.datetime.now()
                end_time_str = end_time.isoformat()
                duration = (end_time - start_time).total_seconds()
                
                job_data = {
                    'job_id': f"protein_folding_{start_time.strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}",
                    'program_name': os.path.basename(sys.argv[0]) if sys.argv else func.__name__,
                    'backend': kwargs.get('backend', 'local'),
                    'result_dir': self._get_latest_result_dir(),
                    'start_time': start_time_str,
                    'end_time': end_time_str,
                    'duration_seconds': duration,
                    'status': f"failed: {str(e)}"
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
