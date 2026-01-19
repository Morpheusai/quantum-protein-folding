import os
import sys
import time
import csv
import json
import uuid
import platform
from datetime import datetime
from functools import wraps

class JobMetadataLogger:
    """
    蛋白质折叠量子计算的作业元数据记录器
    以装饰器形式记录运行信息，支持CSV格式持久化存储
    """
    
    def __init__(self, csv_file_path="job_metadata.csv"):
        self.csv_file_path = csv_file_path
        self._ensure_csv_header()
    
    def _ensure_csv_header(self):
        """确保CSV文件存在并包含正确的表头"""
        headers = [
            "job_id", "program_name", "backend", "result_dir",
            "start_time", "end_time", "duration_seconds", "status",
            "python_version", "system_info"
        ]
        
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    def __call__(self, func):
        """装饰器实现"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成唯一作业ID
            job_id = f"protein_folding_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            # 收集元数据 - 这里直接获取，不依赖globals()
            metadata = {
                "job_id": job_id,
                "program_name": os.path.basename(sys.argv[0]),
                "backend": self._get_backend_value(),
                "result_dir": self._get_result_dir(),
                "start_time": datetime.now().isoformat(),
                "status": "running",
                "python_version": sys.version.split()[0],  # 只保留版本号
                "system_info": self._get_system_info()
            }
            
            start_time = time.time()
            result = None
            try:
                # 执行被装饰的函数
                result = func(*args, **kwargs)
                
                # 记录成功状态
                metadata["status"] = "success"
                metadata["end_time"] = datetime.now().isoformat()
                metadata["duration_seconds"] = round(time.time() - start_time, 2)
                
                # 保存到CSV
                self._save_to_csv(metadata)
                
                return result
                
            except Exception as e:
                # 记录失败状态
                metadata["status"] = f"failed: {str(e)}"
                metadata["end_time"] = datetime.now().isoformat()
                metadata["duration_seconds"] = round(time.time() - start_time, 2)
                
                # 保存到CSV
                self._save_to_csv(metadata)
                raise
        
        return wrapper
    
    def _get_backend_value(self):
        """获取后端配置值"""
        try:
            # 通过解析命令行参数获取backend值
            import sys
            backend_value = "local"  # 默认值
            
            # 解析命令行参数
            for i, arg in enumerate(sys.argv):
                if arg == '--backend' and i + 1 < len(sys.argv):
                    backend_value = sys.argv[i + 1]
                    break
                elif arg.startswith('--backend='):
                    backend_value = arg.split('=', 1)[1]
                    break
            
            return backend_value if backend_value else "local"
        except:
            return "local"  # 默认值
    
    def _get_result_dir(self):
        """获取结果目录 - 直接从当前工作目录推断"""
        try:
            # 查找当前目录下的results文件夹
            current_dir = os.getcwd()
            results_dir = os.path.join(current_dir, "results")
            
            if os.path.exists(results_dir):
                # 查找最新的结果目录
                latest_subdir = None
                latest_time = 0
                
                for item in os.listdir(results_dir):
                    item_path = os.path.join(results_dir, item)
                    if os.path.isdir(item_path):
                        # 检查目录名格式：timestamp_backend
                        if '_' in item:
                            try:
                                dir_time = os.path.getmtime(item_path)
                                if dir_time > latest_time:
                                    latest_time = dir_time
                                    latest_subdir = item_path
                            except:
                                continue
                
                if latest_subdir:
                    return latest_subdir
            
            # 如果找不到，返回默认路径
            backend = self._get_backend_value()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_dir = os.path.join("results", f"{timestamp}_{backend}")
            return default_dir
            
        except Exception as e:
            print(f"⚠ 结果目录获取警告: {e}")
            return "-"
    
    def _get_system_info(self):
        """获取系统信息"""
        try:
            system_info = {
                "platform": platform.system(),
                "platform_release": platform.release(),
                "architecture": platform.architecture()[0],
                "machine": platform.machine(),
                "processor": platform.processor(),
                "working_directory": os.getcwd()
            }
            return json.dumps(system_info, ensure_ascii=False)
        except:
            return "{}"
    
    def _save_to_csv(self, metadata):
        """保存元数据到CSV文件"""
        try:
            with open(self.csv_file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    metadata["job_id"],
                    metadata["program_name"],
                    metadata["backend"],
                    metadata["result_dir"],
                    metadata["start_time"],
                    metadata.get("end_time", ""),
                    metadata.get("duration_seconds", 0),
                    metadata["status"],
                    metadata["python_version"],
                    metadata["system_info"]
                ])
            print(f"✓ 作业元数据已记录到: {self.csv_file_path}")
        except Exception as e:
            print(f"⚠ CSV记录失败: {e}")