"""从根目录运行量子蛋白质折叠工作流的主执行脚本。

该脚本允许从量子项目根目录运行 qthesis-pf 蛋白质折叠模拟，
并支持通过命令行参数进行配置。

使用示例:
    python run_qthesis.py
    python run_qthesis.py --main_chain "APRLRFY" --backend local_statevector

特性:
    - 通过命令行参数配置
    - 动态覆盖 constants 模块的配置
    - 输出到根目录的 results 文件夹
    - 无需修改原始 qthesis-pf 代码
"""

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit_algorithms import SamplingMinimumEigensolverResult

from lib.job_metadata_logger import JobMetadataLogger


def setup_paths() -> None:
    """为 qthesis-pf 项目设置 Python 导入路径。

    该函数将 qthesis-pf/src 目录添加到 sys.path，允许
    脚本从 qthesis-pf 包中导入模块。

    Raises:
        FileNotFoundError: 如果找不到 qthesis-pf 源代码目录。
    """
    root_dir = Path(__file__).parent
    qthesis_src = root_dir / "qthesis-pf" / "src"
    
    if not qthesis_src.exists():
        raise FileNotFoundError(
            f"找不到 qthesis-pf 源代码目录: {qthesis_src}"
        )
    
    sys.path.insert(0, str(qthesis_src))


def parse_args() -> argparse.Namespace:
    """解析命令行参数，使用 constants 中的默认值。

    Returns:
        argparse.Namespace: 解析后的命令行参数。

    可用参数:
        --main_chain: 主蛋白链序列 (默认: APRLRFY)
        --side_chain: 侧链蛋白序列 (默认: '_' * len(main_chain))
        --backend: 量子后端类型 (默认: local_statevector)
        --interaction_type: 相互作用模型 MJ 或 HP (默认: MJ)
        --shots: 硬件执行的测量次数 (默认: 100)
        --output_dir: 结果输出目录 (默认: <root>/results)
        --encoding: 构象编码类型 (默认: DENSE)
    """
    parser = argparse.ArgumentParser(description="运行量子蛋白质折叠模拟")
    
    parser.add_argument("--main_chain",type=str,default="APRLRFY",help="主蛋白链序列 (默认: APRLRFY)")
    parser.add_argument("--side_chain",type=str,default=None,help="侧链蛋白序列 (默认: '_' * len(main_chain))")
    parser.add_argument("--backend",type=str,choices=["local", "ibm_quantum", "aws_sv1", "aws_garnet"],default="local",help="量子后端类型 (默认: local)")
    parser.add_argument("--interaction_type",type=str,choices=["MJ", "HP"],default="MJ",help="相互作用模型: MJ (Miyazawa-Jernigan) 或 HP (疏水-极性) (默认: MJ)")
    parser.add_argument("--shots",type=int,default=100,help="硬件执行的测量次数 (默认: 100)")
    parser.add_argument("--output_dir",type=str,default=None,help="结果输出基础目录，实际结果将保存在该目录下的 {时间戳}_qthesis_{backend} 子目录中 (默认: <root>/results)")
    parser.add_argument("--encoding",type=str,choices=["DENSE", "SPARSE"],default="DENSE",help="构象编码类型 (默认: DENSE)")
    
    return parser.parse_args()


def apply_config(args: argparse.Namespace) -> None:
    """将命令行参数的配置应用到 constants 模块。"""
    from enums import BackendType, ConformationEncoding, InteractionType
    
    backend_map = {
        "local": BackendType.LOCAL_STATEVECTOR,
        "ibm_quantum": BackendType.IBM_QUANTUM,
        "aws_sv1": BackendType.AWS_SIM_QUANTUM,
        "aws_garnet": BackendType.AWS_QC_QUANTUM,
    }
    
    encoding_map = {
        "DENSE": ConformationEncoding.DENSE,
        "SPARSE": ConformationEncoding.SPARSE,
    }
    
    interaction_map = {
        "MJ": InteractionType.MJ,
        "HP": InteractionType.HP,
    }
    
    import constants
    
    constants.BACKEND_TYPE = backend_map[args.backend]
    constants.CONFORMATION_ENCODING = encoding_map[args.encoding]
    constants.INTERACTION_TYPE = interaction_map[args.interaction_type]
    constants.IBM_QUANTUM_SHOTS = args.shots
    
    # 设置结果基础目录
    root_dir = Path(__file__).parent
    from datetime import datetime
    
    # 创建时间戳子目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subdir_name = f"{timestamp}_qthesis_{args.backend}"
    
    if args.output_dir:
        # 如果用户指定了output_dir，则在该目录下创建 {时间戳}_qthesis_{backend} 的子目录
        output_path = Path(args.output_dir) / subdir_name
        output_path.mkdir(parents=True, exist_ok=True)
        constants.RESULTS_DATA_DIRPATH = output_path
    else:
        # 默认情况下在项目根目录下的results文件夹中创建 {时间戳}_qthesis_{backend} 的子目录
        output_path = root_dir / "results" / subdir_name
        output_path.mkdir(parents=True, exist_ok=True)
        constants.RESULTS_DATA_DIRPATH = output_path


metadata_logger = JobMetadataLogger("protein_folding_jobs_detailed.csv")

@metadata_logger
def main() -> None:
    """执行完整的量子蛋白质折叠工作流。

    工作流程步骤:
        1. 解析命令行参数
        2. 设置导入路径
        3. 应用配置覆盖
        4. 初始化日志记录器
        5. 设置蛋白质折叠系统
        6. 构建并压缩哈密顿量
        7. 设置并运行 VQE 优化
        8. 分析和可视化结果

    Note:
        必须在导入其他模块之前应用配置，以确保
        覆盖后的值在整个模拟过程中被使用。
    """
    args = parse_args()
    setup_paths()
    apply_config(args)
    
    from constants import EMPTY_SIDECHAIN_PLACEHOLDER
    from logger import get_logger
    from utils.setup_utils import (
        build_and_compress_hamiltonian,
        run_vqe_optimization,
        setup_folding_system,
        setup_result_analysis,
        setup_vqe_optimization,
    )
    
    logger = get_logger()
    
    main_chain: str = args.main_chain
    side_chain: str = args.side_chain if args.side_chain else EMPTY_SIDECHAIN_PLACEHOLDER * len(main_chain)
    
    logger.info("Starting quantum protein folding simulation")
    logger.info("Main chain: %s", main_chain)
    logger.info("Side chain: %s", side_chain)
    logger.info("Backend: %s", args.backend)
    logger.info("Interaction type: %s", args.interaction_type)
    
    protein, interaction, contact_map, distance_map = setup_folding_system(
        main_chain=main_chain, side_chain=side_chain
    )

    original_h, compressed_h = build_and_compress_hamiltonian(
        protein=protein,
        interaction=interaction,
        contact_map=contact_map,
        distance_map=distance_map,
    )

    vqe, counts, values, circuit_info = setup_vqe_optimization(num_qubits=compressed_h.num_qubits, shots=args.shots)

    print("\n" + "=" * 60)
    print("run_qthesis.py - Qubit 和 Shot 信息")
    print("=" * 60)
    print()
    print("【输入参数】")
    print(f"  主链序列: {main_chain}")
    print(f"  侧链序列: {side_chain}")
    print(f"  量子后端: {args.backend}")
    print(f"  相互作用模型: {args.interaction_type}")
    print(f"  构象编码: {args.encoding}")
    print(f"  Shot 数量: {args.shots}")
    print()
    print("【Qubit 数量】")
    print(f"  原始哈密顿量 Qubits: {compressed_h.num_qubits}")
    print(f"  压缩比例: {(compressed_h.num_qubits / (len(main_chain) * 2 + len(side_chain) * 2) * 100):.1f}%")
    print()
    print("【Qubit 计算说明】")
    print(f"  原始哈密顿量 Qubits = 主链长度 * 2 + 侧链长度 * 2")
    print(f"  主链长度: {len(main_chain)}")
    print(f"  侧链长度: {len(side_chain)}")
    print(f"  总残基数: {len(main_chain) + len(side_chain)}")
    print()
    print("【Shot 配置】")
    print(f"  默认 Shot 数量: 100")
    print(f"  当前 Shot 数量: {args.shots}")
    print(f"  配置方式: --shots 参数")
    print(f"  应用位置: constants.IBM_QUANTUM_SHOTS")
    print()
    print("=" * 60)
    print()

    raw_results: SamplingMinimumEigensolverResult = run_vqe_optimization(
        vqe=vqe, hamiltonian=compressed_h
    )

    result_interpreter, result_visualizer = setup_result_analysis(
        raw_results=raw_results,
        protein=protein,
        vqe_iterations=counts,
        vqe_energies=values,
    )
    
    import constants
    iteration_csv_path = Path(constants.RESULTS_DATA_DIRPATH) / "iteration_details.csv"
    
    with open(iteration_csv_path, 'w', encoding='utf-8') as f:
        f.write('trial_idx,iteration,backend,total_gates,single_qubit_gates,two_qubit_gates,circuit_depth,shots,energy,cumulative_shots,cvar_energy\n')
        for i in range(len(counts)):
            trial_idx = 1
            iteration = i + 1
            backend = args.backend
            total_gates = circuit_info['total_gates'][i] if i < len(circuit_info['total_gates']) else 0
            single_qubit_gates = circuit_info['single_qubit_gates'][i] if i < len(circuit_info['single_qubit_gates']) else 0
            two_qubit_gates = circuit_info['two_qubit_gates'][i] if i < len(circuit_info['two_qubit_gates']) else 0
            circuit_depth = circuit_info['circuit_depth'][i] if i < len(circuit_info['circuit_depth']) else 0
            shots = circuit_info['shots'][i] if i < len(circuit_info['shots']) else args.shots
            energy = values[i] if i < len(values) else 0
            cumulative_shots = circuit_info['cumulative_shots'][i] if i < len(circuit_info['cumulative_shots']) else 0
            cvar_energy = energy
            f.write(f'{trial_idx},{iteration},{backend},{total_gates},{single_qubit_gates},{two_qubit_gates},{circuit_depth},{shots},{energy},{cumulative_shots},{cvar_energy}\n')
    
    logger.info(f"Iteration details saved to {iteration_csv_path}")
    try:
        from datetime import datetime
        iter_dir = result_interpreter._dirpath / "iterations"
        iter_dir.mkdir(parents=True, exist_ok=True)
        # 准备最终蛋白质结构信息（用于每次迭代记录中包含空间坐标）
        turns = []
        try:
            turns = [getattr(t, "name", str(t)) for t in result_interpreter.turn_sequence]
        except Exception:
            turns = []
        xyz_data = []
        try:
            for bead in result_interpreter.coordinates_3d:
                symbol = getattr(bead, "symbol", None)
                x = float(getattr(bead, "x", 0.0))
                y = float(getattr(bead, "y", 0.0))
                z = float(getattr(bead, "z", 0.0))
                xyz_data.append([symbol, x, y, z])
        except Exception:
            xyz_data = []
        for idx, (iter_count, energy) in enumerate(zip(counts, values)):
            cumulative_shots = [int(args.shots) * i for i in range(1, idx + 2)]
            iteration_shots = [int(args.shots)] * (idx + 1)
            payload = {
                "index": idx + 1,
                "eval_count": int(iter_count),
                "energy": float(energy),
                "shots": int(args.shots),
                "backend": args.backend,
                "main_chain_sequence": main_chain,
                "shots_requested": int(args.shots),
                "optimization_convergence": {
                    "evaluation_counts": [int(c) for c in counts[:idx+1]],
                    "energy_values": [float(e) for e in values[:idx+1]],
                    "cumulative_shots": cumulative_shots,
                    "iteration_shots": iteration_shots
                },
                "protein_structure": {
                    "turn_sequence": turns,
                    "xyz_coordinates": xyz_data
                },
                "timestamp": datetime.now().isoformat()
            }
            with (iter_dir / f"iteration_{idx + 1}_result.json").open("w", encoding="utf-8") as f:
                import json as _json
                _json.dump(payload, f, indent=2)
    except Exception:
        pass

    result_interpreter.dump_results_to_files()

    result_visualizer.visualize_3d()
    result_visualizer.visualize_2d()
    result_visualizer.generate_3d_gif()
    
    logger.info("Simulation completed successfully")

    try:
        import json
        import constants
        metrics_path = Path(constants.RESULTS_DATA_DIRPATH) / "metrics.json"
        transpile_metrics = {}
        convergence_metrics = {}
        metrics = {
            "backend": args.backend,
            "shots_requested": int(args.shots),
            "shots_actual_total": int(args.shots) * (int(len(counts)) if isinstance(counts, list) else 0),
            "iteration_count": int(len(counts)) if isinstance(counts, list) else 0,
            "outcome_summary": f"min_energy={float(min(values)):.6f}" if isinstance(values, list) and len(values) > 0 else "",
            "qubits_used": int(compressed_h.num_qubits),
            "qubits_full": int(original_h.num_qubits),
            "transpile_metrics": transpile_metrics,
            "convergence_metrics": convergence_metrics
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    except Exception:
        pass

if __name__ == "__main__":
    main()
