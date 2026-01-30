#!/bin/bash

# =================================================================
# 脚本名称: run_screen.sh
# 功能: 在 screen 会话中，于指定的 Conda 环境下异步启动蛋白质折叠算法
# 支持 run_opt.py (Estimator模式) 和 run_opt_sampler.py (Sampler模式)
# =================================================================

# --- 配置区 ---
CONDA_ENV_NAME="duneq"
SESSION_NAME="protein_fold"
# --------------

usage() {
    echo "用法: $0 [模式] [python 参数]"
    echo "模式:"
    echo "  -s, --sampler      使用 run_opt_sampler.py (默认，Sampler模式)"
    echo "  -e, --estimator    使用 run_opt.py (Estimator模式)"
    echo "参数:"
    echo "  [python 参数]      传递给Python脚本的参数"
    echo "示例:"
    echo "  $0 -s --backend aws_garnet --main_chain APRL   # Sampler模式 (默认)"
    echo "  $0 --sampler --backend local --main_chain APRL --shots 500  # 本地后端"
    echo "  $0 -e --backend aws_sv1 --main_chain APRL      # Estimator模式"
    echo "  $0 --estimator --backend ibm --main_chain APL --shots 1000  # IBM后端"
    echo "---------------------------------------------------"
    echo "管理命令:"
    echo "  screen -ls                      # 查看所有后台任务"
    echo "  screen -r $SESSION_NAME          # 进入正在运行的任务界面"
    echo "  tail -f last_run.log            # 查看最新实时日志"
    exit 1
}

# 解析命令行参数
MODE="sampler"  # 默认模式
PARAMS=()         # 存储传递给Python脚本的参数

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--estimator)
            MODE="estimator"
            shift
            ;;
        -s|--sampler)
            MODE="sampler"
            shift
            ;;
        *)
            # 所有非模式参数都添加到PARAMS数组
            PARAMS+=("$1")
            shift
            ;;
    esac
done

# 检查 screen 是否安装
if ! command -v screen &> /dev/null; then
    echo "错误: 系统未安装 screen。"
    exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 根据模式生成不同的日志文件名
if [ "$MODE" = "estimator" ]; then
    LOG_FILE="run_estimator_${TIMESTAMP}.log"
    PYTHON_SCRIPT="run_estimator.py"
    SESSION_NAME="${SESSION_NAME}_estimator"
else
    LOG_FILE="run_sampler_${TIMESTAMP}.log"
    PYTHON_SCRIPT="run_sampler.py"
    SESSION_NAME="${SESSION_NAME}_sampler"
fi

echo "---------------------------------------------------"
echo "🚀 准备在 Conda 环境 [$CONDA_ENV_NAME] 中启动任务"
echo "模式: $MODE"
echo "会话名称: $SESSION_NAME"
echo "日志文件: $LOG_FILE"
echo "Python脚本: $PYTHON_SCRIPT"
echo "---------------------------------------------------"

# 将参数数组转换为字符串
PARAMS_STR="${PARAMS[*]}"

# 在 screen 中执行的复合命令：
# 1. source 系统的 conda 配置文件（确保可以在脚本中使用 conda activate）
# 2. 激活指定的 duneq 环境
# 3. 运行 python 脚本
# 4. 2>&1 | tee 将标准输出和错误同时打印到日志并显示在 screen 屏幕
CMD="source \$(conda info --base)/etc/profile.d/conda.sh && \
     conda activate $CONDA_ENV_NAME && \
     python $PYTHON_SCRIPT $PARAMS_STR 2>&1 | tee $LOG_FILE"

# 启动分离模式的 screen
screen -d -m -S "$SESSION_NAME" bash -c "$CMD"

# 创建日志快捷链接（根据模式不同）
if [ "$MODE" = "estimator" ]; then
    ln -sf "$LOG_FILE" last_estimator_run.log
else
    ln -sf "$LOG_FILE" last_sampler_run.log
fi

# 创建通用的日志链接
ln -sf "$LOG_FILE" last_run.log

echo "✅ 任务已在 Conda 环境下提交。"
echo "您可以运行 'tail -f $LOG_FILE' 或 'tail -f last_run.log' 监控进度。"