#!/bin/bash

# =================================================================
# 脚本名称: run_screen.sh
# 功能: 在 screen 会话中，于指定的 Conda 环境下异步启动蛋白质折叠算法
# =================================================================

# --- 配置区 ---
CONDA_ENV_NAME="duneq"
SESSION_NAME="protein_fold"
# --------------

usage() {
    echo "用法: $0 [python 参数]"
    echo "示例: $0 --backend aws_sv1 --main_chain APRL"
    echo "---------------------------------------------------"
    echo "管理命令:"
    echo "  screen -ls                # 查看所有后台任务"
    echo "  screen -r $SESSION_NAME    # 进入正在运行的任务界面"
    echo "  tail -f last_run.log      # 查看最新实时日志"
    exit 1
}

# 检查 screen 是否安装
if ! command -v screen &> /dev/null; then
    echo "错误: 系统未安装 screen。"
    exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="run_${TIMESTAMP}.log"

echo "---------------------------------------------------"
echo "🚀 准备在 Conda 环境 [$CONDA_ENV_NAME] 中启动任务"
echo "会话名称: $SESSION_NAME"
echo "日志文件: $LOG_FILE"
echo "---------------------------------------------------"

# 在 screen 中执行的复合命令：
# 1. source 系统的 conda 配置文件（确保可以在脚本中使用 conda activate）
# 2. 激活指定的 duneq 环境
# 3. 运行 python 脚本
# 4. 2>&1 | tee 将标准输出和错误同时打印到日志并显示在 screen 屏幕
CMD="source \$(conda info --base)/etc/profile.d/conda.sh && \
     conda activate $CONDA_ENV_NAME && \
     python run_opt.py $@ 2>&1 | tee $LOG_FILE"

# 启动分离模式的 screen
screen -d -m -S "$SESSION_NAME" bash -c "$CMD"

# 创建日志快捷链接
ln -sf "$LOG_FILE" last_run.log

echo "✅ 任务已在 Conda 环境下提交。"
echo "您可以运行 'tail -f last_run.log' 监控进度。"