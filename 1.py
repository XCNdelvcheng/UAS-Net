import subprocess
import csv
import re
import itertools
import os

# --- 1. 定义超参数网格 ---
# (在这里配置你想要的参数)
param_grid_stage1 = {
    'LR': [2e-4, 4e-4, 5e-4],
    'WEIGHT_DECAY': [2e-5, 3e-5, 5e-4],
    'LAMBDA_CONS': [1.0],
    'DROPOUT_RATE': [0.5]
}

param_grid_stage2 = {
    'LR': [4e-4],
    'WEIGHT_DECAY': [1e-5],
    'LAMBDA_CONS': [0.5, 1.0, 1.5],
    'DROPOUT_RATE': [0.3, 0.5]
}

# --- 选择你要运行的阶段 ---
param_grid = param_grid_stage1
# -------------------------


# --- 2. 配置日志 ---
log_dir = "experiment_logs"
os.makedirs(log_dir, exist_ok=True)

summary_file = "tuning_results.csv"
# [重要] 确保这里是你训练脚本的正确名字
train_script = "y_Uncer.py"  # <-- 确保这个文件名是你的训练脚本 (e.g., train_tunable.py 或 1.py)

# -------------------------


# 生成所有实验组合
keys = param_grid.keys()
values = param_grid.values()
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"--- 准备开始 {len(experiments)} 组实验 ---")
print(f"--- 摘要将保存到: {summary_file} ---")
print(f"--- 完整日志将保存到: {log_dir}/ ---")

# 3. 准备摘要文件 (CSV)
try:
    # [修改] 我们使用 'a' (追加) 模式，这样可以断点续传
    # 首先检查文件是否存在，如果不存在，则写入表头
    file_exists = os.path.isfile(summary_file)

    with open(summary_file, 'a', newline='', encoding='utf-8') as f_summary:
        writer = csv.writer(f_summary)

        header = list(experiments[0].keys()) + ["Test_C_Index", "LogFile"]
        if not file_exists:
            writer.writerow(header)

        # 4. 循环执行实验
        for i, params in enumerate(experiments):
            print("\n" + "=" * 80)
            print(f"--- 实验 {i + 1}/{len(experiments)}: {params} ---")

            # --- A. 创建唯一的日志文件名 ---
            log_filename_parts = []
            for key, value in params.items():
                key_short = key.replace("WEIGHT_DECAY", "WD").replace("LAMBDA_CONS", "L").replace("DROPOUT_RATE", "D")
                log_filename_parts.append(f"{key_short}_{value}")

            log_filename = os.path.join(log_dir, "_".join(log_filename_parts) + ".log")

            # --- B. 构建命令 ---
            # [重要修改]
            # 1. 使用 'python -u' (-u 标志) 来强制 Python 不使用输出缓存
            #    这能确保我们逐行实时看到输出，而不是等它攒一大堆再输出
            # 2. 确保 train_script 变量是你的文件名
            command = ['python', '-u', train_script]
            for key, value in params.items():
                command.append(f'--{key}')
                command.append(str(value))

            test_c_index = "Parse_Error"  # 默认值，如果没找到C-index

            try:
                # --- C. [核心修改] 使用 Popen 实时处理输出 ---
                my_env = os.environ.copy()
                my_env["AUTOMATED_RUN"] = "1"  # "1" 表示 True
                # 1. 打开日志文件准备写入
                with open(log_filename, 'w', encoding='utf-8') as f_log:
                    f_log.write(f"--- 实验参数: {params} ---\n\n")
                    f_log.write("--- STDOUT & STDERR (实时) ---\n")

                    # 2. 启动子进程
                    #    - stdout=subprocess.PIPE: 捕获标准输出
                    #    - stderr=subprocess.STDOUT: 将标准错误合并到标准输出 (这样OOM等错误也会被捕获)
                    #    - bufsize=1: 行缓冲
                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding='utf-8',
                        bufsize=1,
                        env=my_env
                    )

                    # 3. 逐行读取子进程的输出
                    for line in process.stdout:
                        # 3a. (你的需求) 实时打印到控制台
                        print(line, end='')

                        # 3b. 实时写入日志文件
                        f_log.write(line)
                        f_log.flush()  # 确保立即写入

                        # 3c. 实时解析C-index
                        if "FINAL_RESULT::" in line:
                            match = re.search(r"FINAL_RESULT::([\d\.]+)", line)
                            if match:
                                test_c_index = float(match.group(1))

                    # 4. 等待进程结束
                    process.wait()
                    return_code = process.returncode

                    # --- D. 实验结束后的处理 ---
                    if return_code != 0:
                        print(f"\n--- FAILED experiment {i + 1} with exit code {return_code} ---")
                        if test_c_index == "Parse_Error":  # 如果没找到C-index
                            test_c_index = f"Run_Error (Code {return_code})"

                    else:
                        if isinstance(test_c_index, float):
                            print(f"\n--- 实验 {i + 1} 成功: Test C-Index = {test_c_index} ---")
                        elif test_c_index == "Parse_Error":
                            print(f"\n--- FAILED to parse result for experiment {i + 1} (未找到 FINAL_RESULT) ---")

                    print(f"--- 完整日志已保存到: {log_filename} ---")

            except Exception as e_popen:
                # 捕获启动进程时的错误 (例如 'python' 命令未找到)
                print(f"!! 运行子进程时出错: {e_popen}")
                test_c_index = f"Popen_Error: {e_popen}"

            # 5. 记录摘要到 CSV
            row = list(params.values()) + [test_c_index, log_filename]
            writer.writerow(row)
            f_summary.flush()  # 立即写入

except Exception as e_main:
    print(f"自动化脚本发生严重错误: {e_main}")

print(f"\n" + "=" * 80)
print(f"--- 所有实验完成! 摘要已保存到 {summary_file} ---")