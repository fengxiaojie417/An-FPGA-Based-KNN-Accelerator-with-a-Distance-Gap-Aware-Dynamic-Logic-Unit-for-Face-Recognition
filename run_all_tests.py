"""
主控脚本 - 自适应K值L1-KNN人脸识别系统
一键运行
"""
import os
import sys
import subprocess
import time


def print_banner(text):
    """打印标题"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def check_dependencies():
    """检查必要的Python库"""
    print_banner("步骤 0: 检查依赖环境")

    required_packages = {
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib'
    }

    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {package:20s} 已安装")
        except ImportError:
            print(f"✗ {package:20s} 未安装")
            missing.append(package)

    if missing:
        print(f"\n请安装缺失的包:")
        print(f"pip install {' '.join(missing)}")
        return False

    print("\n✓ 所有依赖已满足!")
    return True


def download_dataset_check():
    """检查是否需要下载数据集"""
    print_banner("步骤 0.5: 检查人脸数据集")

    try:
        from sklearn.datasets import fetch_olivetti_faces
        print("正在检查 Olivetti Faces 数据集...")
        faces = fetch_olivetti_faces(download_if_missing=True)
        print(f"✓ 数据集已就绪: {faces.data.shape[0]} 张人脸")
        return True
    except Exception as e:
        print(f"✗ 数据集加载失败: {str(e)}")
        return False


def run_step(step_num, description, script_name):
    """运行单个测试步骤"""
    print_banner(f"步骤 {step_num}: {description}")

    if not os.path.exists(script_name):
        print(f"✗ 脚本文件不存在: {script_name}")
        return False

    try:
        print(f"正在运行: {script_name}")
        print("-" * 80)

        result = subprocess.run([sys.executable, script_name],
                                capture_output=False,
                                text=True)

        if result.returncode == 0:
            print(f"\n✓ {description} 完成!")
            return True
        else:
            print(f"\n✗ {description} 失败 (返回码: {result.returncode})")
            return False

    except Exception as e:
        print(f"\n✗ 运行出错: {str(e)}")
        return False


def generate_final_report():
    """生成最终的总结报告"""
    print_banner("生成最终报告")

    report = []
    report.append("=" * 80)
    report.append("自适应K值L1-KNN人脸识别系统")
    report.append("=" * 80)
    report.append("")

    # 检查生成的文件
    expected_files = {
        '算法验证': [
            'adaptive_face_knn_analysis.png',
            'face_hardware_dataset.txt'
        ],
        '软件基准': [
            'face_software_baseline_comparison.png'
        ],
        '硬件文件': [
            'face_train_data_bram.mem',
            'face_train_data.coe',
            'face_test_vectors.mem',
            'face_knn_data.h',
            'face_knn_params.vh'
        ]
    }

    report.append("【生成的文件清单】")
    report.append("")

    all_present = True
    for category, files in expected_files.items():
        report.append(f"{category}:")
        for file in files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                if size > 1024 * 1024:
                    size_str = f"{size/(1024*1024):.2f} MB"
                elif size > 1024:
                    size_str = f"{size/1024:.2f} KB"
                else:
                    size_str = f"{size} bytes"
                report.append(f"  ✓ {file:40s} ({size_str})")
            else:
                report.append(f"  ✗ {file:40s} (未找到)")
                all_present = False
        report.append("")

    report.append("=" * 80)
    report.append("【使用指南】")
    report.append("")
    report.append("1. 查看算法性能:")
    report.append("   - adaptive_face_knn_analysis.png       → 算法分析图表")
    report.append("   - face_software_baseline_comparison.png → 软件基准对比")
    report.append("")
    report.append("2. Verilog硬件开发:")
    report.append("   ├─ face_train_data_bram.mem   → 仿真时BRAM初始化")
    report.append("   ├─ face_train_data.coe        → Vivado综合BRAM初始化")
    report.append("   ├─ face_test_vectors.mem      → Testbench测试向量")
    report.append("   └─ face_knn_params.vh         → Verilog参数定义")
    report.append("")
    report.append("3. ARM端软件开发:")
    report.append("   - face_knn_data.h              → C程序头文件")
    report.append("")
    report.append("=" * 80)

    # 打印报告
    report_text = "\n".join(report)
    print(report_text)

    # 保存报告
    with open('final_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n最终报告已保存到: final_report.txt")

    return all_present


def main():
    """主函数：按顺序执行所有步骤"""
    print_banner("自适应K值 L1-KNN 人脸识别加速器 - 验证系统")

    start_time = time.time()
    success_steps = 0
    total_steps = 3

    # 步骤0: 检查依赖
    if not check_dependencies():
        print("\n程序终止: 请先安装缺失的依赖包")
        return

    # 步骤0.5: 检查数据集
    if not download_dataset_check():
        print("\n警告: 数据集加载失败，但可以尝试继续")

    input("\n按回车键开始测试...")

    # 步骤1: 算法验证
    if run_step(1, "自适应K值算法验证 (人脸识别)",
                "knn_algorithm_validator.py"):
        success_steps += 1
    else:
        print("\n警告: 算法验证未完全成功，但可以继续")

    time.sleep(1)

    # 步骤2: 软件基准测试
    if run_step(2, "软件基准性能测试 (Python vs NumPy vs FPGA估算)",
                "software_baseline_benchmark.py"):
        success_steps += 1
    else:
        print("\n警告: 基准测试未完全成功，但可以继续")

    time.sleep(1)

    # 步骤3: 硬件测试文件生成
    if run_step(3, "硬件测试文件生成 (BRAM/COE/C/Verilog)",
                "hardware_testbench_generator.py"):
        success_steps += 1
    else:
        print("\n警告: 硬件文件生成未完全成功")

    time.sleep(1)

    # 生成最终报告
    all_files_present = generate_final_report()

    # 统计运行时间
    elapsed_time = time.time() - start_time

    print_banner("运行完成")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"成功步骤: {success_steps}/{total_steps}")

    if success_steps == total_steps and all_files_present:
        print("\n✓✓✓ 所有步骤成功完成! ✓✓✓")
        print("✓ 所有必要文件已生成")
        print("\n你现在可以:")
        print("  1. 查看生成的图表 (如: adaptive_face_knn_analysis.png)")
        print("  2. 阅读最终报告: final_report.txt")
    else:
        print(f"\n⚠ 部分步骤未成功 ({success_steps}/{total_steps})")
        print("请检查错误信息并重新运行")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n\n发生错误: {str(e)}")
        import traceback
        traceback.print_exc()