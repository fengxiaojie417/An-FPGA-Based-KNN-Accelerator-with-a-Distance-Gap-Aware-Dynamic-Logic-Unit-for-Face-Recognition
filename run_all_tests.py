"""
主控脚本 - 一键运行所有算法验证和测试
按照正确的顺序执行所有步骤
"""
import os
import sys
import subprocess
import time


def print_banner(text):
    """打印带格式的标题"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def check_dependencies():
    """检查必要的Python库"""
    print_banner("步骤 0: 检查依赖")

    required_packages = {
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib'
    }

    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"✗ {package} 未安装")
            missing.append(package)

    if missing:
        print(f"\n请安装缺失的包:")
        print(f"pip install {' '.join(missing)}")
        return False

    print("\n所有依赖已满足!")
    return True


def run_step(step_num, description, script_name):
    """运行单个测试步骤"""
    print_banner(f"步骤 {step_num}: {description}")

    try:
        print(f"正在运行: {script_name}")
        print("-" * 80)

        # 运行脚本
        result = subprocess.run([sys.executable, script_name],
                                capture_output=False,
                                text=True)

        if result.returncode == 0:
            print(f"\n✓ {description} 完成!")
            return True
        else:
            print(f"\n✗ {description} 失败!")
            return False

    except Exception as e:
        print(f"\n✗ 运行出错: {str(e)}")
        return False


def generate_final_report():
    """生成最终的总结报告"""
    print_banner("生成最终报告")

    report = []
    report.append("=" * 80)
    report.append("DSP-Free L1-KNN 算法验证 - 最终报告")
    report.append("=" * 80)
    report.append("")

    # 检查生成的文件
    expected_files = {
        '算法验证': [
            'knn_k_comparison_iris.png',
            'knn_k_comparison_digits.png',
            'iris_hardware_dataset.txt',
            'digits_hardware_dataset.txt'
        ],
        '软件基准': [
            # 基准测试不生成文件，只输出结果
        ],
        '硬件文件': [
            'iris_train_data_bram.mem',
            'iris_train_data.coe',
            'iris_test_vectors.mem',
            'iris_knn_data.h',
            'digits_train_data_bram.mem',
            'digits_train_data.coe',
            'digits_test_vectors.mem',
            'digits_knn_data.h'
        ]
    }

    report.append("【生成的文件清单】")
    report.append("")

    all_present = True
    for category, files in expected_files.items():
        report.append(f"{category}:")
        if not files:
            report.append("  (仅输出到控制台)")
            continue

        for file in files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                report.append(f"  ✓ {file} ({size} bytes)")
            else:
                report.append(f"  ✗ {file} (未找到)")
                all_present = False
        report.append("")

    report.append("=" * 80)
    report.append("【下一步操作】")
    report.append("")
    report.append("1. 查看精度对比图:")
    report.append("   - knn_k_comparison_iris.png")
    report.append("   - knn_k_comparison_digits.png")
    report.append("")
    report.append("2. 使用硬件文件进行Verilog开发:")
    report.append("   - *_train_data.coe        → Vivado Block Memory初始化")
    report.append("   - *_train_data_bram.mem   → 仿真时BRAM初始化")
    report.append("   - *_test_vectors.mem      → Verilog testbench测试向量")
    report.append("   - *_knn_data.h            → ARM端C程序测试")
    report.append("")
    report.append("3. 开始硬件设计:")
    report.append("   - 参考文档中的 C. 硬件实现步骤")
    report.append("   - 实现 l1_distance_unit.v")
    report.append("   - 实现 single_point_distance_calc.v")
    report.append("   - 实现 top_k_sorter.v")
    report.append("   - 实现 knn_accelerator_top.v")
    report.append("")
    report.append("4. 性能目标:")
    report.append("   - 软件基线已在步骤2中测量")
    report.append("   - 目标: 实现100x加速")
    report.append("   - 关键: DSP使用率 = 0")
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
    print("""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                                                                        ║
    ║           DSP-Free L1-KNN 并行加速器 - 算法验证系统                    ║
    ║                                                                        ║
    ║                      Algorithm Validation Pipeline                     ║
    ║                                                                        ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """)

    start_time = time.time()

    # 步骤0: 检查依赖
    if not check_dependencies():
        print("\n程序终止: 请先安装缺失的依赖包")
        return

    input("\n按回车键继续...")

    # 步骤1: 算法验证
    success = run_step(
        1,
        "算法精度验证 (L1 vs L2)",
        "knn_algorithm_validator.py"
    )
    if not success:
        print("\n警告: 算法验证未完全成功，但可以继续")

    time.sleep(2)

    # 步骤2: 软件基准测试
    success = run_step(
        2,
        "软件基准性能测试",
        "software_baseline_benchmark.py"
    )
    if not success:
        print("\n警告: 基准测试未完全成功，但可以继续")

    time.sleep(2)

    # 步骤3: 硬件测试文件生成
    success = run_step(
        3,
        "硬件测试文件生成",
        "hardware_testbench_generator.py"
    )
    if not success:
        print("\n警告: 硬件文件生成未完全成功")

    time.sleep(2)

    # 生成最终报告
    all_files_present = generate_final_report()

    # 统计运行时间
    elapsed_time = time.time() - start_time

    print_banner("运行完成")
    print(f"总耗时: {elapsed_time:.2f} 秒")

    if all_files_present:
        print("\n✓ 所有步骤成功完成!")
        print("✓ 所有必要文件已生成")
        print("\n你现在可以:")
        print("  1. 查看生成的图表和报告")
        print("  2. 使用硬件文件开始Verilog开发")
        print("  3. 参考软件基准数据设定加速目标")
    else:
        print("\n⚠ 部分文件未生成，请检查错误信息")

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