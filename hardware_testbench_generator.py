"""
硬件测试向量生成器
生成用于Verilog testbench的测试数据
"""
import numpy as np
from typing import List, Tuple


class HardwareTestbenchGenerator:
    """生成Verilog仿真所需的测试向量"""

    def __init__(self, dataset_file: str):
        self.dataset_file = dataset_file
        self.load_dataset()

    def load_dataset(self):
        """加载数据集"""
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []

        current_section = None

        with open(self.dataset_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    if 'TRAINING SET' in line:
                        current_section = 'train'
                    elif 'TEST SET' in line:
                        current_section = 'test'
                    continue

                parts = line.split()
                features = [int(x) for x in parts[:-1]]
                label = int(parts[-1])

                if current_section == 'train':
                    train_data.append(features)
                    train_labels.append(label)
                elif current_section == 'test':
                    test_data.append(features)
                    test_labels.append(label)

        self.X_train = np.array(train_data, dtype=np.uint8)
        self.y_train = np.array(train_labels, dtype=np.uint8)
        self.X_test = np.array(test_data, dtype=np.uint8)
        self.y_test = np.array(test_labels, dtype=np.uint8)

        print(f"数据集加载完成:")
        print(f"  训练集: {len(self.X_train)} 样本")
        print(f"  测试集: {len(self.X_test)} 样本")
        print(f"  特征数: {self.X_train.shape[1]}")

    def generate_bram_init_file(self, output_file='train_data_bram.mem'):
        """
        生成BRAM初始化文件
        格式: 每行包含一个训练样本的所有特征 + 标签
        """
        print(f"\n生成BRAM初始化文件: {output_file}")

        with open(output_file, 'w') as f:
            f.write("// BRAM初始化文件 - 训练数据集\n")
            f.write(f"// 格式: feature0 feature1 ... featureN label\n")
            f.write(f"// 每行一个训练样本\n")
            f.write(f"// 样本数: {len(self.X_train)}\n")
            f.write(f"// 特征数: {self.X_train.shape[1]}\n\n")

            for i in range(len(self.X_train)):
                # 将特征转换为十六进制
                hex_values = [f"{val:02X}" for val in self.X_train[i]]
                label_hex = f"{self.y_train[i]:02X}"
                f.write(f"@{i:04X} {' '.join(hex_values)} {label_hex}\n")

        print(f"  ✓ 已生成 {len(self.X_train)} 个训练样本")

    def generate_test_vectors(self, output_file='test_vectors.mem', num_tests=10):
        """
        生成Verilog testbench测试向量
        选择前N个测试样本
        """
        print(f"\n生成测试向量文件: {output_file}")
        num_tests = min(num_tests, len(self.X_test))

        with open(output_file, 'w') as f:
            f.write("// Verilog Testbench 测试向量\n")
            f.write(f"// 测试样本数: {num_tests}\n")
            f.write(f"// 特征数: {self.X_test.shape[1]}\n\n")

            for i in range(num_tests):
                f.write(f"// 测试样本 {i + 1}\n")
                f.write(f"// 期望标签: {self.y_test[i]}\n")

                # 生成查询向量
                hex_values = [f"8'h{val:02X}" for val in self.X_test[i]]
                f.write(f"query_vector = {{{', '.join(hex_values)}}};\n")
                f.write(f"expected_label = {self.y_test[i]};\n")
                f.write(f"#10 start = 1;\n")
                f.write(f"#10 start = 0;\n")
                f.write(f"wait(done);\n")
                f.write(f"if (result_label == expected_label) $display(\"Test {i + 1} PASSED\");\n")
                f.write(f"else $display(\"Test {i + 1} FAILED: Expected %d, Got %d\", expected_label, result_label);\n")
                f.write(f"#10;\n\n")

        print(f"  ✓ 已生成 {num_tests} 个测试向量")

    def generate_coe_file(self, output_file='train_data.coe'):
        """
        生成Xilinx COE文件格式
        用于Vivado中直接初始化Block Memory
        """
        print(f"\n生成COE文件: {output_file}")

        with open(output_file, 'w') as f:
            f.write("; Xilinx COE文件 - 训练数据集\n")
            f.write(f"; 样本数: {len(self.X_train)}\n")
            f.write(f"; 特征数: {self.X_train.shape[1]}\n")
            f.write("memory_initialization_radix=16;\n")
            f.write("memory_initialization_vector=\n")

            for i in range(len(self.X_train)):
                # 将整个样本打包成一个宽字（特征 + 标签）
                hex_values = [f"{val:02X}" for val in self.X_train[i]]
                label_hex = f"{self.y_train[i]:02X}"

                # 拼接成一个长十六进制数
                full_hex = ''.join(hex_values) + label_hex

                if i < len(self.X_train) - 1:
                    f.write(f"{full_hex},\n")
                else:
                    f.write(f"{full_hex};\n")

        print(f"  ✓ COE文件生成完成")

    def generate_c_header(self, output_file='knn_data.h'):
        """
        生成C头文件
        用于ARM端的软件测试
        """
        print(f"\n生成C头文件: {output_file}")

        with open(output_file, 'w') as f:
            f.write("// KNN数据集C头文件\n")
            f.write("// 用于ARM端软件测试\n\n")
            f.write("#ifndef KNN_DATA_H\n")
            f.write("#define KNN_DATA_H\n\n")
            f.write("#include <stdint.h>\n\n")

            # 定义常量
            f.write(f"#define N_TRAIN_SAMPLES {len(self.X_train)}\n")
            f.write(f"#define N_TEST_SAMPLES {len(self.X_test)}\n")
            f.write(f"#define N_FEATURES {self.X_train.shape[1]}\n")
            f.write(f"#define N_CLASSES {len(np.unique(self.y_train))}\n\n")

            # 训练集数据
            f.write("// 训练集特征\n")
            f.write("uint8_t train_features[N_TRAIN_SAMPLES][N_FEATURES] = {\n")
            for i in range(len(self.X_train)):
                features_str = ', '.join([str(val) for val in self.X_train[i]])
                if i < len(self.X_train) - 1:
                    f.write(f"    {{{features_str}}},\n")
                else:
                    f.write(f"    {{{features_str}}}\n")
            f.write("};\n\n")

            # 训练集标签
            f.write("// 训练集标签\n")
            f.write("uint8_t train_labels[N_TRAIN_SAMPLES] = {\n    ")
            for i in range(len(self.y_train)):
                f.write(str(self.y_train[i]))
                if i < len(self.y_train) - 1:
                    f.write(", ")
                    if (i + 1) % 20 == 0:
                        f.write("\n    ")
            f.write("\n};\n\n")

            # 测试集数据（只包含前10个）
            num_test = min(10, len(self.X_test))
            f.write("// 测试集特征（前10个样本）\n")
            f.write(f"uint8_t test_features[{num_test}][N_FEATURES] = {{\n")
            for i in range(num_test):
                features_str = ', '.join([str(val) for val in self.X_test[i]])
                if i < num_test - 1:
                    f.write(f"    {{{features_str}}},\n")
                else:
                    f.write(f"    {{{features_str}}}\n")
            f.write("};\n\n")

            # 测试集标签
            f.write("// 测试集标签（前10个样本）\n")
            f.write(f"uint8_t test_labels[{num_test}] = {{\n    ")
            for i in range(num_test):
                f.write(str(self.y_test[i]))
                if i < num_test - 1:
                    f.write(", ")
            f.write("\n};\n\n")

            f.write("#endif // KNN_DATA_H\n")

        print(f"  ✓ C头文件生成完成")

    def generate_all(self, prefix=''):
        """生成所有硬件相关文件"""
        if prefix:
            prefix = prefix + '_'

        self.generate_bram_init_file(f'{prefix}train_data_bram.mem')
        self.generate_test_vectors(f'{prefix}test_vectors.mem', num_tests=10)
        self.generate_coe_file(f'{prefix}train_data.coe')
        self.generate_c_header(f'{prefix}knn_data.h')

        print("\n" + "=" * 70)
        print("所有硬件测试文件生成完成!")
        print("=" * 70)
        print(f"✓ {prefix}train_data_bram.mem  - BRAM初始化文件（MEM格式）")
        print(f"✓ {prefix}train_data.coe       - Xilinx COE文件")
        print(f"✓ {prefix}test_vectors.mem     - Verilog测试向量")
        print(f"✓ {prefix}knn_data.h           - C头文件（ARM测试）")


def main():
    """主函数"""
    print("=" * 70)
    print("硬件测试向量生成器")
    print("=" * 70)

    # 生成Iris数据集的硬件文件
    print("\n>>> 处理 IRIS 数据集")
    try:
        gen_iris = HardwareTestbenchGenerator('iris_hardware_dataset.txt')
        gen_iris.generate_all(prefix='iris')
    except FileNotFoundError:
        print("✗ 文件不存在，请先运行 knn_algorithm_validator.py")

    # 生成Digits数据集的硬件文件
    print("\n\n>>> 处理 DIGITS 数据集")
    try:
        gen_digits = HardwareTestbenchGenerator('digits_hardware_dataset.txt')
        gen_digits.generate_all(prefix='digits')
    except FileNotFoundError:
        print("✗ 文件不存在，请先运行 knn_algorithm_validator.py")

    print("\n" + "=" * 70)
    print("生成完成! 文件可直接用于:")
    print("  - Vivado仿真 (testbench)")
    print("  - BRAM初始化")
    print("  - ARM端C程序测试")
    print("=" * 70)


if __name__ == "__main__":
    main()