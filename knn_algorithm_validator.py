"""
L1-KNN算法验证器
用于验证L1-norm KNN相比L2-norm KNN的精度损失
"""
import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time


class KNNValidator:
    def __init__(self, dataset_name='iris', k=5):
        """
        初始化KNN验证器
        Args:
            dataset_name: 'iris' 或 'digits'
            k: KNN的K值
        """
        self.dataset_name = dataset_name
        self.k = k
        self.load_dataset()

    def load_dataset(self):
        """加载并预处理数据集"""
        if self.dataset_name == 'iris':
            data = load_iris()
            print(
                f"使用Iris数据集: {data.data.shape[0]}个样本, {data.data.shape[1]}个特征, {len(np.unique(data.target))}个类别")
        elif self.dataset_name == 'digits':
            data = load_digits()
            print(
                f"使用Digits数据集: {data.data.shape[0]}个样本, {data.data.shape[1]}个特征, {len(np.unique(data.target))}个类别")
        else:
            raise ValueError("不支持的数据集")

        # 数据归一化到[0, 255]范围（模拟8位量化）
        scaler = MinMaxScaler(feature_range=(0, 255))
        X_scaled = scaler.fit_transform(data.data)

        # 量化为整数（模拟硬件中的8位数据）
        self.X = np.round(X_scaled).astype(np.uint8)
        self.y = data.target
        self.feature_names = data.feature_names if hasattr(data, 'feature_names') else None

        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )

        print(f"训练集大小: {self.X_train.shape[0]}, 测试集大小: {self.X_test.shape[0]}")
        print(f"特征维度: {self.X_train.shape[1]}")
        print(f"数据类型: {self.X_train.dtype}, 范围: [{self.X_train.min()}, {self.X_train.max()}]")

    def compare_metrics(self):
        """对比L1和L2距离的分类性能"""
        results = {}

        # L2-norm (欧氏距离)
        print("\n" + "=" * 50)
        print("测试 L2-norm (欧氏距离) KNN")
        print("=" * 50)
        knn_l2 = KNeighborsClassifier(n_neighbors=self.k, metric='euclidean')
        start_time = time.time()
        knn_l2.fit(self.X_train, self.y_train)
        train_time_l2 = time.time() - start_time

        start_time = time.time()
        accuracy_l2 = knn_l2.score(self.X_test, self.y_test)
        predict_time_l2 = (time.time() - start_time) / len(self.X_test)

        results['L2'] = {
            'accuracy': accuracy_l2,
            'train_time': train_time_l2,
            'predict_time_per_sample': predict_time_l2
        }
        print(f"L2-norm 准确率: {accuracy_l2 * 100:.2f}%")
        print(f"L2-norm 单样本预测时间: {predict_time_l2 * 1000:.4f} ms")

        # L1-norm (曼哈顿距离)
        print("\n" + "=" * 50)
        print("测试 L1-norm (曼哈顿距离) KNN")
        print("=" * 50)
        knn_l1 = KNeighborsClassifier(n_neighbors=self.k, metric='manhattan')
        start_time = time.time()
        knn_l1.fit(self.X_train, self.y_train)
        train_time_l1 = time.time() - start_time

        start_time = time.time()
        accuracy_l1 = knn_l1.score(self.X_test, self.y_test)
        predict_time_l1 = (time.time() - start_time) / len(self.X_test)

        results['L1'] = {
            'accuracy': accuracy_l1,
            'train_time': train_time_l1,
            'predict_time_per_sample': predict_time_l1
        }
        print(f"L1-norm 准确率: {accuracy_l1 * 100:.2f}%")
        print(f"L1-norm 单样本预测时间: {predict_time_l1 * 1000:.4f} ms")

        # 计算精度损失
        accuracy_loss = (accuracy_l2 - accuracy_l1) * 100
        print("\n" + "=" * 50)
        print("性能对比")
        print("=" * 50)
        print(f"精度损失: {accuracy_loss:.2f}%")
        print(f"L1相比L2的精度保持率: {(accuracy_l1 / accuracy_l2) * 100:.2f}%")

        return results

    def test_different_k_values(self, k_range=range(1, 21)):
        """测试不同K值下L1和L2的性能"""
        l1_accuracies = []
        l2_accuracies = []
        k_values = list(k_range)

        print("\n" + "=" * 50)
        print("测试不同K值的影响")
        print("=" * 50)

        for k in k_values:
            knn_l1 = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
            knn_l2 = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

            knn_l1.fit(self.X_train, self.y_train)
            knn_l2.fit(self.X_train, self.y_train)

            acc_l1 = knn_l1.score(self.X_test, self.y_test)
            acc_l2 = knn_l2.score(self.X_test, self.y_test)

            l1_accuracies.append(acc_l1)
            l2_accuracies.append(acc_l2)

            if k % 5 == 1 or k <= 5:
                print(
                    f"K={k:2d}: L2={acc_l2 * 100:.2f}%, L1={acc_l1 * 100:.2f}%, 差距={abs(acc_l2 - acc_l1) * 100:.2f}%")

        # 绘制结果
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, np.array(l2_accuracies) * 100, 'b-o', label='L2-norm (Euclidean)', linewidth=2)
        plt.plot(k_values, np.array(l1_accuracies) * 100, 'r-s', label='L1-norm (Manhattan)', linewidth=2)
        plt.xlabel('K Value', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title(f'KNN Accuracy vs K Value ({self.dataset_name.upper()} Dataset)', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'knn_k_comparison_{self.dataset_name}.png', dpi=300)
        print(f"\n图表已保存为: knn_k_comparison_{self.dataset_name}.png")

        return k_values, l1_accuracies, l2_accuracies

    def export_hardware_dataset(self, output_file='hardware_dataset.txt'):
        """
        导出用于硬件测试的数据集
        格式: 每行一个样本，特征用空格分隔，最后一列是标签
        """
        print(f"\n导出硬件测试数据集到: {output_file}")

        with open(output_file, 'w') as f:
            # 写入元数据
            f.write(f"# Dataset: {self.dataset_name}\n")
            f.write(f"# Train samples: {self.X_train.shape[0]}\n")
            f.write(f"# Test samples: {self.X_test.shape[0]}\n")
            f.write(f"# Features: {self.X_train.shape[1]}\n")
            f.write(f"# Classes: {len(np.unique(self.y_train))}\n")
            f.write(f"# Data format: feature1 feature2 ... featureN label\n")
            f.write("#\n")

            # 写入训练集
            f.write("# TRAINING SET\n")
            for i in range(len(self.X_train)):
                features = ' '.join(map(str, self.X_train[i]))
                f.write(f"{features} {self.y_train[i]}\n")

            # 写入测试集
            f.write("# TEST SET\n")
            for i in range(len(self.X_test)):
                features = ' '.join(map(str, self.X_test[i]))
                f.write(f"{features} {self.y_test[i]}\n")

        print(f"导出完成! 训练集: {len(self.X_train)}个样本, 测试集: {len(self.X_test)}个样本")

    def generate_summary_report(self):
        """生成完整的性能报告"""
        print("\n" + "=" * 70)
        print("L1-KNN 硬件加速器 - 算法验证报告")
        print("=" * 70)

        results = self.compare_metrics()

        print("\n【结论】")
        print(f"✓ L1-norm KNN在{self.dataset_name.upper()}数据集上的精度: {results['L1']['accuracy'] * 100:.2f}%")
        print(f"✓ L2-norm KNN在{self.dataset_name.upper()}数据集上的精度: {results['L2']['accuracy'] * 100:.2f}%")
        print(f"✓ 精度差距: {abs(results['L2']['accuracy'] - results['L1']['accuracy']) * 100:.2f}%")

        if abs(results['L2']['accuracy'] - results['L1']['accuracy']) < 0.05:
            print("✓ L1-norm精度损失可接受 (<5%), 适合硬件实现!")

        print(f"\n【数据集信息】")
        print(f"✓ 训练集样本数 N = {len(self.X_train)}")
        print(f"✓ 特征维度 D = {self.X_train.shape[1]}")
        print(f"✓ 类别数 = {len(np.unique(self.y_train))}")
        print(f"✓ 数据位宽 = 8-bit (uint8)")

        print(f"\n【硬件设计参考】")
        print(f"✓ 每次查询需要计算 {len(self.X_train)} 个距离")
        print(f"✓ 每个距离需要 {self.X_train.shape[1]} 次L1距离计算")
        print(f"✓ 总计算量: {len(self.X_train) * self.X_train.shape[1]} 次 |a-b| 操作")

        return results


def main():
    """主函数：运行完整的验证流程"""
    print("=" * 70)
    print("DSP-Free L1-KNN 算法验证程序")
    print("=" * 70)

    # 测试Iris数据集
    print("\n>>> 开始验证 Iris 数据集")
    validator_iris = KNNValidator(dataset_name='iris', k=5)
    validator_iris.generate_summary_report()
    validator_iris.test_different_k_values(k_range=range(1, 16))
    validator_iris.export_hardware_dataset('iris_hardware_dataset.txt')

    # 测试Digits数据集
    print("\n\n>>> 开始验证 Digits 数据集")
    validator_digits = KNNValidator(dataset_name='digits', k=5)
    validator_digits.generate_summary_report()
    validator_digits.test_different_k_values(k_range=range(1, 16))
    validator_digits.export_hardware_dataset('digits_hardware_dataset.txt')

    print("\n" + "=" * 70)
    print("验证完成! 所有结果已保存.")
    print("=" * 70)


if __name__ == "__main__":
    main()