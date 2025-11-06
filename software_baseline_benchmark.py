"""
软件基准测试程序
用于在ARM处理器上测试纯软件L1-KNN的性能
这将作为硬件加速器的对比基准
"""
import numpy as np
import time
from typing import Tuple, List


class SoftwareL1KNN:
    """
    纯软件实现的L1-KNN分类器
    完全模拟硬件实现的算法流程
    """

    def __init__(self, k=5):
        self.k = k
        self.train_data = None
        self.train_labels = None
        self.n_samples = 0
        self.n_features = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """训练（实际上就是存储训练数据）"""
        self.train_data = X_train.astype(np.uint8)
        self.train_labels = y_train.astype(np.uint8)
        self.n_samples = len(X_train)
        self.n_features = X_train.shape[1]
        print(f"训练集加载完成: {self.n_samples}个样本, {self.n_features}个特征")

    def compute_l1_distance(self, query_point: np.ndarray, train_point: np.ndarray) -> int:
        """
        计算L1距离（曼哈顿距离）
        模拟硬件中的实现：逐特征计算绝对差并累加
        """
        distance = 0
        for i in range(self.n_features):
            # 模拟硬件：abs(a - b) 使用减法和条件判断
            if query_point[i] > train_point[i]:
                delta = int(query_point[i]) - int(train_point[i])
            else:
                delta = int(train_point[i]) - int(query_point[i])
            distance += delta
        return distance

    def find_top_k_neighbors(self, query_point: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        找到K个最近邻居
        模拟硬件中的Top-K排序器
        """
        # 初始化Top-K列表（距离，标签）
        top_k_distances = [float('inf')] * self.k
        top_k_labels = [-1] * self.k

        # 遍历所有训练样本
        for j in range(self.n_samples):
            # 计算距离
            distance = self.compute_l1_distance(query_point, self.train_data[j])
            label = int(self.train_labels[j])

            # 查找插入位置（从大到小）
            insert_pos = -1
            for i in range(self.k):
                if distance < top_k_distances[i]:
                    insert_pos = i
                    break

            # 如果需要插入
            if insert_pos != -1:
                # 移位并插入（模拟硬件的移位寄存器）
                for i in range(self.k - 1, insert_pos, -1):
                    top_k_distances[i] = top_k_distances[i - 1]
                    top_k_labels[i] = top_k_labels[i - 1]
                top_k_distances[insert_pos] = distance
                top_k_labels[insert_pos] = label

        return top_k_distances, top_k_labels

    def vote(self, labels: List[int]) -> int:
        """
        投票机制：找到出现次数最多的标签
        模拟硬件中的投票电路
        """
        # 使用简单的计数器数组（假设类别数不超过16）
        vote_counts = [0] * 16

        for label in labels:
            if label >= 0:  # 忽略无效标签
                vote_counts[label] += 1

        # 找到票数最多的标签
        max_votes = 0
        winner_label = 0
        for i in range(16):
            if vote_counts[i] > max_votes:
                max_votes = vote_counts[i]
                winner_label = i

        return winner_label

    def predict_single(self, query_point: np.ndarray) -> int:
        """
        预测单个样本
        完整模拟硬件流程：距离计算 -> Top-K排序 -> 投票
        """
        # 步骤1: 找到K个最近邻居
        _, top_k_labels = self.find_top_k_neighbors(query_point)

        # 步骤2: 投票
        prediction = self.vote(top_k_labels)

        return prediction

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """批量预测"""
        predictions = []
        for i in range(len(X_test)):
            pred = self.predict_single(X_test[i])
            predictions.append(pred)
        return np.array(predictions)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """评估性能"""
        predictions = []
        inference_times = []

        print(f"\n开始评估 {len(X_test)} 个测试样本...")

        for i in range(len(X_test)):
            start_time = time.perf_counter()
            pred = self.predict_single(X_test[i])
            end_time = time.perf_counter()

            predictions.append(pred)
            inference_times.append((end_time - start_time) * 1000)  # 转换为ms

        predictions = np.array(predictions)
        accuracy = np.mean(predictions == y_test)
        avg_time = np.mean(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)

        return {
            'accuracy': accuracy,
            'avg_inference_time_ms': avg_time,
            'min_inference_time_ms': min_time,
            'max_inference_time_ms': max_time,
            'total_samples': len(X_test),
            'correct_predictions': np.sum(predictions == y_test)
        }


def load_hardware_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """从导出的硬件数据集文件加载数据"""
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    current_section = None

    with open(filename, 'r') as f:
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

    return (np.array(train_data, dtype=np.uint8),
            np.array(train_labels, dtype=np.uint8),
            np.array(test_data, dtype=np.uint8),
            np.array(test_labels, dtype=np.uint8))


def run_benchmark(dataset_name='iris', k=5):
    """运行完整的基准测试"""
    print("=" * 70)
    print(f"软件基准测试 - {dataset_name.upper()} 数据集")
    print("=" * 70)

    # 加载数据
    filename = f'{dataset_name}_hardware_dataset.txt'
    try:
        X_train, y_train, X_test, y_test = load_hardware_dataset(filename)
        print(f"✓ 数据加载成功")
    except FileNotFoundError:
        print(f"✗ 文件 {filename} 不存在，请先运行 knn_algorithm_validator.py")
        return None

    # 创建KNN分类器
    knn = SoftwareL1KNN(k=k)
    knn.fit(X_train, y_train)

    # 运行性能评估
    print(f"\n运行性能基准测试 (K={k})...")
    results = knn.evaluate(X_test, y_test)

    # 打印结果
    print("\n" + "=" * 70)
    print("基准测试结果")
    print("=" * 70)
    print(f"准确率: {results['accuracy'] * 100:.2f}%")
    print(f"正确预测: {results['correct_predictions']}/{results['total_samples']}")
    print(f"\n【性能指标】")
    print(f"平均推理时间: {results['avg_inference_time_ms']:.4f} ms")
    print(f"最小推理时间: {results['min_inference_time_ms']:.4f} ms")
    print(f"最大推理时间: {results['max_inference_time_ms']:.4f} ms")

    # 计算硬件加速目标
    print(f"\n【硬件加速目标】")
    target_speedup = 100
    target_time_us = results['avg_inference_time_ms'] * 1000 / target_speedup
    print(f"✓ 软件基线: {results['avg_inference_time_ms']:.4f} ms")
    print(f"✓ {target_speedup}x加速目标: {target_time_us:.2f} μs")
    print(f"✓ 硬件需要达到: < {target_time_us:.2f} μs/推理")

    # 估算硬件资源
    print(f"\n【硬件实现估算】")
    n_samples = len(X_train)
    n_features = X_train.shape[1]
    print(f"✓ 距离计算单元数: {n_features} (并行计算)")
    print(f"✓ Top-K排序器大小: {k}")
    print(f"✓ 每次查询的计算次数: {n_samples * n_features}")
    print(f"✓ BRAM需求（训练集）: ~{n_samples * n_features} bytes")

    return results


def main():
    """主函数"""
    print("=" * 70)
    print("DSP-Free L1-KNN 软件基准测试程序")
    print("用于ARM处理器性能基线测量")
    print("=" * 70)

    # 测试Iris数据集
    print("\n>>> 测试 IRIS 数据集")
    results_iris = run_benchmark('iris', k=5)

    # 测试Digits数据集
    print("\n\n>>> 测试 DIGITS 数据集")
    results_digits = run_benchmark('digits', k=5)

    # 生成对比报告
    if results_iris and results_digits:
        print("\n" + "=" * 70)
        print("数据集对比总结")
        print("=" * 70)
        print(
            f"Iris    - 准确率: {results_iris['accuracy'] * 100:.2f}%, 平均时间: {results_iris['avg_inference_time_ms']:.4f} ms")
        print(
            f"Digits  - 准确率: {results_digits['accuracy'] * 100:.2f}%, 平均时间: {results_digits['avg_inference_time_ms']:.4f} ms")
        print("\n注意: Digits数据集特征更多，推理时间更长，是更好的加速测试场景")

    print("\n" + "=" * 70)
    print("基准测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()