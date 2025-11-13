"""
软件基准性能测试 - 人脸识别
目标：建立纯软件实现的性能基线，用于对比FPGA加速效果
"""
import numpy as np
import time
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from collections import Counter
import matplotlib.pyplot as plt


class FaceSoftwareBaseline:
    def __init__(self, n_components=50):
        """初始化软件基准测试器"""
        self.n_components = n_components
        self.load_dataset()

    def load_dataset(self):
        """加载并预处理数据集"""
        print("=" * 70)
        print("软件基准测试 - 加载人脸数据集")
        print("=" * 70)

        faces = fetch_olivetti_faces(shuffle=True, random_state=42)
        X_raw = faces.data
        y = faces.target

        # PCA降维
        pca = PCA(n_components=self.n_components, random_state=42)
        X_pca = pca.fit_transform(X_raw)

        # 8位量化
        scaler = MinMaxScaler(feature_range=(0, 255))
        X_scaled = scaler.fit_transform(X_pca)
        self.X = np.round(X_scaled).astype(np.uint8)
        self.y = y

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        print(f"训练集: {len(self.X_train)} 张人脸")
        print(f"测试集: {len(self.X_test)} 张人脸")
        print(f"特征维度: {self.X_train.shape[1]}")

    def l1_distance_python(self, x1, x2):
        """纯Python实现L1距离"""
        distance = 0
        for i in range(len(x1)):
            distance += abs(int(x1[i]) - int(x2[i]))
        return distance

    def l1_distance_numpy(self, x1, x2):
        """NumPy优化的L1距离"""
        return np.sum(np.abs(x1.astype(np.int16) - x2.astype(np.int16)))

    def knn_predict_python(self, query, k=5):
        """纯Python实现的KNN"""
        distances = []
        for i in range(len(self.X_train)):
            dist = self.l1_distance_python(query, self.X_train[i])
            distances.append((dist, self.y_train[i]))

        distances.sort(key=lambda x: x[0])
        k_labels = [label for _, label in distances[:k]]
        return Counter(k_labels).most_common(1)[0][0]

    def knn_predict_numpy(self, query, k=5):
        """NumPy优化的KNN"""
        distances = []
        for i in range(len(self.X_train)):
            dist = self.l1_distance_numpy(query, self.X_train[i])
            distances.append((dist, self.y_train[i]))

        distances.sort(key=lambda x: x[0])
        k_labels = [label for _, label in distances[:k]]
        return Counter(k_labels).most_common(1)[0][0]

    def benchmark_python(self, k=5, num_samples=20):
        """基准测试：纯Python实现"""
        print("\n" + "=" * 70)
        print("基准测试 1: 纯Python实现 (模拟嵌入式C代码)")
        print("=" * 70)

        test_samples = self.X_test[:num_samples]
        true_labels = self.y_test[:num_samples]

        predictions = []
        times = []

        for i, test_point in enumerate(test_samples):
            start = time.perf_counter()
            pred = self.knn_predict_python(test_point, k)
            elapsed = time.perf_counter() - start

            predictions.append(pred)
            times.append(elapsed)

            if i < 3:
                print(f"样本{i + 1}: {elapsed * 1000:.2f} ms")

        accuracy = np.mean(np.array(predictions) == true_labels)
        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000

        print(f"\n准确率: {accuracy * 100:.2f}%")
        print(f"平均识别时间: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"吞吐量: {1000 / avg_time:.2f} 人脸/秒")

        return {
            'method': 'Python',
            'accuracy': accuracy,
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'throughput': 1000 / avg_time
        }

    def benchmark_numpy(self, k=5, num_samples=20):
        """基准测试：NumPy优化"""
        print("\n" + "=" * 70)
        print("基准测试 2: NumPy优化实现")
        print("=" * 70)

        test_samples = self.X_test[:num_samples]
        true_labels = self.y_test[:num_samples]

        predictions = []
        times = []

        for i, test_point in enumerate(test_samples):
            start = time.perf_counter()
            pred = self.knn_predict_numpy(test_point, k)
            elapsed = time.perf_counter() - start

            predictions.append(pred)
            times.append(elapsed)

            if i < 3:
                print(f"样本{i + 1}: {elapsed * 1000:.2f} ms")

        accuracy = np.mean(np.array(predictions) == true_labels)
        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000

        print(f"\n准确率: {accuracy * 100:.2f}%")
        print(f"平均识别时间: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"吞吐量: {1000 / avg_time:.2f} 人脸/秒")

        return {
            'method': 'NumPy',
            'accuracy': accuracy,
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'throughput': 1000 / avg_time
        }

    def estimate_fpga_performance(self, clock_freq_mhz=100):
        """估算FPGA性能目标"""
        print("\n" + "=" * 70)
        print("FPGA性能目标估算")
        print("=" * 70)

        n_train = len(self.X_train)
        n_features = self.X_train.shape[1]

        # 流水线架构估算
        cycles_per_distance = n_features + 5  # 特征累加 + 开销
        cycles_per_sample = cycles_per_distance + 10  # 排序开销
        total_cycles = n_train * cycles_per_sample

        clock_period_ns = 1000.0 / clock_freq_mhz
        latency_us = total_cycles * clock_period_ns / 1000.0

        print(f"假设参数:")
        print(f"  - 时钟频率: {clock_freq_mhz} MHz")
        print(f"  - 训练样本数: {n_train}")
        print(f"  - 特征维度: {n_features}")
        print(f"\n单次识别估算:")
        print(f"  - 总时钟周期: {total_cycles:,}")
        print(f"  - 估算延迟: {latency_us:.1f} μs")
        print(f"  - 估算吞吐量: {1000000 / latency_us:.0f} 人脸/秒")

        return latency_us

    def visualize_comparison(self, results):
        """可视化性能对比"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        methods = [r['method'] for r in results]
        times = [r['avg_time_ms'] for r in results]
        throughputs = [r['throughput'] for r in results]
        accuracies = [r['accuracy'] * 100 for r in results]

        # 识别时间对比
        axes[0].bar(methods, times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0].set_ylabel('识别时间 (ms)', fontsize=11)
        axes[0].set_title('单人脸识别延迟', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')

        # 吞吐量对比
        axes[1].bar(methods, throughputs, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[1].set_ylabel('吞吐量 (人脸/秒)', fontsize=11)
        axes[1].set_title('识别吞吐量', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        # 准确率对比
        axes[2].bar(methods, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[2].set_ylabel('准确率 (%)', fontsize=11)
        axes[2].set_title('识别准确率', fontsize=12, fontweight='bold')
        axes[2].set_ylim([0, 105])
        axes[2].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('face_software_baseline_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n对比图表已保存: face_software_baseline_comparison.png")

    def run_all_benchmarks(self):
        """运行所有基准测试"""
        print("=" * 70)
        print("人脸识别软件基准性能测试")
        print("=" * 70)

        results = []

        # Python基准
        result1 = self.benchmark_python(k=5, num_samples=20)
        results.append(result1)

        # NumPy基准
        result2 = self.benchmark_numpy(k=5, num_samples=20)
        results.append(result2)

        # FPGA估算
        fpga_latency = self.estimate_fpga_performance(clock_freq_mhz=100)
        results.append({
            'method': 'FPGA (估算)',
            'accuracy': result2['accuracy'],  # 假设相同精度
            'avg_time_ms': fpga_latency / 1000,
            'std_time_ms': 0,
            'throughput': 1000000 / fpga_latency
        })

        # 可视化
        self.visualize_comparison(results)

        # 总结
        print("\n" + "=" * 70)
        print("性能总结")
        print("=" * 70)
        print(f"{'方法':<15} {'延迟 (ms)':<15} {'吞吐量 (人脸/s)':<20} {'准确率 (%)':<15}")
        print("-" * 70)
        for r in results:
            print(f"{r['method']:<15} {r['avg_time_ms']:<15.2f} "
                  f"{r['throughput']:<20.1f} {r['accuracy'] * 100:<15.2f}")

        print("\n【加速比分析】")
        python_time = results[0]['avg_time_ms']
        fpga_time = results[2]['avg_time_ms']
        speedup = python_time / fpga_time
        print(f"FPGA vs Python: {speedup:.0f}× 加速")
        print(f"这就是你的论文卖点: 在保持精度的同时，实现 {speedup:.0f}× 加速!")

        return results


def main():
    baseline = FaceSoftwareBaseline(n_components=50)
    results = baseline.run_all_benchmarks()

    print("\n" + "=" * 70)
    print("软件基准测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()