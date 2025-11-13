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

# 设置中文字体
def setup_chinese_font():
    """设置中文字体，如果失败则使用英文"""
    try:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except Exception as e:
        print(f"Warning: Could not set Chinese font. Using English labels. Error: {e}")
        return False

USE_CHINESE = setup_chinese_font()

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
        """NumPy优化的KNN（向量化）"""
        # 向量化距离计算
        distances_vec = np.sum(np.abs(self.X_train.astype(np.int16) - query.astype(np.int16)), axis=1)

        # 获取前K个最小距离的索引
        k_indices = np.argpartition(distances_vec, k)[:k]
        k_labels = self.y_train[k_indices]

        return Counter(k_labels).most_common(1)[0][0]

    def benchmark_python(self, k=5, num_samples=None):
        """基准测试：纯Python实现"""
        print("\n" + "=" * 70)
        print("基准测试 1: 纯Python实现")
        print("=" * 70)

        # 使用指定数量的样本，None表示全部
        if num_samples is None:
            test_samples = self.X_test
            true_labels = self.y_test
        else:
            test_samples = self.X_test[:num_samples]
            true_labels = self.y_test[:num_samples]

        predictions = []
        times = []

        print(f"测试样本数: {len(test_samples)}")

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

    def benchmark_numpy(self, k=5, num_samples=None):
        """基准测试：NumPy优化"""
        print("\n" + "=" * 70)
        print("基准测试 2: NumPy优化实现")
        print("=" * 70)

        if num_samples is None:
            test_samples = self.X_test
            true_labels = self.y_test
        else:
            test_samples = self.X_test[:num_samples]
            true_labels = self.y_test[:num_samples]

        predictions = []
        times = []

        print(f"测试样本数: {len(test_samples)}")

        # 预热
        _ = self.knn_predict_numpy(test_samples[0], k)

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
        """
        估算FPGA性能
        """
        print("\n" + "=" * 70)
        print("FPGA性能目标估算")
        print("=" * 70)

        n_train = len(self.X_train)
        n_features = self.X_train.shape[1]

        # 流水线估算
        # 1. 查询向量加载: 10周期
        # 2. 流水线填充: n_features周期
        # 3. 距离计算吞吐: 每2周期产生一个距离（保守估计）
        # 4. 排序和投票: 30周期（考虑实际硬件复杂度）

        cycles_load = 10
        cycles_pipeline_fill = n_features
        cycles_streaming = n_train * 2  # 保守估计：2周期/样本
        cycles_sorting_voting = 30

        total_cycles = cycles_load + cycles_pipeline_fill + cycles_streaming + cycles_sorting_voting

        clock_period_ns = 1000.0 / clock_freq_mhz
        latency_us = total_cycles * clock_period_ns / 1000.0

        print(f"假设参数:")
        print(f"  - 时钟频率: {clock_freq_mhz} MHz")
        print(f"  - 训练样本数: {n_train}")
        print(f"  - 特征维度: {n_features}")
        print(f"\n单次识别估算 (保守流水线模型):")
        print(f"  - 查询加载: {cycles_load} 周期")
        print(f"  - 流水线填充: {cycles_pipeline_fill} 周期")
        print(f"  - 数据流处理: {cycles_streaming} 周期 (2周期/样本)")
        print(f"  - 排序投票: {cycles_sorting_voting} 周期")
        print(f"  - 总时钟周期: {total_cycles:,}")
        print(f"  - 估算延迟: {latency_us:.2f} μs")
        print(f"  - 估算吞吐量: {1000000 / latency_us:.1f} 人脸/秒")

        # 乐观估算
        total_cycles_optimistic = cycles_load + cycles_pipeline_fill + n_train + 15
        latency_us_optimistic = total_cycles_optimistic * clock_period_ns / 1000.0
        print(f"\n乐观估算:")
        print(f"  - 总时钟周期: {total_cycles_optimistic:,}")
        print(f"  - 估算延迟: {latency_us_optimistic:.2f} μs")
        print(f"  - 估算吞吐量: {1000000 / latency_us_optimistic:.1f} 人脸/秒")

        return latency_us, latency_us_optimistic

    def visualize_comparison(self, results):
        """可视化性能对比"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        methods = [r['method'] for r in results]
        times = [r['avg_time_ms'] for r in results]
        throughputs = [r['throughput'] for r in results]
        accuracies = [r['accuracy'] * 100 for r in results]

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3']

        # 1. 识别时间对比 (对数坐标)
        bars0 = axes[0].bar(methods, times, color=colors[:len(methods)])
        axes[0].set_ylabel('Latency (ms) [Log Scale]' if not USE_CHINESE else '延迟 (ms) [对数坐标]',
                          fontsize=11)
        axes[0].set_title('Single Face Recognition Latency' if not USE_CHINESE else '单人脸识别延迟',
                         fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].bar_label(bars0, fmt='%.3f')
        axes[0].set_yscale('log')
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=15, ha='right')

        # 2. 吞吐量对比 (对数坐标)
        bars1 = axes[1].bar(methods, throughputs, color=colors[:len(methods)])
        axes[1].set_ylabel('Throughput (faces/s) [Log Scale]' if not USE_CHINESE else '吞吐量 (人脸/秒) [对数坐标]',
                          fontsize=11)
        axes[1].set_title('Recognition Throughput' if not USE_CHINESE else '识别吞吐量',
                         fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].bar_label(bars1, fmt='%.0f')
        axes[1].set_yscale('log')
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=15, ha='right')

        # 3. 准确率对比
        bars2 = axes[2].bar(methods, accuracies, color=colors[:len(methods)])
        axes[2].set_ylabel('Accuracy (%)' if not USE_CHINESE else '准确率 (%)', fontsize=11)
        axes[2].set_title('Recognition Accuracy' if not USE_CHINESE else '识别准确率',
                         fontsize=12, fontweight='bold')
        axes[2].set_ylim([min(accuracies) - 5, 105])
        axes[2].grid(True, alpha=0.3, axis='y')
        axes[2].bar_label(bars2, fmt='%.2f')
        plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=15, ha='right')

        plt.tight_layout()
        plt.savefig('face_software_baseline_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n对比图表已保存: face_software_baseline_comparison.png")

    def run_all_benchmarks(self, use_full_test=True):
        """运行所有基准测试"""
        print("=" * 70)
        print("人脸识别软件基准性能测试")
        print("=" * 70)

        results = []
        K_BENCHMARK = 5  # 使用固定K=5进行公平对比

        # 决定测试样本数
        num_samples = None if use_full_test else 20
        sample_info = f"全部{len(self.X_test)}个" if use_full_test else f"{num_samples}个"
        print(f"\n使用 K={K_BENCHMARK} 进行基准测试 ({sample_info}测试样本)")

        # Python基准
        result1 = self.benchmark_python(k=K_BENCHMARK, num_samples=20)
        results.append(result1)

        # NumPy基准
        result2 = self.benchmark_numpy(k=K_BENCHMARK, num_samples=num_samples)
        results.append(result2)

        # FPGA估算
        fpga_latency_conservative, fpga_latency_optimistic = self.estimate_fpga_performance(clock_freq_mhz=100)

        results.append({
            'method': 'FPGA (Conservative)' if not USE_CHINESE else 'FPGA (保守估算)',
            'accuracy': result2['accuracy'],
            'avg_time_ms': fpga_latency_conservative / 1000.0,
            'std_time_ms': 0,
            'throughput': 1000000.0 / fpga_latency_conservative
        })

        results.append({
            'method': 'FPGA (Optimistic)' if not USE_CHINESE else 'FPGA (乐观估算)',
            'accuracy': result2['accuracy'],
            'avg_time_ms': fpga_latency_optimistic / 1000.0,
            'std_time_ms': 0,
            'throughput': 1000000.0 / fpga_latency_optimistic
        })

        # 可视化
        self.visualize_comparison(results)

        # 总结
        print("\n" + "=" * 70)
        print("性能总结")
        print("=" * 70)
        print(f"{'方法':<20} {'延迟 (ms)':<15} {'吞吐量 (人脸/s)':<20} {'准确率 (%)':<15}")
        print("-" * 75)
        for r in results:
            print(f"{r['method']:<20} {r['avg_time_ms']:<15.3f} "
                  f"{r['throughput']:<20.1f} {r['accuracy'] * 100:<15.2f}")

        print("\n【加速比分析】")
        python_time = results[0]['avg_time_ms']
        numpy_time = results[1]['avg_time_ms']
        fpga_time_conservative = results[2]['avg_time_ms']
        fpga_time_optimistic = results[3]['avg_time_ms']

        print(f"FPGA(保守) vs 纯Python: {python_time / fpga_time_conservative:.0f}× 加速")
        print(f"FPGA(保守) vs NumPy: {numpy_time / fpga_time_conservative:.0f}× 加速")
        print(f"FPGA(乐观) vs NumPy: {numpy_time / fpga_time_optimistic:.0f}× 加速")

        return results


def main():
    baseline = FaceSoftwareBaseline(n_components=50)
    results = baseline.run_all_benchmarks(use_full_test=True)

    print("\n" + "=" * 70)
    print("软件基准测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()