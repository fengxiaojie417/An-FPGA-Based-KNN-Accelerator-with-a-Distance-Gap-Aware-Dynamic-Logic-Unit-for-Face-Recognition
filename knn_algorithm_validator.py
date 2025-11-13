"""
自适应K值L1-KNN算法验证器 - 人脸识别应用
数据集：Olivetti Faces (AT&T人脸数据库)
"""
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
from collections import Counter
import warnings

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
warnings.filterwarnings('ignore')

class AdaptiveFaceKNNValidator:
    def __init__(self, k_range=(3, 15), n_components=50):
        """
        初始化自适应人脸识别KNN验证器
        Args:
            k_range: K值搜索范围 (min_k, max_k)
            n_components: PCA降维后的特征数量
        """
        self.k_min, self.k_max = k_range
        self.n_components = n_components
        self.load_dataset()

    def load_dataset(self):
        """加载并预处理Olivetti人脸数据集"""
        print("="*70)
        print("加载 Olivetti Faces 人脸识别数据集")
        print("="*70)

        # 加载数据集
        faces = fetch_olivetti_faces(shuffle=True, random_state=42)
        X_raw = faces.data
        y = faces.target

        print(f"原始数据: {X_raw.shape[0]}个样本 (40人 × 10张照片/人)")
        print(f"原始特征维度: {X_raw.shape[1]} (64×64像素)")
        print(f"人数: {len(np.unique(y))}人")

        # PCA降维
        print(f"\n使用PCA降维至 {self.n_components} 维...")
        pca = PCA(n_components=self.n_components, random_state=42)
        X_pca = pca.fit_transform(X_raw)

        explained_var = np.sum(pca.explained_variance_ratio_) * 100
        print(f"保留方差: {explained_var:.2f}%")

        # 归一化到[0, 255]范围（8位量化）
        scaler = MinMaxScaler(feature_range=(0, 255))
        X_scaled = scaler.fit_transform(X_pca)
        self.X = np.round(X_scaled).astype(np.uint8)
        self.y = y

        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        print(f"\n训练集: {self.X_train.shape[0]}个样本 (40人 × 8张)")
        print(f"测试集: {self.X_test.shape[0]}个样本 (40人 × 2张)")
        print(f"特征维度: {self.X_train.shape[1]}")
        print(f"数据类型: {self.X_train.dtype}, 范围: [{self.X_train.min()}, {self.X_train.max()}]")

    def l1_distance(self, x1, x2):
        """计算L1距离（曼哈顿距离）"""
        return np.sum(np.abs(x1.astype(np.int16) - x2.astype(np.int16)))

    def adaptive_knn_predict(self, query_point, return_details=False):
        """
        自适应K值预测
        """
        # 计算所有训练样本的距离
        distances = []
        for i in range(len(self.X_train)):
            dist = self.l1_distance(query_point, self.X_train[i])
            distances.append((dist, self.y_train[i]))

        distances.sort(key=lambda x: x[0])

        # 自适应K值选择算法
        best_k = self.k_min
        max_confidence = -1
        best_prediction = None
        k_confidences = []

        for k in range(self.k_min, min(self.k_max + 1, len(distances))):
            k_neighbors = distances[:k]
            k_labels = [label for _, label in k_neighbors]
            k_dists = [dist for dist, _ in k_neighbors]

            # 统计投票
            label_counts = Counter(k_labels)
            majority_class, majority_count = label_counts.most_common(1)[0]

            # 计算置信度
            vote_ratio = majority_count / k

            # 距离权重
            weights = [1.0 / (1.0 + d) for d in k_dists]
            weighted_votes = sum([weights[i] for i in range(k) if k_labels[i] == majority_class])
            total_weights = sum(weights)
            distance_weight = weighted_votes / (total_weights + 1e-6)

            # 决策边界间隙
            dist_in_class_max = 0
            dist_out_class_min = float('inf')
            found_out_class = False

            for i in range(k):
                if k_labels[i] == majority_class:
                    if k_dists[i] > dist_in_class_max:
                        dist_in_class_max = k_dists[i]
                else:
                    found_out_class = True
                    if k_dists[i] < dist_out_class_min:
                        dist_out_class_min = k_dists[i]

            gap_bonus = 1.0
            if not found_out_class:
                gap_bonus = 2.0
            elif dist_out_class_min > dist_in_class_max:
                gap = dist_out_class_min - dist_in_class_max
                gap_bonus = 1.0 + min(0.5, gap / (dist_in_class_max + 1e-6))

            # 综合置信度
            confidence = (vote_ratio + distance_weight) * gap_bonus

            k_confidences.append((k, confidence, majority_class))

            if confidence > max_confidence:
                max_confidence = confidence
                best_k = k
                best_prediction = majority_class

        # 如果置信度极低，使用最小K
        if max_confidence < 0.1:
            for k, conf, pred in k_confidences:
                if k == self.k_min:
                    best_prediction = pred
                    best_k = self.k_min
                    max_confidence = conf
                    break

        if return_details:
            return best_prediction, best_k, max_confidence, k_confidences, distances[:best_k]
        return best_prediction, best_k, max_confidence

    def evaluate(self, verbose=True):
        """评估自适应KNN性能"""
        if verbose:
            print("\n" + "="*70)
            print("自适应K值L1-KNN 人脸识别性能评估")
            print("="*70)

        predictions = []
        selected_ks = []
        confidences = []

        start_time = time.time()

        for i, test_point in enumerate(self.X_test):
            pred, k_used, conf = self.adaptive_knn_predict(test_point)
            predictions.append(pred)
            selected_ks.append(k_used)
            confidences.append(conf)

            if verbose and i < 5:
                match = "✓" if pred == self.y_test[i] else "✗"
                print(f"样本{i+1} {match}: 预测=人{pred}, 实际=人{self.y_test[i]}, "
                      f"K={k_used}, 置信度={conf:.3f}")

        total_time = time.time() - start_time

        predictions = np.array(predictions)
        accuracy = np.mean(predictions == self.y_test)
        avg_k = np.mean(selected_ks)
        avg_confidence = np.mean(confidences)

        # Top-3准确率
        top3_correct = 0
        for i, test_point in enumerate(self.X_test):
            _, _, _, k_confs, _ = self.adaptive_knn_predict(test_point, return_details=True)
            sorted_by_conf = sorted(k_confs, key=lambda x: x[1], reverse=True)
            top3_labels = []
            for _, _, pred_label in sorted_by_conf:
                if pred_label not in top3_labels:
                    top3_labels.append(pred_label)
                if len(top3_labels) >= 3:
                    break
            if self.y_test[i] in top3_labels:
                top3_correct += 1
        top3_accuracy = top3_correct / len(self.y_test)

        if verbose:
            print(f"\n【性能指标】")
            print(f"Top-1 准确率: {accuracy*100:.2f}%")
            print(f"Top-3 准确率: {top3_accuracy*100:.2f}%")
            print(f"平均使用K值: {avg_k:.2f}")
            print(f"平均置信度: {avg_confidence:.3f}")
            print(f"总预测时间: {total_time:.3f}秒")
            print(f"单人脸识别时间: {total_time/len(self.X_test)*1000:.2f}ms")

        return {
            'accuracy': accuracy,
            'top3_accuracy': top3_accuracy,
            'avg_k': avg_k,
            'avg_confidence': avg_confidence,
            'k_distribution': selected_ks,
            'confidences': confidences,
            'predictions': predictions
        }

    def compare_with_fixed_k(self):
        """对比自适应K与固定K的性能"""
        print("\n" + "="*70)
        print("自适应K vs 固定K 人脸识别性能对比")
        print("="*70)

        adaptive_results = self.evaluate()

        # 测试多个固定K值
        fixed_k_results = {}
        test_k_values = [3, 5, 7, 9, 11]

        for fixed_k in test_k_values:
            correct = 0
            for test_point, true_label in zip(self.X_test, self.y_test):
                distances = []
                for train_point, train_label in zip(self.X_train, self.y_train):
                    dist = self.l1_distance(test_point, train_point)
                    distances.append((dist, train_label))

                distances.sort(key=lambda x: x[0])
                k_labels = [label for _, label in distances[:fixed_k]]
                pred = Counter(k_labels).most_common(1)[0][0]

                if pred == true_label:
                    correct += 1

            accuracy = correct / len(self.X_test)
            fixed_k_results[fixed_k] = accuracy
            print(f"固定K={fixed_k:2d} 准确率: {accuracy*100:.2f}%")

        print(f"\n自适应K 准确率: {adaptive_results['accuracy']*100:.2f}%")
        best_fixed_k = max(fixed_k_results, key=fixed_k_results.get)
        best_fixed_acc = fixed_k_results[best_fixed_k]
        improvement = (adaptive_results['accuracy'] - best_fixed_acc) * 100
        print(f"最佳固定K={best_fixed_k} 准确率: {best_fixed_acc*100:.2f}%")
        print(f"相对提升: {improvement:+.2f}%")

        return adaptive_results, fixed_k_results

    def visualize_results(self, adaptive_results, fixed_k_results):
        """可视化分析结果"""
        fig = plt.figure(figsize=(15, 10))

        # 1. K值分布直方图
        ax1 = plt.subplot(2, 3, 1)
        plt.hist(adaptive_results['k_distribution'],
                 bins=range(self.k_min, self.k_max+2),
                 edgecolor='black', alpha=0.7, color='steelblue')
        plt.xlabel('Selected K' if not USE_CHINESE else '选择的K值', fontsize=11)
        plt.ylabel('Frequency' if not USE_CHINESE else '频次', fontsize=11)
        plt.title('Adaptive K Distribution' if not USE_CHINESE else '自适应K值分布 (人脸识别)',
                  fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # 2. 置信度分布
        ax2 = plt.subplot(2, 3, 2)
        plt.hist(adaptive_results['confidences'], bins=30,
                 edgecolor='black', alpha=0.7, color='green')
        plt.xlabel('Confidence Score' if not USE_CHINESE else '置信度分数', fontsize=11)
        plt.ylabel('Frequency' if not USE_CHINESE else '频次', fontsize=11)
        plt.title('Confidence Distribution' if not USE_CHINESE else '预测置信度分布',
                  fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # 3. 固定K vs 自适应K准确率对比
        ax3 = plt.subplot(2, 3, 3)
        k_vals = sorted(fixed_k_results.keys())
        fixed_accs = [fixed_k_results[k]*100 for k in k_vals]
        plt.plot(k_vals, fixed_accs, 'o-', linewidth=2, markersize=8,
                 label='Fixed K' if not USE_CHINESE else '固定K', color='orange')
        plt.axhline(y=adaptive_results['accuracy']*100, color='red',
                    linestyle='--', linewidth=2,
                    label='Adaptive K' if not USE_CHINESE else '自适应K')
        plt.xlabel('K Value' if not USE_CHINESE else 'K值', fontsize=11)
        plt.ylabel('Accuracy (%)' if not USE_CHINESE else '准确率 (%)', fontsize=11)
        plt.title('Fixed K vs Adaptive K' if not USE_CHINESE else '固定K vs 自适应K',
                  fontsize=12, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # 4. 混淆矩阵
        ax4 = plt.subplot(2, 3, 4)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.y_test, adaptive_results['predictions'])
        cm_subset = cm[:10, :10]
        im = plt.imshow(cm_subset, cmap='Blues', aspect='auto')
        plt.colorbar(im, ax=ax4)
        plt.xlabel('Predicted ID' if not USE_CHINESE else '预测人员ID', fontsize=11)
        plt.ylabel('True ID' if not USE_CHINESE else '真实人员ID', fontsize=11)
        plt.title('Confusion Matrix (Top 10)' if not USE_CHINESE else '混淆矩阵 (前10人)',
                  fontsize=12, fontweight='bold')

        # 5. 置信度 vs 准确性
        ax5 = plt.subplot(2, 3, 5)
        correct = (adaptive_results['predictions'] == self.y_test).astype(int)
        colors = ['green' if c else 'red' for c in correct]
        plt.scatter(range(len(correct)), adaptive_results['confidences'],
                    c=colors, alpha=0.6, s=50)
        plt.xlabel('Test Sample Index' if not USE_CHINESE else '测试样本索引', fontsize=11)
        plt.ylabel('Confidence' if not USE_CHINESE else '置信度', fontsize=11)
        plt.title('Confidence vs Correctness' if not USE_CHINESE else '置信度 vs 分类正确性',
                  fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # 6. K值 vs 准确性
        ax6 = plt.subplot(2, 3, 6)
        k_acc = {}
        for k, pred, true in zip(adaptive_results['k_distribution'],
                                   adaptive_results['predictions'],
                                   self.y_test):
            if k not in k_acc:
                k_acc[k] = {'correct': 0, 'total': 0}
            k_acc[k]['total'] += 1
            if pred == true:
                k_acc[k]['correct'] += 1

        k_vals_used = sorted(k_acc.keys())
        k_accs = [k_acc[k]['correct']/k_acc[k]['total']*100 for k in k_vals_used]
        k_counts = [k_acc[k]['total'] for k in k_vals_used]

        plt.scatter(k_vals_used, k_accs, s=[c*10 for c in k_counts],
                    alpha=0.6, color='purple')
        plt.xlabel('K Value Used' if not USE_CHINESE else '使用的K值', fontsize=11)
        plt.ylabel('Accuracy (%)' if not USE_CHINESE else '该K值下的准确率 (%)', fontsize=11)
        title_text = 'Accuracy by K (bubble size = usage count)' if not USE_CHINESE else '不同K值的准确率 (气泡大小=使用次数)'
        plt.title(title_text, fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = 'adaptive_face_knn_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n分析图表已保存: {filename}")

    def export_hardware_dataset(self, output_file='face_hardware_dataset.txt'):
        """导出用于硬件测试的人脸数据集"""
        print(f"\n导出硬件测试数据集: {output_file}")

        with open(output_file, 'w') as f:
            f.write("# Adaptive K-NN Face Recognition Hardware Dataset\n")
            f.write(f"# Train samples: {self.X_train.shape[0]}\n")
            f.write(f"# Test samples: {self.X_test.shape[0]}\n")
            f.write(f"# Features: {self.X_train.shape[1]}\n")
            f.write(f"# Persons: {len(np.unique(self.y_train))}\n")
            f.write(f"# K_range: [{self.k_min}, {self.k_max}]\n")
            f.write(f"# Data format: feature1 feature2 ... featureN person_id\n")
            f.write("#\n")

            f.write("# TRAINING SET\n")
            for i in range(len(self.X_train)):
                features = ' '.join(map(str, self.X_train[i]))
                f.write(f"{features} {self.y_train[i]}\n")

            f.write("# TEST SET\n")
            for i in range(len(self.X_test)):
                features = ' '.join(map(str, self.X_test[i]))
                f.write(f"{features} {self.y_test[i]}\n")

        print(f"导出完成! 训练集: {len(self.X_train)}张, 测试集: {len(self.X_test)}张")

    def generate_report(self):
        """生成完整的算法验证报告"""
        print("\n" + "="*70)
        print("自适应K值L1-KNN 人脸识别系统 - 算法验证报告")
        print("="*70)

        adaptive_results, fixed_k_results = self.compare_with_fixed_k()
        self.visualize_results(adaptive_results, fixed_k_results)
        self.export_hardware_dataset()

        print("\n【数据集信息】")
        print(f"✓ 人脸样本总数: {len(self.X) // 10} 人 × 10张照片/人")
        print(f"✓ 训练样本: {len(self.X_train)} (每人8张)")
        print(f"✓ 测试样本: {len(self.X_test)} (每人2张)")
        print(f"✓ 特征维度: {self.X_train.shape[1]} (PCA降维后)")
        print(f"✓ 数据位宽: 8-bit")

        return adaptive_results


def main():
    print("="*70)
    print("自适应K值 L1-KNN 人脸识别算法验证")
    print("="*70)

    validator = AdaptiveFaceKNNValidator(k_range=(3, 15), n_components=50)
    results = validator.generate_report()

    print("\n" + "="*70)
    print("验证完成! 所有结果已保存.")
    print("生成文件:")
    print("  - adaptive_face_knn_analysis.png (分析图表)")
    print("  - face_hardware_dataset.txt (硬件测试数据)")
    print("="*70)


if __name__ == "__main__":
    main()