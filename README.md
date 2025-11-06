# DSP-Free L1-KNN 并行加速器 - 算法验证工具集

## 项目概述

这是一个用于验证和测试 L1-norm KNN 算法的完整工具集，为后续的 FPGA 硬件实现提供算法基础和性能基准。

**核心创新点:**
- ✅ 使用曼哈顿距离（L1-norm）替代欧氏距离（L2-norm）
- ✅ 完全不使用 DSP，仅使用 LUT 和加法器
- ✅ 适合 FPGA 并行化实现
- ✅ 提供完整的算法验证和硬件准备流程

## 文件结构

```
.
├── run_all_tests.py                    # 主控脚本（一键运行）
├── knn_algorithm_validator.py          # 算法精度验证
├── software_baseline_benchmark.py      # 软件性能基准测试
├── hardware_testbench_generator.py     # 硬件测试文件生成器
└── README.md                            # 本文档
```

## 快速开始

### 环境要求

- Python 3.7+
- 必需库：
  ```bash
  pip install numpy scikit-learn matplotlib
  ```

### 一键运行（推荐）

```bash
python run_all_tests.py
```

这个脚本会自动按顺序执行：
1. 依赖检查
2. 算法精度验证
3. 软件性能基准测试
4. 硬件测试文件生成
5. 生成最终报告

### 分步运行

如果需要单独运行某个步骤：

```bash
# 步骤1: 算法验证
python knn_algorithm_validator.py

# 步骤2: 软件基准测试
python software_baseline_benchmark.py

# 步骤3: 生成硬件测试文件
python hardware_testbench_generator.py
```

## 详细说明

### 1. knn_algorithm_validator.py

**功能：**
- 验证 L1-KNN 相比 L2-KNN 的精度损失
- 测试不同 K 值的影响
- 生成精度对比图表
- 导出硬件测试数据集

**输出文件：**
- `knn_k_comparison_iris.png` - Iris 数据集精度对比图
- `knn_k_comparison_digits.png` - Digits 数据集精度对比图
- `iris_hardware_dataset.txt` - Iris 数据集（8位量化）
- `digits_hardware_dataset.txt` - Digits 数据集（8位量化）

**关键参数：**
```python
dataset_name = 'iris' 或 'digits'
k = 5  # KNN的K值
```

**预期结果示例：**
```
Iris数据集：
  L2-norm 准确率: 97.78%
  L1-norm 准确率: 95.56%
  精度损失: 2.22%  ✓ 可接受
```

### 2. software_baseline_benchmark.py

**功能：**
- 完全模拟硬件实现流程的软件版本
- 测量单次推理的精确时间
- 提供硬件加速的性能基准
- 估算硬件资源需求

**关键特性：**
- 使用 8-bit 无符号整数（uint8）
- 逐特征计算 L1 距离（模拟硬件）
- 实现 Top-K 排序器（模拟移位寄存器）
- 简单投票机制（模拟硬件计数器）

**输出示例：**
```
软件基准测试结果:
  平均推理时间: 1.0500 ms
  100x 加速目标: 10.50 μs
  
硬件实现估算:
  距离计算单元数: 4 (并行)
  Top-K 排序器大小: 5
  BRAM 需求: ~420 bytes
```

### 3. hardware_testbench_generator.py

**功能：**
- 生成 Verilog testbench 测试向量
- 生成 BRAM 初始化文件
- 生成 Xilinx COE 文件
- 生成 ARM 端 C 头文件

**输出文件：**

#### Iris 数据集：
- `iris_train_data_bram.mem` - MEM 格式 BRAM 初始化
- `iris_train_data.coe` - Xilinx COE 格式
- `iris_test_vectors.mem` - Verilog testbench 测试向量
- `iris_knn_data.h` - C 头文件（ARM 测试）

#### Digits 数据集：
- `digits_train_data_bram.mem`
- `digits_train_data.coe`
- `digits_test_vectors.mem`
- `digits_knn_data.h`

**文件格式说明：**

**MEM 文件示例：**
```
@0000 33 54 96 E9 00
@0001 31 58 95 E8 00
```
格式：地址 特征1 特征2 ... 特征N 标签

**COE 文件示例：**
```
memory_initialization_radix=16;
memory_initialization_vector=
335496E900,
315895E800;
```

**测试向量示例：**
```verilog
query_vector = {8'h33, 8'h54, 8'h96, 8'hE9};
expected_label = 0;
#10 start = 1;
#10 start = 0;
wait(done);
```

## 硬件实现指南

### 数据格式

所有数据均使用 **8-bit 无符号整数** (uint8)：
- 特征值范围：[0, 255]
- 标签范围：[0, 15]（支持最多16个类别）

### 推荐的 Verilog 模块设计

根据生成的测试文件，你的 Verilog 设计应包含：

#### 1. l1_distance_unit.v
```verilog
module l1_distance_unit(
    input [7:0] feature_q,
    input [7:0] feature_t,
    output [8:0] distance_delta
);
    assign distance_delta = (feature_q > feature_t) ? 
                           (feature_q - feature_t) : 
                           (feature_t - feature_q);
endmodule
```

#### 2. BRAM 接口
使用生成的 COE 文件初始化 Block Memory：
- 在 Vivado 中添加 "Block Memory Generator" IP
- 选择 "Load Init File" → 选择 `*_train_data.coe`
- 数据宽度 = (特征数 + 1) × 8 bits

#### 3. Testbench
直接使用生成的 `*_test_vectors.mem`：
```verilog
`include "iris_test_vectors.mem"
```

### 性能目标

基于软件基准测试结果：

| 指标 | 软件基线 | 硬件目标 | 加速比 |
|------|---------|---------|--------|
| 延迟 | ~1 ms   | <10 μs  | 100x   |
| 吞吐量 | ~1000 inferences/s | >100,000 inferences/s | 100x |
| DSP 使用 | N/A | **0** | - |

### 资源估算（Zynq-7020）

**Iris 数据集 (4特征):**
- LUT: ~1000
- FF: ~500
- BRAM: 1-2 个
- DSP: **0**

**Digits 数据集 (64特征):**
- LUT: ~3000
- FF: ~1500
- BRAM: 4-6 个
- DSP: **0**

## 论文撰写建议

基于这些验证结果，你可以撰写：

### 标题建议
"A DSP-Free, Parallel L1-Norm KNN Accelerator for Real-Time Edge Classification on FPGA"

### 关键实验数据（来自本工具）

1. **算法验证章节：**
   - 图：L1 vs L2 精度对比（来自步骤1）
   - 表：不同 K 值的性能对比

2. **性能评估章节：**
   - 软件基线数据（来自步骤2）
   - 硬件加速比计算
   - 资源使用报告（来自 Vivado）

3. **能效分析章节：**
   - 功耗测量（使用 Vivado Power Analyzer）
   - 能效比：Inferences/Joule

## 常见问题

### Q1: 为什么精度会有损失？
A: L1 距离和 L2 距离的几何意义不同。L2 考虑了特征间的相关性，而 L1 将所有维度等权重处理。但在很多实际应用中，这种差异很小（<5%）。

### Q2: 如何选择合适的 K 值？
A: 运行 `knn_algorithm_validator.py` 会自动测试 K=1 到 K=20。通常 K=5 是一个很好的平衡点。

### Q3: 数据量化到 8-bit 会影响精度吗？
A: 本工具已经使用 8-bit 量化。实验表明，在归一化后，8-bit 量化对精度影响很小（<1%）。

### Q4: 如何扩展到自己的数据集？
A: 修改 `knn_algorithm_validator.py`：
```python
# 加载你的数据
X = your_data  # shape: (n_samples, n_features)
y = your_labels  # shape: (n_samples,)

# 归一化到 [0, 255]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 255))
X_scaled = np.round(scaler.fit_transform(X)).astype(np.uint8)
```

### Q5: 生成的文件如何在 Vivado 中使用？
A: 
1. COE 文件：Block Memory Generator → COE File
2. MEM 文件：仿真时使用 `$readmemh()`
3. 测试向量：直接 `include` 到 testbench

## 下一步

完成算法验证后，请参考原始文档的：
- **C. 硬件实现步骤** - Verilog 代码编写
- **D. 硬件平台与实现** - Vivado 集成流程

## 技术支持

如果遇到问题：
1. 检查 Python 版本和依赖库版本
2. 确认数据集文件已正确生成
3. 查看 `final_report.txt` 获取详细信息

## 许可证

本工具集用于学术研究和教育目的。

---

**祝你的 FPGA 项目顺利！🚀**