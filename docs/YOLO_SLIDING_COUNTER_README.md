# YOLO Sliding Window Counter - 完整版本

## 概述

YOLO滑动窗口计数器是一个增强版的目标检测与计数系统，基于YOLO模型实现。该系统采用滑动窗口方法处理大型图像，并提供全面的图像处理、错误处理、性能优化和详细统计功能。

## 主要特性

### 1. 图片处理增强

- **ROI标注图输出**: 可视化检测区域，使用红色边框标记
- **中间图片保存**: 保存ROI裁剪后的图片以便检查
- **最终结果可视化**: 保存带有检测框和标签的结果图片

### 2. 错误处理改进

- **图片读取/写入异常捕获**: 全面的异常处理，提供清晰的错误信息
- **YOLO模型加载失败提示**: 详细的模型加载错误诊断
- **输入参数验证**: 严格的参数验证，防止无效输入

### 3. 性能优化

- **大图片自动缩放**: 自动检测并缩放过大的图片，优化处理速度
- **批量处理进度提示**: 实时显示批处理进度
- **内存使用优化**: 高效的内存管理，适合处理多张大图

### 4. 代码质量提升

- **完善的日志系统**: 详细的调试和运行信息
- **详细的函数文档**: 每个函数都有完整的文档字符串
- **常量定义**: 所有魔法数字都定义为常量

### 5. API增强

- **丰富的返回值**: `count_kernels_with_yolo`函数返回包含所有处理结果的对象
- **详细的统计信息**: 检测数量、处理时间、窗口数量等

## 安装

### 依赖项

```bash
pip install -r requirements.txt
```

主要依赖：
- numpy>=1.21.0
- opencv-python>=4.5.0
- ultralytics>=8.0.0
- pillow>=9.0.0

### YOLO模型准备

下载预训练的YOLO模型：

```bash
# 使用ultralytics命令行工具
pip install ultralytics
yolo download model=yolov8n.pt

# 或手动下载
# 从 https://github.com/ultralytics/ultralytics 下载模型
```

## 使用方法

### 基本使用

```python
from yolo_sliding_counter import YOLOSlidingCounter

# 初始化计数器
counter = YOLOSlidingCounter(
    model_path="yolov8n.pt",
    confidence_threshold=0.25,
    nms_threshold=0.45,
    window_size=(640, 640),
    overlap_ratio=0.2,
    output_dir="output/yolo_results"
)

# 对单张图片进行计数
result = counter.count_kernels_with_yolo(
    image_path="test_image.jpg",
    roi=None,  # 可选: 指定ROI为 (x1, y1, x2, y2)
    save_intermediate=True,
    save_visualizations=True
)

# 查看结果
print(f"检测到的目标数量: {result.total_detections}")
print(f"处理时间: {result.processing_time:.2f}秒")
print(f"结果图片保存在: {result.result_image_path}")
```

### 指定ROI区域

```python
# 只处理图片的特定区域
result = counter.count_kernels_with_yolo(
    image_path="large_image.jpg",
    roi=(100, 100, 1000, 800),  # (x1, y1, x2, y2)
    save_intermediate=True,
    save_visualizations=True
)
```

### 批量处理

```python
# 批量处理多张图片
image_paths = [
    "image1.jpg",
    "image2.jpg",
    "image3.jpg"
]

results = counter.batch_count(
    image_paths,
    save_intermediate=True,
    save_visualizations=True
)

# 统计总结果
total = sum(r.total_detections for r in results)
print(f"总共检测到 {total} 个目标")
```

### 自定义参数

```python
# 使用自定义参数初始化
counter = YOLOSlidingCounter(
    model_path="yolov8n.pt",
    confidence_threshold=0.5,      # 提高置信度阈值
    nms_threshold=0.3,              # 降低NMS阈值以减少重叠
    window_size=(512, 512),         # 使用较小的窗口
    overlap_ratio=0.3,              # 增加窗口重叠
    output_dir="custom_output"      # 自定义输出目录
)
```

## API参考

### YOLOSlidingCounter类

#### 构造函数

```python
YOLOSlidingCounter(
    model_path: str,
    confidence_threshold: float = 0.25,
    nms_threshold: float = 0.45,
    window_size: Tuple[int, int] = (640, 640),
    overlap_ratio: float = 0.2,
    output_dir: str = "output"
)
```

**参数:**
- `model_path`: YOLO模型权重文件路径
- `confidence_threshold`: 检测置信度阈值 (0-1)
- `nms_threshold`: 非极大值抑制阈值 (0-1)
- `window_size`: 滑动窗口大小 (宽度, 高度)
- `overlap_ratio`: 窗口重叠率 (0-1)
- `output_dir`: 输出图片保存目录

#### count_kernels_with_yolo方法

```python
count_kernels_with_yolo(
    image_path: str,
    roi: Optional[Tuple[int, int, int, int]] = None,
    save_intermediate: bool = True,
    save_visualizations: bool = True
) -> DetectionResult
```

**参数:**
- `image_path`: 输入图片路径
- `roi`: 可选的感兴趣区域 (x1, y1, x2, y2)
- `save_intermediate`: 是否保存中间图片
- `save_visualizations`: 是否保存可视化图片

**返回:**
- `DetectionResult`: 包含所有检测结果和统计信息的对象

#### batch_count方法

```python
batch_count(
    image_paths: List[str],
    **kwargs
) -> List[DetectionResult]
```

**参数:**
- `image_paths`: 图片路径列表
- `**kwargs`: 传递给count_kernels_with_yolo的额外参数

**返回:**
- `List[DetectionResult]`: 所有图片的检测结果列表

### DetectionResult类

包含检测结果和统计信息的数据类。

**属性:**
- `total_detections`: 检测到的目标总数
- `detection_boxes`: 检测框列表 [(x1, y1, x2, y2), ...]
- `detection_scores`: 置信度分数列表
- `detection_classes`: 类别ID列表
- `processing_time`: 处理时间（秒）
- `image_size`: 图片尺寸 (宽度, 高度)
- `window_count`: 处理的窗口数量
- `roi_image_path`: ROI标注图路径（如果保存）
- `cropped_image_path`: 裁剪图路径（如果保存）
- `result_image_path`: 结果图路径（如果保存）

## 输出文件

系统会生成以下输出文件（基于原始文件名）:

1. **ROI标注图**: `{basename}_roi_annotated.jpg`
   - 显示检测区域的红色边框
   - 包含ROI尺寸信息

2. **裁剪图片**: `{basename}_cropped.jpg`
   - ROI区域的裁剪图片
   - 用于验证处理区域

3. **结果图片**: `{basename}_result.jpg`
   - 显示所有检测框和标签
   - 包含总检测数量

## 配置参数说明

### 置信度阈值 (confidence_threshold)

- **范围**: 0.0 - 1.0
- **默认值**: 0.25
- **说明**: 只保留置信度高于此值的检测结果
- **建议**: 
  - 较低值 (0.2-0.3): 检测更多目标，可能有误检
  - 较高值 (0.5-0.7): 更准确，但可能遗漏目标

### NMS阈值 (nms_threshold)

- **范围**: 0.0 - 1.0
- **默认值**: 0.45
- **说明**: 控制重叠检测框的合并
- **建议**:
  - 较低值 (0.3-0.4): 更激进的去重，减少重复
  - 较高值 (0.5-0.6): 保留更多检测框

### 窗口大小 (window_size)

- **默认值**: (640, 640)
- **说明**: 滑动窗口的尺寸
- **建议**:
  - 小目标: 使用较小窗口 (320x320, 416x416)
  - 大目标: 使用较大窗口 (640x640, 1280x1280)

### 重叠率 (overlap_ratio)

- **范围**: 0.0 - 1.0
- **默认值**: 0.2
- **说明**: 相邻窗口的重叠比例
- **建议**:
  - 较低值 (0.1-0.2): 处理速度快，可能遗漏边界目标
  - 较高值 (0.3-0.5): 更全面，但处理时间长

## 性能优化建议

### 1. 大图片处理

对于超大图片（>2048像素）:
- 系统会自动缩放
- 可以手动指定ROI来处理特定区域
- 考虑使用较大的window_size

### 2. 批量处理

处理多张图片时:
- 使用`batch_count`方法
- 系统会自动显示进度
- 出错的图片会被跳过，继续处理下一张

### 3. 内存管理

- 处理完成后图片数据会自动释放
- 批量处理时建议每次不超过100张图片
- 可以设置`save_intermediate=False`减少磁盘占用

## 故障排除

### 模型加载失败

**问题**: RuntimeError: Failed to load YOLO model

**解决方案**:
1. 确认模型文件存在且路径正确
2. 检查是否安装了ultralytics: `pip install ultralytics`
3. 尝试重新下载模型

### 图片读取失败

**问题**: ValueError: Failed to load image

**解决方案**:
1. 确认图片文件存在
2. 检查图片格式是否支持
3. 确认图片文件没有损坏

### 内存不足

**问题**: 处理大图片时内存溢出

**解决方案**:
1. 系统会自动缩放大图片
2. 可以手动指定更小的window_size
3. 减小overlap_ratio
4. 分批处理图片

## 示例代码

完整的示例代码请参考 `yolo_sliding_counter.py` 中的 `main()` 函数。

## 测试

运行单元测试:

```bash
python -m pytest tests/test_yolo_sliding_counter.py -v
```

或直接运行测试文件:

```bash
python tests/test_yolo_sliding_counter.py
```

## 向后兼容性

该版本保持与之前版本的向后兼容性:
- 核心API `count_kernels_with_yolo` 签名保持不变
- 默认参数设置合理，无需修改现有代码
- 新增功能通过可选参数提供

## 许可证

本项目遵循仓库的整体许可证。

## 贡献

欢迎提交Issue和Pull Request！

## 更新日志

### Version 2.0 (2025-11-06)

- ✅ 添加ROI可视化功能
- ✅ 实现中间图片保存
- ✅ 增强错误处理机制
- ✅ 实现自动图片缩放
- ✅ 添加批处理进度显示
- ✅ 完善日志系统
- ✅ 添加详细文档
- ✅ 实现完整的单元测试
