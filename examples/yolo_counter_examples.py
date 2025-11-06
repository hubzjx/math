"""
Example usage of YOLO Sliding Counter
示例：YOLO滑动窗口计数器使用方法

This script demonstrates how to use the enhanced YOLO Sliding Counter
for object detection and counting.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolo_sliding_counter import YOLOSlidingCounter


def example_1_basic_counting():
    """示例1：基本的目标计数"""
    print("\n" + "="*60)
    print("示例1：基本的目标计数")
    print("="*60)
    
    # 配置参数
    model_path = "yolov8n.pt"  # 替换为实际的模型路径
    image_path = "test_image.jpg"  # 替换为实际的图片路径
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"⚠️  模型文件不存在: {model_path}")
        print("请先下载YOLO模型：")
        print("  pip install ultralytics")
        print("  python -c 'from ultralytics import YOLO; YOLO(\"yolov8n.pt\")'")
        return
    
    if not os.path.exists(image_path):
        print(f"⚠️  图片文件不存在: {image_path}")
        print("请准备一张测试图片")
        return
    
    # 初始化计数器
    counter = YOLOSlidingCounter(
        model_path=model_path,
        confidence_threshold=0.25,
        nms_threshold=0.45,
        window_size=(640, 640),
        overlap_ratio=0.2,
        output_dir="output/yolo_results"
    )
    
    # 执行计数
    result = counter.count_kernels_with_yolo(
        image_path=image_path,
        save_intermediate=True,
        save_visualizations=True
    )
    
    # 显示结果
    print("\n✅ 处理完成！")
    print(f"检测到的目标数量: {result.total_detections}")
    print(f"处理时间: {result.processing_time:.2f}秒")
    print(f"图片尺寸: {result.image_size}")
    print(f"处理的窗口数: {result.window_count}")
    
    if result.result_image_path:
        print(f"\n📊 结果图片已保存: {result.result_image_path}")


def example_2_roi_detection():
    """示例2：指定ROI区域的检测"""
    print("\n" + "="*60)
    print("示例2：指定ROI区域的检测")
    print("="*60)
    
    model_path = "yolov8n.pt"
    image_path = "large_image.jpg"  # 大尺寸图片
    
    if not os.path.exists(model_path) or not os.path.exists(image_path):
        print("⚠️  请准备模型和图片文件")
        return
    
    # 初始化计数器
    counter = YOLOSlidingCounter(
        model_path=model_path,
        output_dir="output/roi_results"
    )
    
    # 定义感兴趣区域 (x1, y1, x2, y2)
    roi = (100, 100, 1000, 800)
    
    # 只在ROI区域进行检测
    result = counter.count_kernels_with_yolo(
        image_path=image_path,
        roi=roi,  # 指定ROI
        save_intermediate=True,
        save_visualizations=True
    )
    
    print("\n✅ ROI检测完成！")
    print(f"ROI区域: {roi}")
    print(f"检测数量: {result.total_detections}")
    
    if result.roi_image_path:
        print(f"\n📷 ROI标注图: {result.roi_image_path}")
    if result.cropped_image_path:
        print(f"📷 裁剪图片: {result.cropped_image_path}")
    if result.result_image_path:
        print(f"📷 结果图片: {result.result_image_path}")


def example_3_batch_processing():
    """示例3：批量处理多张图片"""
    print("\n" + "="*60)
    print("示例3：批量处理多张图片")
    print("="*60)
    
    model_path = "yolov8n.pt"
    
    # 准备图片列表
    image_paths = [
        "image1.jpg",
        "image2.jpg",
        "image3.jpg",
        "image4.jpg",
        "image5.jpg"
    ]
    
    # 过滤存在的图片
    existing_images = [p for p in image_paths if os.path.exists(p)]
    
    if not os.path.exists(model_path):
        print("⚠️  模型文件不存在")
        return
    
    if not existing_images:
        print("⚠️  没有找到可处理的图片")
        return
    
    print(f"准备处理 {len(existing_images)} 张图片...")
    
    # 初始化计数器
    counter = YOLOSlidingCounter(
        model_path=model_path,
        output_dir="output/batch_results"
    )
    
    # 批量处理
    results = counter.batch_count(
        existing_images,
        save_intermediate=True,
        save_visualizations=True
    )
    
    # 统计结果
    print("\n✅ 批量处理完成！")
    print(f"成功处理: {len(results)}/{len(existing_images)} 张图片")
    
    total_detections = sum(r.total_detections for r in results)
    total_time = sum(r.processing_time for r in results)
    
    print(f"\n📊 统计信息:")
    print(f"  总检测数: {total_detections}")
    print(f"  总处理时间: {total_time:.2f}秒")
    print(f"  平均处理时间: {total_time/len(results):.2f}秒/图")
    
    # 显示每张图片的结果
    print("\n详细结果:")
    for i, (path, result) in enumerate(zip(existing_images, results), 1):
        print(f"  {i}. {os.path.basename(path)}: {result.total_detections} 个目标")


def example_4_custom_parameters():
    """示例4：使用自定义参数"""
    print("\n" + "="*60)
    print("示例4：使用自定义参数")
    print("="*60)
    
    model_path = "yolov8n.pt"
    image_path = "test_image.jpg"
    
    if not os.path.exists(model_path) or not os.path.exists(image_path):
        print("⚠️  请准备模型和图片文件")
        return
    
    # 使用自定义参数初始化
    counter = YOLOSlidingCounter(
        model_path=model_path,
        confidence_threshold=0.5,      # 更高的置信度阈值
        nms_threshold=0.3,              # 更严格的NMS
        window_size=(512, 512),         # 较小的窗口
        overlap_ratio=0.3,              # 更大的重叠率
        output_dir="output/custom_results"
    )
    
    result = counter.count_kernels_with_yolo(
        image_path=image_path,
        save_intermediate=True,
        save_visualizations=True
    )
    
    print("\n✅ 自定义参数处理完成！")
    print(f"参数配置:")
    print(f"  置信度阈值: {counter.confidence_threshold}")
    print(f"  NMS阈值: {counter.nms_threshold}")
    print(f"  窗口大小: {counter.window_size}")
    print(f"  重叠率: {counter.overlap_ratio}")
    print(f"\n结果:")
    print(f"  检测数量: {result.total_detections}")
    print(f"  处理时间: {result.processing_time:.2f}秒")


def example_5_detection_details():
    """示例5：获取详细的检测信息"""
    print("\n" + "="*60)
    print("示例5：获取详细的检测信息")
    print("="*60)
    
    model_path = "yolov8n.pt"
    image_path = "test_image.jpg"
    
    if not os.path.exists(model_path) or not os.path.exists(image_path):
        print("⚠️  请准备模型和图片文件")
        return
    
    counter = YOLOSlidingCounter(
        model_path=model_path,
        output_dir="output/detailed_results"
    )
    
    result = counter.count_kernels_with_yolo(
        image_path=image_path,
        save_intermediate=True,
        save_visualizations=True
    )
    
    print("\n✅ 处理完成！")
    print(f"\n📊 检测统计:")
    print(f"  总检测数: {result.total_detections}")
    print(f"  图片尺寸: {result.image_size[0]}x{result.image_size[1]}")
    print(f"  处理窗口数: {result.window_count}")
    print(f"  处理时间: {result.processing_time:.2f}秒")
    print(f"  平均每窗口时间: {result.processing_time/result.window_count:.3f}秒")
    
    # 显示检测框信息
    if result.detection_boxes:
        print(f"\n🎯 前5个检测框:")
        for i, (box, score, cls) in enumerate(
            zip(result.detection_boxes[:5], 
                result.detection_scores[:5],
                result.detection_classes[:5]),
            1
        ):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            print(f"  {i}. 位置:({x1},{y1}) 尺寸:{w}x{h} "
                  f"置信度:{score:.3f} 类别:{cls}")
    
    # 显示输出文件
    print(f"\n📁 输出文件:")
    if result.roi_image_path:
        print(f"  ROI标注: {result.roi_image_path}")
    if result.cropped_image_path:
        print(f"  裁剪图片: {result.cropped_image_path}")
    if result.result_image_path:
        print(f"  结果图片: {result.result_image_path}")


def main():
    """运行所有示例"""
    print("\n" + "="*60)
    print("YOLO Sliding Counter - 使用示例")
    print("="*60)
    
    print("\n可用的示例:")
    print("  1. 基本的目标计数")
    print("  2. 指定ROI区域的检测")
    print("  3. 批量处理多张图片")
    print("  4. 使用自定义参数")
    print("  5. 获取详细的检测信息")
    print("  all. 运行所有示例")
    
    choice = input("\n请选择要运行的示例 (1-5 或 all): ").strip()
    
    examples = {
        '1': example_1_basic_counting,
        '2': example_2_roi_detection,
        '3': example_3_batch_processing,
        '4': example_4_custom_parameters,
        '5': example_5_detection_details,
    }
    
    if choice == 'all':
        for func in examples.values():
            try:
                func()
            except Exception as e:
                print(f"\n❌ 错误: {str(e)}")
    elif choice in examples:
        try:
            examples[choice]()
        except Exception as e:
            print(f"\n❌ 错误: {str(e)}")
    else:
        print("\n⚠️  无效的选择")
    
    print("\n" + "="*60)
    print("示例运行结束")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
