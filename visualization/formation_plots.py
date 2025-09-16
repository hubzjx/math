"""
Visualization module for UAV formation positioning

This module provides visualization capabilities for UAV formations,
positioning algorithms, and adjustment strategies.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple
import math
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.positioning_models import UAVPosition, AngleMeasurement, CircularFormationModel, ConeFormationModel

class FormationVisualizer:
    """Main class for visualizing UAV formations and positioning algorithms"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        self.figsize = figsize
        plt.style.use('default')
    
    def plot_circular_formation(self, positions: List[UAVPosition], 
                              ideal_positions: Optional[List[UAVPosition]] = None,
                              title: str = "UAV Circular Formation"):
        """Plot circular formation with current and ideal positions"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot ideal positions if provided
        if ideal_positions:
            ideal_x = [pos.x for pos in ideal_positions]
            ideal_y = [pos.y for pos in ideal_positions]
            ax.scatter(ideal_x[1:], ideal_y[1:], c='lightblue', s=100, alpha=0.7, 
                      label='Ideal Positions', marker='o')
            ax.scatter(ideal_x[0], ideal_y[0], c='lightblue', s=150, alpha=0.7, 
                      marker='s')  # Center position
        
        # Plot current positions
        current_x = [pos.x for pos in positions]
        current_y = [pos.y for pos in positions]
        
        # Separate center UAV from circle UAVs
        center_pos = [pos for pos in positions if pos.uav_id == "FY00"]
        circle_pos = [pos for pos in positions if pos.uav_id != "FY00"]
        
        if center_pos:
            ax.scatter(center_pos[0].x, center_pos[0].y, c='red', s=200, 
                      label='Center UAV (FY00)', marker='s')
        
        if circle_pos:
            circle_x = [pos.x for pos in circle_pos]
            circle_y = [pos.y for pos in circle_pos]
            ax.scatter(circle_x, circle_y, c='blue', s=120, 
                      label='Circle UAVs', marker='o')
        
        # Add UAV labels
        for pos in positions:
            ax.annotate(pos.uav_id, (pos.x, pos.y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
        
        # Draw ideal circle
        if ideal_positions and len(ideal_positions) > 1:
            # Calculate radius from ideal positions
            center = ideal_positions[0]
            radius = center.distance_to(ideal_positions[1])
            circle = plt.Circle((center.x, center.y), radius, 
                              fill=False, color='gray', linestyle='--', alpha=0.5)
            ax.add_patch(circle)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        return fig, ax
    
    def plot_cone_formation(self, positions: List[UAVPosition],
                           ideal_positions: Optional[List[UAVPosition]] = None,
                           title: str = "UAV Cone Formation"):
        """Plot cone formation"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot ideal positions if provided
        if ideal_positions:
            ideal_x = [pos.x for pos in ideal_positions]
            ideal_y = [pos.y for pos in ideal_positions]
            ax.scatter(ideal_x, ideal_y, c='lightblue', s=100, alpha=0.7, 
                      label='Ideal Positions', marker='o')
        
        # Plot current positions
        current_x = [pos.x for pos in positions]
        current_y = [pos.y for pos in positions]
        ax.scatter(current_x, current_y, c='blue', s=120, 
                  label='Current Positions', marker='o')
        
        # Add UAV labels
        for pos in positions:
            ax.annotate(pos.uav_id, (pos.x, pos.y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        return fig, ax
    
    def plot_angle_measurements(self, receiver: UAVPosition, 
                              transmitters: List[UAVPosition],
                              measurements: List[AngleMeasurement],
                              title: str = "Angle Measurements Visualization"):
        """Visualize angle measurements between transmitters as seen from receiver"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot receiver
        ax.scatter(receiver.x, receiver.y, c='green', s=200, 
                  label='Receiver', marker='^')
        ax.annotate(receiver.uav_id, (receiver.x, receiver.y), 
                   xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
        
        # Plot transmitters
        trans_x = [pos.x for pos in transmitters]
        trans_y = [pos.y for pos in transmitters]
        ax.scatter(trans_x, trans_y, c='red', s=150, 
                  label='Transmitters', marker='s')
        
        # Add transmitter labels
        for pos in transmitters:
            ax.annotate(pos.uav_id, (pos.x, pos.y), 
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        # Draw lines from receiver to transmitters and show angles
        colors = ['orange', 'purple', 'brown', 'pink', 'gray']
        for i, trans in enumerate(transmitters):
            ax.plot([receiver.x, trans.x], [receiver.y, trans.y], 
                   color=colors[i % len(colors)], linestyle='-', alpha=0.7)
        
        # Show angle measurements
        for i, measurement in enumerate(measurements):
            # Find the corresponding transmitters
            trans1 = next((t for t in transmitters if t.uav_id == measurement.source1_id), None)
            trans2 = next((t for t in transmitters if t.uav_id == measurement.source2_id), None)
            
            if trans1 and trans2:
                # Draw angle arc
                angle1 = receiver.angle_to(trans1)
                angle2 = receiver.angle_to(trans2)
                
                # Create angle arc
                arc_radius = 30  # Visual radius for the arc
                arc_angles = np.linspace(min(angle1, angle2), max(angle1, angle2), 20)
                arc_x = receiver.x + arc_radius * np.cos(arc_angles)
                arc_y = receiver.y + arc_radius * np.sin(arc_angles)
                ax.plot(arc_x, arc_y, color=colors[i % len(colors)], linewidth=2)
                
                # Add angle label
                mid_angle = (angle1 + angle2) / 2
                label_x = receiver.x + (arc_radius + 10) * np.cos(mid_angle)
                label_y = receiver.y + (arc_radius + 10) * np.sin(mid_angle)
                ax.annotate(f'{math.degrees(measurement.angle):.1f}°', 
                           (label_x, label_y), fontsize=9, ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        return fig, ax
    
    def plot_positioning_process(self, initial_positions: List[UAVPosition],
                               final_positions: List[UAVPosition],
                               adjustment_steps: Optional[List[List[UAVPosition]]] = None,
                               title: str = "UAV Position Adjustment Process"):
        """Plot the positioning adjustment process"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot initial positions
        self._plot_formation_state(axes[0], initial_positions, "Initial Positions", "red")
        
        # Plot final positions
        self._plot_formation_state(axes[1], final_positions, "Final Positions", "green")
        
        plt.tight_layout()
        return fig, axes
    
    def _plot_formation_state(self, ax, positions: List[UAVPosition], 
                            title: str, color: str):
        """Helper method to plot a single formation state"""
        x_coords = [pos.x for pos in positions]
        y_coords = [pos.y for pos in positions]
        
        # Separate center and circle UAVs
        center_pos = [pos for pos in positions if pos.uav_id == "FY00"]
        circle_pos = [pos for pos in positions if pos.uav_id != "FY00"]
        
        if center_pos:
            ax.scatter(center_pos[0].x, center_pos[0].y, c=color, s=200, 
                      marker='s', alpha=0.8, label='Center UAV')
        
        if circle_pos:
            circle_x = [pos.x for pos in circle_pos]
            circle_y = [pos.y for pos in circle_pos]
            ax.scatter(circle_x, circle_y, c=color, s=120, 
                      marker='o', alpha=0.8, label='Circle UAVs')
        
        # Add labels
        for pos in positions:
            ax.annotate(pos.uav_id, (pos.x, pos.y), xytext=(3, 3), 
                       textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

def create_mind_map_visualization():
    """Create a text-based mind map visualization"""
    mind_map = """
    UAV编队纯方位无源定位系统 思维导图
    ═══════════════════════════════════════════════════════════
    
    ┌─ 问题分析
    │  ├─ 纯方位测量原理 (Direction-Only Measurement)
    │  │  ├─ 角度信息提取
    │  │  ├─ 电磁静默要求
    │  │  └─ 方向精度分析
    │  ├─ 编队队形要求
    │  │  ├─ 圆形编队 (10架: 1中心+9圆周)
    │  │  ├─ 锥形编队 (直线间距50m)
    │  │  └─ 位置保持精度
    │  └─ 约束条件
    │     ├─ 最多3架发射机
    │     ├─ 已知编号要求
    │     └─ 实时调整能力
    │
    ├─ 数学建模
    │  ├─ 几何模型 (Geometric Models)
    │  │  ├─ 角度-位置关系
    │  │  │  ├─ 三角函数关系
    │  │  │  ├─ 坐标变换
    │  │  │  └─ 角度测量方程
    │  │  ├─ 三角定位算法
    │  │  │  ├─ 三点定位
    │  │  │  ├─ 最小二乘法
    │  │  │  └─ 几何约束优化
    │  │  └─ 误差传播分析
    │  ├─ 优化模型 (Optimization Models)
    │  │  ├─ 目标函数设计
    │  │  │  ├─ 位置误差最小化
    │  │  │  ├─ 编队形状保持
    │  │  │  └─ 多目标优化
    │  │  ├─ 约束条件
    │  │  │  ├─ 物理约束
    │  │  │  ├─ 通信约束
    │  │  │  └─ 安全约束
    │  │  └─ 求解算法
    │  │     ├─ 梯度下降法
    │  │     ├─ 粒子群算法
    │  │     └─ 遗传算法
    │  └─ 信号处理模型
    │     ├─ 角度估计算法
    │     ├─ 噪声滤波
    │     └─ 数据融合
    │
    ├─ 算法实现 (Algorithm Implementation)
    │  ├─ 圆形编队定位
    │  │  ├─ 问题1: 已知发射机定位
    │  │  │  ├─ 三角定位实现
    │  │  │  ├─ 位置解算
    │  │  │  └─ 精度评估
    │  │  ├─ 问题2: 最小发射机数量
    │  │  │  ├─ 理论分析 (3架)
    │  │  │  ├─ 实际验证
    │  │  │  └─ 鲁棒性分析
    │  │  └─ 问题3: 位置调整策略
    │  │     ├─ 迭代调整算法
    │  │     ├─ 发射机选择策略
    │  │     └─ 收敛性保证
    │  ├─ 锥形编队定位
    │  │  ├─ 线性排列特性
    │  │  ├─ 间距控制算法
    │  │  └─ 形状保持策略
    │  └─ 通用算法模块
    │     ├─ 角度计算函数
    │     ├─ 位置估计算法
    │     └─ 误差分析工具
    │
    ├─ 可视化与分析 (Visualization & Analysis)
    │  ├─ 编队状态可视化
    │  │  ├─ 实时位置显示
    │  │  ├─ 理想vs实际对比
    │  │  └─ 调整过程动画
    │  ├─ 角度测量可视化
    │  │  ├─ 测量角度显示
    │  │  ├─ 误差分布图
    │  │  └─ 精度热力图
    │  └─ 性能分析图表
    │     ├─ 收敛曲线
    │     ├─ 误差统计
    │     └─ 算法比较
    │
    └─ 验证与测试 (Validation & Testing)
       ├─ 仿真测试
       │  ├─ 理想条件测试
       │  ├─ 噪声环境测试
       │  └─ 极端情况测试
       ├─ 数据验证
       │  ├─ 表1数据处理
       │  ├─ 结果对比分析
       │  └─ 精度评估
       └─ 性能分析
          ├─ 计算效率
          ├─ 内存使用
          └─ 实时性能
    
    ═══════════════════════════════════════════════════════════
    核心技术栈: Python + NumPy + SciPy + Matplotlib + 优化算法
    """
    return mind_map

if __name__ == "__main__":
    # Test visualization functions
    print("=== UAV Formation Visualization Module ===")
    
    # Create test data
    from models.positioning_models import generate_sample_data, CircularFormationModel
    
    positions, measurements = generate_sample_data()
    model = CircularFormationModel()
    ideal_positions = model.get_ideal_positions()
    
    # Create visualizer
    visualizer = FormationVisualizer()
    
    # Test circular formation plot
    print("Creating circular formation visualization...")
    fig, ax = visualizer.plot_circular_formation(positions, ideal_positions)
    plt.savefig('/tmp/circular_formation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: /tmp/circular_formation.png")
    
    # Test angle measurements plot
    print("Creating angle measurements visualization...")
    receiver = positions[3]  # FY03
    transmitters = positions[:3]  # FY00, FY01, FY02
    fig, ax = visualizer.plot_angle_measurements(receiver, transmitters, measurements)
    plt.savefig('/tmp/angle_measurements.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: /tmp/angle_measurements.png")
    
    # Display mind map
    print("\n" + create_mind_map_visualization())