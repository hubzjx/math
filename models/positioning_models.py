"""
UAV Formation Flying - Core Mathematical Models

This module implements the mathematical models for UAV passive positioning
using direction-only measurements in formation flying scenarios.
"""

import numpy as np
from typing import List, Tuple, Optional
import math

class UAVPosition:
    """Represents a UAV position in 2D space"""
    
    def __init__(self, x: float, y: float, uav_id: str):
        self.x = x
        self.y = y
        self.uav_id = uav_id
    
    def distance_to(self, other: 'UAVPosition') -> float:
        """Calculate Euclidean distance to another UAV"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def angle_to(self, other: 'UAVPosition') -> float:
        """Calculate angle to another UAV from positive x-axis"""
        return math.atan2(other.y - self.y, other.x - self.x)
    
    def __repr__(self):
        return f"UAV({self.uav_id}): ({self.x:.2f}, {self.y:.2f})"

class AngleMeasurement:
    """Represents an angle measurement between two signal sources"""
    
    def __init__(self, receiver_id: str, source1_id: str, source2_id: str, angle: float):
        self.receiver_id = receiver_id
        self.source1_id = source1_id
        self.source2_id = source2_id
        self.angle = angle  # angle in radians
    
    def __repr__(self):
        return f"Angle({self.receiver_id}): {self.source1_id}-{self.source2_id} = {math.degrees(self.angle):.2f}°"

class CircularFormationModel:
    """Mathematical model for circular UAV formation positioning"""
    
    def __init__(self, radius: float = 100.0):
        self.radius = radius
        self.center = UAVPosition(0, 0, "FY00")
        self.uavs = {}
        
    def add_uav(self, uav: UAVPosition):
        """Add a UAV to the formation"""
        self.uavs[uav.uav_id] = uav
    
    def get_ideal_positions(self, num_uavs: int = 9) -> List[UAVPosition]:
        """Generate ideal positions for circular formation"""
        positions = [self.center]
        
        for i in range(num_uavs):
            angle = 2 * math.pi * i / num_uavs
            x = self.radius * math.cos(angle)
            y = self.radius * math.sin(angle)
            uav_id = f"FY{i+1:02d}"
            positions.append(UAVPosition(x, y, uav_id))
        
        return positions
    
    def calculate_angle_between_sources(self, receiver: UAVPosition, 
                                      source1: UAVPosition, 
                                      source2: UAVPosition) -> float:
        """Calculate angle between two signal sources as seen from receiver"""
        angle1 = receiver.angle_to(source1)
        angle2 = receiver.angle_to(source2)
        
        # Calculate the absolute difference, considering circular nature of angles
        diff = abs(angle1 - angle2)
        if diff > math.pi:
            diff = 2 * math.pi - diff
            
        return diff
    
    def trilateration_positioning(self, receiver_measurements: List[AngleMeasurement],
                                known_positions: dict) -> Tuple[float, float]:
        """
        Estimate receiver position using angle measurements from known transmitter positions
        
        This implements a simplified trilateration algorithm based on angle measurements.
        """
        if len(receiver_measurements) < 2:
            raise ValueError("At least 2 angle measurements needed for positioning")
        
        # For circular formation, we can use geometric constraints
        # This is a simplified implementation - in practice, more sophisticated
        # optimization algorithms would be used
        
        best_pos = None
        min_error = float('inf')
        
        # Grid search for demonstration (in practice, use gradient descent or other optimization)
        search_radius = self.radius * 1.5
        resolution = 1.0  # 1 meter resolution
        
        for x in np.arange(-search_radius, search_radius, resolution):
            for y in np.arange(-search_radius, search_radius, resolution):
                candidate = UAVPosition(x, y, "candidate")
                error = 0
                
                for measurement in receiver_measurements:
                    if (measurement.source1_id in known_positions and 
                        measurement.source2_id in known_positions):
                        
                        source1 = known_positions[measurement.source1_id]
                        source2 = known_positions[measurement.source2_id]
                        
                        predicted_angle = self.calculate_angle_between_sources(
                            candidate, source1, source2)
                        error += (predicted_angle - measurement.angle) ** 2
                
                if error < min_error:
                    min_error = error
                    best_pos = (x, y)
        
        return best_pos if best_pos else (0, 0)
    
    def minimum_transmitters_needed(self, num_receivers: int) -> int:
        """
        Calculate minimum number of transmitters needed for unique positioning
        
        For 2D positioning with angle-only measurements, theoretical minimum is 3 transmitters
        for unique positioning of multiple receivers in general case.
        """
        # For circular formation with known center, minimum is often 2 additional transmitters
        # plus the center transmitter
        return min(3, num_receivers + 1)

class ConeFormationModel:
    """Mathematical model for cone-shaped UAV formation positioning"""
    
    def __init__(self, spacing: float = 50.0):
        self.spacing = spacing  # spacing between adjacent UAVs
        self.uavs = {}
    
    def get_ideal_positions(self, num_lines: int = 5, uavs_per_line: List[int] = None) -> List[UAVPosition]:
        """Generate ideal positions for cone formation"""
        if uavs_per_line is None:
            uavs_per_line = [1, 2, 3, 4, 5]  # Example cone structure
        
        positions = []
        uav_counter = 0
        
        for line_idx, num_in_line in enumerate(uavs_per_line):
            y = line_idx * self.spacing
            
            # Center the line
            start_x = -(num_in_line - 1) * self.spacing / 2
            
            for pos_in_line in range(num_in_line):
                x = start_x + pos_in_line * self.spacing
                uav_id = f"FY{uav_counter:02d}"
                positions.append(UAVPosition(x, y, uav_id))
                uav_counter += 1
        
        return positions
    
    def positioning_strategy(self, measurements: List[AngleMeasurement]) -> dict:
        """Develop positioning strategy for cone formation"""
        # This would implement specific algorithms for cone formation positioning
        # For now, return a placeholder strategy
        return {
            "strategy": "sequential_positioning",
            "transmitter_selection": ["FY00", "FY01", "FY02"],
            "positioning_order": []
        }

def generate_sample_data() -> Tuple[List[UAVPosition], List[AngleMeasurement]]:
    """Generate sample data for testing"""
    # Create circular formation with some position errors
    model = CircularFormationModel(radius=100.0)
    ideal_positions = model.get_ideal_positions()
    
    # Add some random noise to simulate real-world positioning errors
    noisy_positions = []
    np.random.seed(42)  # For reproducible results
    
    for pos in ideal_positions:
        noise_x = np.random.normal(0, 2.0)  # 2-meter standard deviation
        noise_y = np.random.normal(0, 2.0)
        noisy_pos = UAVPosition(pos.x + noise_x, pos.y + noise_y, pos.uav_id)
        noisy_positions.append(noisy_pos)
    
    # Generate some angle measurements
    measurements = []
    receiver = noisy_positions[3]  # FY03 as receiver
    transmitters = [noisy_positions[0], noisy_positions[1], noisy_positions[2]]  # FY00, FY01, FY02
    
    for i in range(len(transmitters)):
        for j in range(i+1, len(transmitters)):
            angle = model.calculate_angle_between_sources(
                receiver, transmitters[i], transmitters[j])
            measurement = AngleMeasurement(
                receiver.uav_id, transmitters[i].uav_id, transmitters[j].uav_id, angle)
            measurements.append(measurement)
    
    return noisy_positions, measurements

if __name__ == "__main__":
    # Test the models
    print("=== UAV Formation Positioning Models ===")
    
    # Test circular formation
    print("\n1. Circular Formation Model:")
    circular_model = CircularFormationModel()
    ideal_positions = circular_model.get_ideal_positions()
    
    for pos in ideal_positions[:5]:  # Show first 5 positions
        print(f"  {pos}")
    
    print(f"\nMinimum transmitters needed: {circular_model.minimum_transmitters_needed(9)}")
    
    # Test with sample data
    print("\n2. Sample Data Generation:")
    positions, measurements = generate_sample_data()
    
    print("Angle measurements:")
    for measurement in measurements:
        print(f"  {measurement}")
    
    # Test cone formation
    print("\n3. Cone Formation Model:")
    cone_model = ConeFormationModel()
    cone_positions = cone_model.get_ideal_positions()
    
    for pos in cone_positions[:8]:  # Show first 8 positions
        print(f"  {pos}")