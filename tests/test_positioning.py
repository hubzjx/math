"""
Unit tests for UAV positioning models and algorithms

This module contains comprehensive tests for the UAV formation positioning system.
"""

import unittest
import numpy as np
import math
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.positioning_models import UAVPosition, AngleMeasurement, CircularFormationModel, ConeFormationModel
from src.positioning_algorithms import UAVPositioningAlgorithms, Problem1CSolver

class TestUAVPosition(unittest.TestCase):
    """Test UAVPosition class functionality"""
    
    def test_distance_calculation(self):
        pos1 = UAVPosition(0, 0, "FY00")
        pos2 = UAVPosition(3, 4, "FY01")
        self.assertAlmostEqual(pos1.distance_to(pos2), 5.0, places=2)
    
    def test_angle_calculation(self):
        pos1 = UAVPosition(0, 0, "FY00")
        pos2 = UAVPosition(1, 1, "FY01")
        angle = pos1.angle_to(pos2)
        self.assertAlmostEqual(angle, math.pi/4, places=2)

class TestCircularFormationModel(unittest.TestCase):
    """Test circular formation model"""
    
    def setUp(self):
        self.model = CircularFormationModel(radius=100.0)
    
    def test_ideal_positions_generation(self):
        positions = self.model.get_ideal_positions(9)
        self.assertEqual(len(positions), 10)  # 9 + center
        
        # Check center position
        center = positions[0]
        self.assertEqual(center.uav_id, "FY00")
        self.assertAlmostEqual(center.x, 0, places=1)
        self.assertAlmostEqual(center.y, 0, places=1)
        
        # Check circle positions are at correct radius
        for pos in positions[1:]:
            distance = center.distance_to(pos)
            self.assertAlmostEqual(distance, 100.0, places=1)
    
    def test_angle_calculation(self):
        receiver = UAVPosition(0, 0, "receiver")
        source1 = UAVPosition(1, 0, "source1")
        source2 = UAVPosition(0, 1, "source2")
        
        angle = self.model.calculate_angle_between_sources(receiver, source1, source2)
        self.assertAlmostEqual(angle, math.pi/2, places=2)

class TestPositioningAlgorithms(unittest.TestCase):
    """Test positioning algorithms"""
    
    def setUp(self):
        self.algorithms = UAVPositioningAlgorithms()
    
    def test_minimum_transmitters_calculation(self):
        # Test with different numbers of known transmitters
        transmitters_1 = {"FY00": UAVPosition(0, 0, "FY00")}
        result_1 = self.algorithms.solve_problem_1b(transmitters_1, [])
        self.assertEqual(result_1, 2)
        
        transmitters_2 = {
            "FY00": UAVPosition(0, 0, "FY00"),
            "FY01": UAVPosition(100, 0, "FY01")
        }
        result_2 = self.algorithms.solve_problem_1b(transmitters_2, [])
        self.assertEqual(result_2, 1)
        
        transmitters_3 = {
            "FY00": UAVPosition(0, 0, "FY00"),
            "FY01": UAVPosition(100, 0, "FY01"),
            "FY02": UAVPosition(0, 100, "FY02")
        }
        result_3 = self.algorithms.solve_problem_1b(transmitters_3, [])
        self.assertEqual(result_3, 0)
    
    def test_simple_positioning(self):
        # Test basic positioning with known transmitters
        transmitters = {
            "FY00": UAVPosition(0, 0, "FY00"),
            "FY01": UAVPosition(100, 0, "FY01"),
            "FY02": UAVPosition(0, 100, "FY02")
        }
        
        # Create angle measurements for a receiver at (50, 50)
        true_receiver = UAVPosition(50, 50, "receiver")
        model = CircularFormationModel()
        
        measurements = []
        angle1 = model.calculate_angle_between_sources(
            true_receiver, transmitters["FY00"], transmitters["FY01"])
        measurements.append(AngleMeasurement("receiver", "FY00", "FY01", angle1))
        
        angle2 = model.calculate_angle_between_sources(
            true_receiver, transmitters["FY00"], transmitters["FY02"])
        measurements.append(AngleMeasurement("receiver", "FY00", "FY02", angle2))
        
        # Solve for position
        estimated_pos = self.algorithms.solve_problem_1a(transmitters, measurements)
        
        # Check accuracy
        error = true_receiver.distance_to(estimated_pos)
        self.assertLess(error, 5.0)  # Should be within 5 meters

class TestProblem1CSolver(unittest.TestCase):
    """Test Problem 1(3) solver"""
    
    def setUp(self):
        self.solver = Problem1CSolver()
    
    def test_table1_data_parsing(self):
        positions = self.solver.parse_table1_data()
        self.assertEqual(len(positions), 10)  # 10 UAVs total
        
        # Check that center UAV is present
        center_uav = next((pos for pos in positions if pos.uav_id == "FY00"), None)
        self.assertIsNotNone(center_uav)
    
    def test_solve_with_table1_data(self):
        result = self.solver.solve_with_table1_data()
        
        # Check that solution structure is correct
        self.assertIn('steps', result)
        self.assertIn('transmitter_selections', result)
        self.assertIn('final_positions', result)
        self.assertIn('iterations', result)
        self.assertIn('accuracy_analysis', result)
        
        # Check that iterations converged
        self.assertGreater(result['iterations'], 0)
        self.assertLess(result['iterations'], 20)  # Should converge reasonably quickly

class TestVisualizationIntegration(unittest.TestCase):
    """Test integration with visualization components"""
    
    def test_sample_data_generation(self):
        from models.positioning_models import generate_sample_data
        positions, measurements = generate_sample_data()
        
        self.assertGreater(len(positions), 0)
        self.assertGreater(len(measurements), 0)
        
        # Check that measurements are valid
        for measurement in measurements:
            self.assertIsInstance(measurement.angle, float)
            self.assertGreaterEqual(measurement.angle, 0)
            self.assertLessEqual(measurement.angle, math.pi)

if __name__ == '__main__':
    # Set up test environment
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for testing
    
    # Run tests
    unittest.main(verbosity=2)