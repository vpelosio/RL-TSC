import random
import numpy as np
from enum import Enum
from vehicle_generator import *

class Scenario(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNBALANCED = "unbalanced"
    WAVE = "wave"


class TrafficGenerator:
    def __init__(self, sim_config, simulation_step):
        self.sim_config = sim_config
        self.simulation_step = simulation_step

        self.scenario_probs = {
            Scenario.LOW:        0.05,
            Scenario.MEDIUM:     0.25,
            Scenario.HIGH:       0.35,
            Scenario.UNBALANCED: 0.20,
            Scenario.WAVE:       0.15
        }

        self.vehicle_distribution = {
            'PassengerCar':             0.75891, 
            'LightCommercialVehicle':   0.08343, 
            'HeavyGoodsVehicle':        0.01393, 
            'Truck':                    0.00403, 
            'MotorCycle':               0.13781, 
            'Bus':                      0.00189
        }

    def generate_traffic(self, episode_index):
        # For reproducibility
        random.seed(episode_index)
        np.random.seed(episode_index)

        scenarios = list(self.scenario_probs.keys())
        weights = list(self.scenario_probs.values())

        selected_scenario = random.choices(scenarios, weights=weights, k=1)[0]

        n_vehicles = self._get_vehicle_count(selected_scenario)
        depart_times = self._get_depart_times(n_vehicles, selected_scenario)
        routes = self._get_routes(n_vehicles, selected_scenario)

        vehicle_list = VehicleList()
        v_types = list(self.vehicle_distribution.keys())
        v_probs = list(self.vehicle_distribution.values())

        # normalization
        v_probs = np.array(v_probs)
        v_probs = v_probs / v_probs.sum()

        random_vtype_strings = np.random.choice(v_types, size=n_vehicles, p=v_probs)

        for i, vtype_str in enumerate(random_vtype_strings):
            new_vehicle = eval(vtype_str).generateRandom("vehicle"+str(i))
            new_vehicle.depart = depart_times[i]
            new_vehicle.routeID = routes[i]
            new_vehicle.departLane = "free"

            vehicle_list.append(new_vehicle)

        return vehicle_list, n_vehicles, selected_scenario
    
    
    # Private Methods
    def _get_vehicle_count(self, scenario: Scenario):
        if scenario == Scenario.LOW:
            val = np.random.normal(600, 150)
            return int(np.clip(val, 250, 900))
        
        elif scenario == Scenario.MEDIUM:
            val = np.random.normal(1400, 300)
            return int(np.clip(val, 900, 2000))
        
        elif scenario == Scenario.HIGH:
            val = np.random.normal(2800, 400)
            return int(np.clip(val, 2000, 3200))
            
        elif scenario == Scenario.UNBALANCED:
            val = np.random.normal(2000, 300)
            return int(np.clip(val, 1500, 2500))
            
        elif scenario == Scenario.WAVE:
            val = np.random.normal(2600, 300)
            return int(np.clip(val, 2200, 3000))
            
        return 1000 
    
    def _get_depart_times(self, n, scenario: Scenario):
        max_depart_time = 3300 # 1h - 5m buffer so that the crossway can drain vehicles generated near the end

        if scenario == Scenario.WAVE:
            # 10% (15m) -> 80% (30m) -> 10% (10m)
            n_peak = int(n*0.80)
            n_rest = n - n_peak
            n_start = int(n_rest/2)
            n_end = n_rest - n_start # in case they are not even

            t1 = np.random.uniform(0, 900, n_start)                 # 0-15m
            t2 = np.random.uniform(900, 2700, n_peak)               # 15-45m
            t3 = np.random.uniform(2700, max_depart_time, n_end)    # 45-55m

            times = np.concatenate([t1, t2, t3])
        else:
            times = np.random.uniform(0, max_depart_time, n)
        
        times = np.round(times / self.simulation_step) * self.simulation_step # round to simulation_step

        return times
    
    def _get_routes(self, n, scenario: Scenario):
        weights = {rid: 0.0 for rid in self.sim_config.route_ids}

        def set_w(group, val):
            if group in self.sim_config.routes_map:
                for r in self.sim_config.routes_map[group]:
                    weights[r] = val

        # Case 1: Standard (Low, Med, Wave) -> 70% Straight, 15% Turn
        if scenario in [Scenario.LOW, Scenario.MEDIUM, Scenario.WAVE]:
            for prefix in ["NS", "EW"]:
                set_w(f"{prefix}_Straight", 70.0)
                set_w(f"{prefix}_Right", 15.0)
                set_w(f"{prefix}_Left", 15.0)
        
        # Case 2: High -> 85% Straight, 10% dx, 5% sx (to minimize starvation for left turn)
        elif scenario == Scenario.HIGH:
            for prefix in ["NS", "EW"]:
                set_w(f"{prefix}_Straight", 85.0)
                set_w(f"{prefix}_Right", 10.0)
                set_w(f"{prefix}_Left", 5.0)

        # Case 3: Unbalanced -> N-S dominates. Main road during rush hour
        elif scenario == Scenario.UNBALANCED:
            vol_ns = 9.0
            vol_ew = 1.0

            set_w('NS_Straight', 85.0 * vol_ns)
            set_w('NS_Right', 10.0 * vol_ns)
            set_w('NS_Left', 5.0 * vol_ns)
            
            set_w('EW_Straight', 70.0 * vol_ew)
            set_w('EW_Right', 15.0 * vol_ew)
            set_w('EW_Left', 15.0 * vol_ew)
        
        ordered_weights = np.array([weights[rid] for rid in self.sim_config.route_ids])
        normalized_weights = ordered_weights / ordered_weights.sum()

        routes = np.random.choice(self.sim_config.route_ids, size=n, p=normalized_weights)
        
        return routes


