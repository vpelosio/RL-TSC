import gymnasium as gym
import numpy as np
import libsumo as traci
import os
import sys
import csv
from traffic_generator import TrafficGenerator
from xml.dom import minidom
from gymnasium import spaces
from sim_config import *

def startSumo(map_name, simulation_step):
    try:
        traci.close()
    except:
        pass

    traci.start(["sumo", "-c", "sumo_xml_files/" + map_name + "/" + map_name + ".sumocfg", "--waiting-time-memory", "3000", "--start", "--quit-on-end", "--verbose", "--step-length", str(simulation_step)])

def addVehiclesToSimulation(vehicleList):
    for v in vehicleList:
        traci.vehicle.add(vehID=v.vehicleID, routeID=v.routeID, typeID='vtype-'+v.vehicleID, depart=v.depart, departSpeed=v.initialSpeed, departLane=v.departLane)

def generateVehicleTypesXML(vehicleList):
    rootXML = minidom.Document()
    routes = rootXML.createElement('routes')
    rootXML.appendChild(routes)

    for v in vehicleList:
        vtype = rootXML.createElement('vType')
        vtype.setAttribute('id', 'vtype-'+v.vehicleID)
        vtype.setAttribute('length', str(v.length))
        vtype.setAttribute('mass', str(v.weight))
        vtype.setAttribute('maxSpeed', str(v.maxSpeed))
        vtype.setAttribute('accel', str(v.acceleration))
        vtype.setAttribute('decel', str(v.brakingAcceleration))
        vtype.setAttribute('emergencyDecel', str(v.fullBrakingAcceleration))
        vtype.setAttribute('minGap', str(v.minGap))
        vtype.setAttribute('tau', str(v.driverProfile.tau))
        vtype.setAttribute('sigma', str(v.driverProfile.sigma))
        vtype.setAttribute('speedFactor', str(v.driverProfile.speedLimitComplianceFactor))
        vtype.setAttribute('vClass', str(v.vClass))
        vtype.setAttribute('emissionClass', str(v.emissionClass))
        vtype.setAttribute('color', str(v.color))
        vtype.setAttribute('guiShape', str(v.shape))
        routes.appendChild(vtype)

    # scrittura dell'XML generato
    with open("sumo_xml_files/vehicletypes.rou.xml", 'w') as fd:
        fd.write(rootXML.toprettyxml(indent="    "))


class SumoEnv(gym.Env):
    def __init__(self, sim_config, sim_step, action_step, episode_duration, log_file_path, gui=False):
        super(SumoEnv, self).__init__()
        self.sim_config = sim_config
        self.gui = gui

        self.episode_count = 0
        
        self.sim_step = sim_step
        self.action_step = action_step
        self.episode_duration = episode_duration

        self.steps_per_action = int(action_step / sim_step)
        self.traffic_gen = TrafficGenerator(self.sim_config, sim_step)

        self.log_file_path = log_file_path

        # Action 0: N/S Green
        # Action 1: E/W Green
        self.action_space = spaces.Discrete(2)
        
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(9,), dtype=np.float32
        )

        self.lane_ids = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1

        vehicle_list = self.traffic_gen.generate_traffic(self.episode_count)
        generateVehicleTypesXML(vehicle_list)
        startSumo(CONFIG_4WAY_160M.name, self.sim_step)
        addVehiclesToSimulation(vehicle_list)
        traci.trafficlight.setProgram(self.sim_config.tl_id, self.sim_config.tl_program)

        if not self.lane_ids:
            lanes = sorted(list(set(traci.trafficlight.getControlledLanes(self.sim_config.tl_id))))
            self.lane_ids = lanes[:8] if len(lanes) >= 8 else lanes

        return self._compute_observation(), {}
    
    def step(self, action):
        action = int(action)
        target_phase = action * 3

        current_phase = traci.trafficlight.getPhase(self.sim_config.tl_id)

        # Green->Yellow and Yellow->Red transition management
        if current_phase != target_phase:
            next_phase = (current_phase + 1) % 6
            while next_phase != target_phase:
                traci.trafficlight.setPhase(self.sim_config.tl_id, next_phase)
                duration = traci.trafficlight.getPhaseDuration(self.sim_config.tl_id)
                steps = int(duration / self.sim_step)
                for _ in range(steps): traci.simulationStep()

                next_phase = (next_phase + 1) % 6

        # Green execution
        traci.trafficlight.setPhase(self.sim_config.tl_id, target_phase)

        total_co2 = 0.0
        total_waiting_time = 0.0
        max_waiting_time = 0.0

        for _ in range(self.steps_per_action): # steps per action -> min green time
            traci.simulationStep()
            vehicles = traci.vehicle.getIDList()
            for v in vehicles:
                co2 = traci.vehicle.getCO2Emission(v) # mg
                waiting_time = traci.vehicle.getWaitingTime(v) # s

                total_co2 += co2
                total_waiting_time += waiting_time

                max_waiting_time = max(max_waiting_time, waiting_time)

        # --- Reward computation ---
        
        # From test data CO2 (g) is 7.5 times the Wait time (s)
        # Use weight 4.0 to balance the ratio
        co2_grams = total_co2 / 1000.0
        w_co2 = 1.0
        w_waiting_time = 4.0

        r_co2_part = w_co2 * co2_grams
        r_waiting_time = w_waiting_time * total_waiting_time
        reward = -(r_co2_part + r_waiting_time)

        # Anti-Starvation
        penalty = 0.0
        if max_waiting_time > 180: # max phase duration in Denny's code
            penalty = (max_waiting_time - 180) * 0.5
            reward -= penalty

        current_time = traci.simulation.getTime()
        terminated = traci.simulation.getMinExpectedNumber() == 0
        truncated = current_time >= self.episode_duration

        if truncated and not terminated:
            vehicles_left = traci.vehicle.getIDCount()
            reward -= (vehicles_left * 0.5)

        # Logs
        with open(self.log_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                self.episode_count,
                round(current_time, 1),
                action,
                round(co2_grams, 2),
                round(total_waiting_time, 2),
                round(r_co2_part, 2),
                round(r_waiting_time, 2),
                round(reward, 2),
                round(max_waiting_time, 1),
                round(penalty, 2)
            ])

        obs = self._compute_observation()
        info = {
            "co2": total_co2,
            "waiting_time": total_waiting_time,
            "max_waiting_time": max_waiting_time
        }

        return obs, reward, terminated, truncated, info
    
    def close(self):
        traci.close()

    def _compute_observation(self):
        obs = []
        max_cars = 26.0
        for lane in self.lane_ids:
            q = traci.lane.getLastStepHaltingNumber(lane)
            obs.append(min(q, max_cars) / max_cars) # norm 0 to 1
        
        phase = traci.trafficlight.getPhase(self.sim_config.tl_id)
        obs.append(1.0 if phase == 3 else 0.0)

        return np.array(obs, dtype=np.float32)        