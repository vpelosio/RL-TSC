import gymnasium as gym
import numpy as np
import libsumo
import os
import shutil
from traffic_generator import TrafficGenerator
from xml.dom import minidom
from gymnasium import spaces
from sim_config import *

def log_scenario(log_folder, episode_index, vehicle_num, scenario):
    episode_info_file = os.path.join(log_folder, f"episode_info_ep{episode_index}.txt")

    episode_info = (
        f"==================================================\n"
        f" EPISODE INFO\n"
        f"==================================================\n"
        f" Episode Index  : {episode_index}\n"
        f" Scenario Type  : {scenario}\n"
        f" Total Vehicles : {vehicle_num}\n"
        f"==================================================\n"
    )

    with open(episode_info_file, 'w') as f:
        f.write(episode_info)

def startSumo(config_file_path, simulation_step, log_folder, episode_index):
    try:
        libsumo.close()
    except:
        pass

    sumo_log_file = os.path.join(log_folder, f"sumo_output_ep{episode_index}.txt")
    libsumo.start([
        "sumo", 
        "-c", config_file_path, 
        "--waiting-time-memory", "3600", 
        "--start", 
        "--quit-on-end", 
        "--verbose", 
        "--step-length", str(simulation_step),
        "--log", sumo_log_file,
        "--time-to-teleport", "-1" # disable teleport
        ])

def addVehiclesToSimulation(vehicleList):
    for v in vehicleList:
        libsumo.vehicle.add(vehID=v.vehicleID, routeID=v.routeID, typeID='vtype-'+v.vehicleID, depart=v.depart, departSpeed=v.initialSpeed, departLane=v.departLane)

def generateVehicleTypesXML(vehicleList, output_folder):
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

    output_path = os.path.join(output_folder, "vehicletypes.rou.xml")
    with open(output_path, 'w') as fd:
        fd.write(rootXML.toprettyxml(indent="    "))


class SumoEnv(gym.Env):
    def __init__(self, sim_config, sim_step, action_step, episode_duration, log_folder, rank = 0, episode_offset = 0, enable_measure = False, gui=False):
        super(SumoEnv, self).__init__()
        self.sim_config = sim_config
        self.gui = gui
        self.rank = rank

        self.template_xml_path = "sumo_xml_template_files"
        self.workspace_path = f"sumo_workspace\\env_{self.rank}"

        self._setup_workspace()

        self.episode_count = episode_offset

        self.measure_enabled = enable_measure
        self.active_vehicles = set()
        self.vehicle_list = None
        
        self.sim_step = sim_step
        self.action_step = action_step
        self.episode_duration = episode_duration

        self.steps_per_action = int(action_step / sim_step)
        self.traffic_gen = TrafficGenerator(self.sim_config, sim_step)

        self.log_folder = log_folder

        # Action 0: N/S Green
        # Action 1: E/W Green
        self.action_space = spaces.Discrete(2)
        
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(9,), dtype=np.float32
        )

        self.lane_ids = []

    def _setup_workspace(self):
        if os.path.exists(self.workspace_path):
            shutil.rmtree(self.workspace_path, ignore_errors=True)
        
        shutil.copytree(self.template_xml_path, self.workspace_path)
        
        print(f"[Env {self.rank}] Workspace created in: {self.workspace_path}")

    def _simulation_step(self):
        libsumo.simulationStep()

        if self.measure_enabled:
            self.active_vehicles.update(libsumo.simulation.getDepartedIDList())
            self.active_vehicles.difference_update(libsumo.simulation.getArrivedIDList())

            for vehicle in self.active_vehicles:
                self.vehicle_list.getVehicle(vehicle).doMeasures()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1
        self.active_vehicles = set()
        self.vehicle_list = None

        vehicle_list, vehicle_num, scenario = self.traffic_gen.generate_traffic(self.episode_count)
        self.vehicle_list = vehicle_list
        generateVehicleTypesXML(vehicle_list, output_folder=self.workspace_path)

        log_scenario(self.log_folder, self.episode_count, vehicle_num, scenario)
        config_path = os.path.join(
            self.workspace_path, 
            CONFIG_4WAY_160M.name, 
            CONFIG_4WAY_160M.name + ".sumocfg"
        )
        startSumo(config_path, self.sim_step, self.log_folder, self.episode_count)
        addVehiclesToSimulation(vehicle_list)
        libsumo.trafficlight.setProgram(self.sim_config.tl_id, self.sim_config.tl_program)

        if not self.lane_ids:
            lanes = sorted(list(set(libsumo.trafficlight.getControlledLanes(self.sim_config.tl_id))))
            self.lane_ids = lanes[:8] if len(lanes) >= 8 else lanes

        return self._compute_observation(), {}
    
    def get_measures(self):
        mesaured_vehicle_data = []
        for v in self.vehicle_list:
            data = {"vehicleID": v.vehicleID, "totalDistance": v.totalDistance, "totalTravelTime": v.totalTravelTime, "totalWaitingTime": v.totalWaitingTime, "meanSpeed": v.meanSpeed, "totalCO2Emissions": v.totalCO2Emissions, "totalCOEmissions": v.totalCOEmissions, "totalHCEmissions": v.totalHCEmissions, "totalPMxEmissions": v.totalPMxEmissions, "totalNOxEmissions": v.totalNOxEmissions, "totalFuelConsumption": v.totalFuelConsumption, "totalElectricityConsumption": v.totalElectricityConsumption, "totalNoiseEmission": v.totalNoiseEmission}
            mesaured_vehicle_data.append(data)
        return mesaured_vehicle_data
    
    def dump_vehicle_population(self, filename):
        self.vehicle_list.dump(filename)
    
    def step(self, action):
        action = int(action)
        target_phase = action * 3

        current_phase = libsumo.trafficlight.getPhase(self.sim_config.tl_id)

        # Green->Yellow and Yellow->Red transition management
        if current_phase != target_phase:
            next_phase = (current_phase + 1) % 6
            while next_phase != target_phase:
                libsumo.trafficlight.setPhase(self.sim_config.tl_id, next_phase)
                duration = libsumo.trafficlight.getPhaseDuration(self.sim_config.tl_id)
                steps = int(duration / self.sim_step)
                for _ in range(steps): 
                    self._simulation_step()

                next_phase = (next_phase + 1) % 6

        # Green execution
        libsumo.trafficlight.setPhase(self.sim_config.tl_id, target_phase)

        total_co2 = 0.0
        total_waiting_time = 0.0
        max_waiting_time = 0.0

        for _ in range(self.steps_per_action): # steps per action -> min green time
            self._simulation_step()
            vehicles = libsumo.vehicle.getIDList()
            for v in vehicles:
                co2 = (libsumo.vehicle.getCO2Emission(v) * libsumo.simulation.getDeltaT()) / 1000  # g
                waiting_time = libsumo.vehicle.getWaitingTime(v) # s

                total_co2 += co2
                total_waiting_time += waiting_time

                max_waiting_time = max(max_waiting_time, waiting_time)

        # --- Reward computation ---
        
        # From test data CO2 (g) is 7.5 times the Wait time (s)
        # Use weight 4.0 to balance the ratio
        co2_grams = total_co2
        w_co2 = 1.0
        w_waiting_time = 4.0

        r_co2_part = w_co2 * co2_grams
        r_waiting_time = w_waiting_time * total_waiting_time
        reward = -(r_co2_part + r_waiting_time)/10000

        # Anti-Starvation
        penalty = 0.0
        if max_waiting_time > 180: # max phase duration in Denny's code
            penalty = (max_waiting_time - 180) * 0.5
            reward -= penalty

        current_time = libsumo.simulation.getTime()
        terminated = libsumo.simulation.getMinExpectedNumber() == 0
        truncated = current_time >= self.episode_duration

        if truncated and not terminated:
            vehicles_left = libsumo.vehicle.getIDCount()
            reward -= (vehicles_left * 0.5)

        obs = self._compute_observation()
        info = {
            "co2": total_co2,
            "waiting_time": total_waiting_time,
            "max_waiting_time": max_waiting_time
        }

        return obs, reward, terminated, truncated, info
    
    def close(self):
        libsumo.close()

    def _compute_observation(self):
        obs = []
        max_cars = 26.0
        for lane in self.lane_ids:
            q = libsumo.lane.getLastStepHaltingNumber(lane)
            obs.append(min(q, max_cars) / max_cars) # norm 0 to 1
        
        phase = libsumo.trafficlight.getPhase(self.sim_config.tl_id)
        obs.append(1.0 if phase == 3 else 0.0)

        return np.array(obs, dtype=np.float32)        