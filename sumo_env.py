import gymnasium as gym
import numpy as np
import libsumo
import os
import shutil
from traffic_generator import TrafficGenerator
from xml.dom import minidom
from gymnasium import spaces
from sim_config import *
from traffic_light import TrafficLight



class SumoEnv(gym.Env):
    def __init__(self, sim_config, sim_step, action_step, episode_duration, log_folder, rank = 0, episode_offset = 0, enable_measure = False, gui=False, episode_list = []):
        super(SumoEnv, self).__init__()
        self.sim_config = sim_config
        self.gui = gui
        self.rank = rank

        self.template_xml_path = "sumo_xml_template_files"
        self.workspace_path = os.path.join("sumo_workspace", f"env_{self.rank}")
        self.sumo_config_path = os.path.join(
            self.workspace_path, 
            CONFIG_4WAY_160M.name, 
            CONFIG_4WAY_160M.name + ".sumocfg"
        )

        self._setup_workspace()
        self.episode_list = episode_list
        self.episode_list_mode = len(self.episode_list)
        self.episode_count = 0
        self.episode_id = episode_offset

        self.measure_enabled = enable_measure
        self.active_vehicles = set()
        self.vehicle_list = []
        self.obs_history = []
        
        self.sim_step = sim_step
        self.action_step = action_step
        self.episode_duration = episode_duration

        self.steps_per_action = int(action_step / sim_step)
        self.traffic_gen = TrafficGenerator(self.sim_config, sim_step)

        self.log_folder = log_folder

        # Action 0: N/S Green
        # Action 1: E/W Green
        self.action_space = spaces.Discrete(2)
        
        # Discrete Traffic State Encoding DTSE
        self.num_lanes = 8 
        self.lane_length = 160 
        self.cell_length = 5 
        self.num_cells = int(self.lane_length / self.cell_length) 
        
        # Matrix (8 * 32) + phase (2 one-hot) + duration (1 float)
        input_dims = (self.num_lanes * self.num_cells) + 3 
        
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(input_dims,), dtype=np.float32
        )

        self.lane_ids_list = []

    def _reset_vehicles_measures(self):
        for v in self.vehicle_list:
            v.resetMeasures()
    
    def run_smart_traffic_light(self, improvments):
        self._reset_vehicles_measures()
        self._startSumo(self.sumo_config_path, self.sim_step, self.log_folder, self.episode_id)
        self._addVehiclesToSimulation(self.vehicle_list)
        libsumo.trafficlight.setProgram(self.sim_config.tl_id, self.sim_config.tl_program)
        tl = TrafficLight(self.sim_config.tl_id, improvments)
        while libsumo.simulation.getMinExpectedNumber() > 0:
            self._simulation_step()
            tl.performStep()

    def _startSumo(self, config_file_path, simulation_step, log_folder, episode_index):
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

    def _addVehiclesToSimulation(self, vehicleList):
        for v in vehicleList:
            libsumo.vehicle.add(vehID=v.vehicleID, routeID=v.routeID, typeID='vtype-'+v.vehicleID, depart=v.depart, departSpeed=v.initialSpeed, departLane=v.departLane)

    def _generateVehicleTypesXML(self, vehicleList, output_folder):
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

    def _log_scenario(self, log_folder, episode_index, vehicle_num, scenario):
        episode_info_file = os.path.join(log_folder, f"episode_info_ep{episode_index}.txt")

        episode_info = (
            f"==================================================\n"
            f" EPISODE INFO\n"
            f"==================================================\n"
            f" Episode ID  : {episode_index}\n"
            f" Scenario Type  : {scenario}\n"
            f" Total Vehicles : {vehicle_num}\n"
            f"==================================================\n"
        )

        with open(episode_info_file, 'w') as f:
            f.write(episode_info)

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
        self.obs_history = []
        if self.episode_list_mode:
            list_index = self.episode_count % len(self.episode_list)
            self.episode_id = self.episode_list[list_index]
        else:
            self.episode_id = self.episode_id + 1
        
        self.episode_count += 1
        
        self.active_vehicles = set()
        self.vehicle_list = []

        vehicle_list, vehicle_num, scenario = self.traffic_gen.generate_traffic(self.episode_id)
        self.vehicle_list = vehicle_list
        self._generateVehicleTypesXML(self.vehicle_list, output_folder=self.workspace_path)

        self._log_scenario(self.log_folder, self.episode_id, vehicle_num, scenario)

        self._startSumo(self.sumo_config_path, self.sim_step, self.log_folder, self.episode_id)
        self._addVehiclesToSimulation(self.vehicle_list)
        libsumo.trafficlight.setProgram(self.sim_config.tl_id, self.sim_config.tl_program)

        if not self.lane_ids_list:
            lanes = sorted(list(set(libsumo.trafficlight.getControlledLanes(self.sim_config.tl_id))))
            self.lane_ids_list = lanes[:8] if len(lanes) >= 8 else lanes

        obs = self._compute_observation()
        self.episode_co2_total = 0.0
        return obs, {}
    
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
        
        total_co2 = 0.0
        total_waiting_time = 0.0
        delta_t = libsumo.simulation.getDeltaT()

        # Green->Yellow and Yellow->Red transition management
        if current_phase != target_phase:
            next_phase = (current_phase + 1) % 6
            while next_phase != target_phase:
                libsumo.trafficlight.setPhase(self.sim_config.tl_id, next_phase)
                duration = libsumo.trafficlight.getPhaseDuration(self.sim_config.tl_id)
                steps = int(duration / self.sim_step)
                for _ in range(steps): 
                    self._simulation_step()
                    ids = libsumo.vehicle.getIDList()
                    for v in ids:
                        total_co2 += (libsumo.vehicle.getCO2Emission(v) * delta_t) / 1000 # um: g
                        total_waiting_time += libsumo.vehicle.getWaitingTime(v)

                next_phase = (next_phase + 1) % 6

        # Green execution
        libsumo.trafficlight.setPhase(self.sim_config.tl_id, target_phase)

        max_waiting_time = 0.0
        for _ in range(self.steps_per_action): # steps per action -> min green time
            self._simulation_step()
            ids = libsumo.vehicle.getIDList()
            for v in ids:
                total_co2 += (libsumo.vehicle.getCO2Emission(v) * delta_t) / 1000 # um: g
                total_waiting_time += libsumo.vehicle.getWaitingTime(v)
                max_waiting_time = max(max_waiting_time, libsumo.vehicle.getWaitingTime(v))

        # --- Reward computation ---
        self.episode_co2_total += total_co2
        co2_grams = total_co2
        w_waiting_time = 1

        r_waiting_time = w_waiting_time * total_waiting_time
        reward = -(r_waiting_time)/10000

        # Anti-Starvation
        penalty = 0.0
        if max_waiting_time > 180: # max phase duration in Denny's code
            penalty = (max_waiting_time - 180) * 0.5
            reward -= (penalty / 10000)
        
        current_time = libsumo.simulation.getTime()
        terminated = libsumo.simulation.getMinExpectedNumber() == 0
        truncated = current_time >= self.episode_duration

        obs = self._compute_observation()
        info = {
            "co2": total_co2,
            "waiting_time": total_waiting_time,
            "max_waiting_time": max_waiting_time
        }

        if terminated or truncated:
            info["episode_avgco2"] = self.episode_co2_total / len(self.vehicle_list)

        return obs, reward, terminated, truncated, info
    
    def close(self):
        libsumo.close()

    def _compute_observation(self):
        # -1 empty cell, 0 stopped vehicle, >0 normalized speed
        traffic_grid = np.full((self.num_lanes, self.num_cells), -1.0, dtype=np.float32)

        # TO BE FIXED: hardcoded lane order        
        ordered_lanes = [
            "E4_0", "E4_1", # Nord (Incoming)
            "E3_0", "E3_1", # Est
            "E2_0", "E2_1", # Sud
            "E1_0", "E1_1"  # Ovest
        ]
        
        for i, lane_id in enumerate(ordered_lanes):
            vehicle_ids = libsumo.lane.getLastStepVehicleIDs(lane_id)
            
            for veh_id in vehicle_ids:
                pos = libsumo.vehicle.getLanePosition(veh_id)
                speed = libsumo.vehicle.getSpeed(veh_id)
                max_speed = libsumo.vehicle.getAllowedSpeed(veh_id)
                
                dist_to_intersection = self.lane_length - pos
                
                cell_idx = int(dist_to_intersection / self.cell_length)
                
                cell_idx = min(cell_idx, self.num_cells - 1)
                cell_idx = max(cell_idx, 0)
                
                if max_speed > 0:
                    norm_speed = speed / max_speed
                else:
                    norm_speed = 0.0
                    
                traffic_grid[i][cell_idx] = norm_speed

        flat_grid = traffic_grid.flatten()
        
        phase = libsumo.trafficlight.getPhase(self.sim_config.tl_id)
        duration = libsumo.trafficlight.getSpentDuration(self.sim_config.tl_id)
        
        phase_info = np.array([
            1.0 if phase == 0 else 0.0,
            1.0 if phase == 3 else 0.0,
            min(1.0, duration / 120.0)
        ], dtype=np.float32)
        
        return np.concatenate((flat_grid, phase_info))