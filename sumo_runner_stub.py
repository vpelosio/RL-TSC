import os
import sys
import time
import libsumo
from xml.dom import minidom
from vehicle_generator import *
from traffic_generator import *
from sim_config import *

def startSumo(map_name, simulation_step):
    libsumo.start(["sumo", "-c", "sumo_xml_files/" + map_name + "/" + map_name + ".sumocfg", "--waiting-time-memory", "3000", "--start", "--quit-on-end", "--verbose", "--step-length", str(simulation_step)])

def addVehiclesToSimulation(vehicleList):
    for v in vehicleList:
        libsumo.vehicle.add(vehID=v.vehicleID, routeID=v.routeID, typeID='vtype-'+v.vehicleID, depart=v.depart, departSpeed=v.initialSpeed, departLane=v.departLane)

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


def main():
    simulation_step = 0.5
    traffic_generator = TrafficGenerator(CONFIG_4WAY_160M, simulation_step)
    vehicle_list = traffic_generator.generate_traffic(0)

    generateVehicleTypesXML(vehicle_list)
    startSumo(CONFIG_4WAY_160M.name, simulation_step)
    addVehiclesToSimulation(vehicle_list)

    # temporary
    for tl in libsumo.trafficlight.getIDList():
        libsumo.trafficlight.setProgram(tl, "0")

    activeVehicles = set()

    while libsumo.simulation.getMinExpectedNumber() > 0:
        libsumo.simulationStep()

        # aggiornamento set
        activeVehicles.update(libsumo.simulation.getDepartedIDList())
        activeVehicles.difference_update(libsumo.simulation.getArrivedIDList())

        # misure
        for vehicle in activeVehicles:
            vehicle_list.getVehicle(vehicle).doMeasures()
        
    # fine simulazione
    libsumo.close()
    sys.stdout.flush()

if __name__ == "__main__":
    main()