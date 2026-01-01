import os
import sys
import time
from xml.dom import minidom
from vehicle_generator import *
from traffic_generator import *
from sim_config import *

def generateVehicleTypesXML(vehicleList):
    rootXML = minidom.Document()
    routes = rootXML.createElement('routes')
    rootXML.appendChild(routes)

    # creazione vTypes
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
    traffic_generator = TrafficGenerator(CONFIG_4WAY_160M, 0.5)
    vehicle_list = traffic_generator.generate_traffic(1)

    generateVehicleTypesXML(vehicle_list)

if __name__ == "__main__":
    main()