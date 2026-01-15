import libsumo

J = 100
K = 1
Ke = 5

# Implementation of Denny Ciccia from: https://github.com/dennyciccia/sumo-simulations
class TrafficLight:
    def __init__(self, tlID, enhancements):
        self.__tlID = tlID
        self.__enhancements = enhancements  # list of algorithm improvements
    
    @property
    def tlID(self):
        return self.__tlID

    @property
    def enhancements(self):
        return self.__enhancements
    
    @property
    def movingFlow(self):
        if libsumo.trafficlight.getPhase(self.tlID) in [3,4,5]:   # horizontal flow
            return 'HORIZONTAL'
        elif libsumo.trafficlight.getPhase(self.tlID) in [0,1,2]: # vertical flow
            return 'VERTICAL'

    # traffic light switch to change the flow of traffic
    def switchTrafficLight(self):
        libsumo.trafficlight.setPhase(self.tlID, libsumo.trafficlight.getPhase(self.tlID)+1)

    def getHorizontalEdges(self):
        horizontalEdges = []
        for edge in libsumo.junction.getIncomingEdges(self.tlID):
            if libsumo.edge.getAngle(edge) in [90.0, 270.0]:
                horizontalEdges.append(edge)

        return horizontalEdges

    def getVerticalEdges(self):
        verticalEdges = []
        for edge in libsumo.junction.getIncomingEdges(self.tlID):
            if libsumo.edge.getAngle(edge) in [0.0, 180.0]:
                verticalEdges.append(edge)

        return verticalEdges
    
    # cost flow calculation
    def getFlowCosts(self):
        costH = costV = 0
        
        for edge in self.getHorizontalEdges():
            for vehicle in libsumo.edge.getLastStepVehicleIDs(edge):
                if 1 not in self.enhancements:
                    costH += J + K * (libsumo.vehicle.getSpeed(vehicle) ** 2)
                else:
                    costH += J + (Ke if self.movingFlow == 'HORIZONTAL' else K) * (libsumo.vehicle.getSpeed(vehicle) ** 2)

        for edge in self.getVerticalEdges():
            for vehicle in libsumo.edge.getLastStepVehicleIDs(edge):
                if 1 not in self.enhancements:
                    costV += J + K * (libsumo.vehicle.getSpeed(vehicle) ** 2)
                else:
                    costV += J + (Ke if self.movingFlow == 'VERTICAL' else K) * (libsumo.vehicle.getSpeed(vehicle) ** 2)
        
        return costH, costV
    
    def tryToSkipRed(self):
        meanSpeedH = meanSpeedV = 0
        vehicleNumberH = vehicleNumberV = 0

        for edge in self.getHorizontalEdges():
            meanSpeedH += libsumo.edge.getLastStepMeanSpeed(edge)
            vehicleNumberH += libsumo.edge.getLastStepVehicleNumber(edge)
        meanSpeedH /= len(self.getHorizontalEdges())

        for edge in self.getVerticalEdges():
            meanSpeedV += libsumo.edge.getLastStepMeanSpeed(edge)
            vehicleNumberV += libsumo.edge.getLastStepVehicleNumber(edge)
        meanSpeedV /= len(self.getVerticalEdges())

        # if the vehicles are stationary or there are none, proceed to the green phase
        if (self.movingFlow == 'HORIZONTAL' and (meanSpeedH < 1.0 or vehicleNumberH == 0)) or (self.movingFlow == 'VERTICAL' and (meanSpeedV < 1.0 or vehicleNumberV == 0)):
            libsumo.trafficlight.setPhase(self.tlID, (libsumo.trafficlight.getPhase(self.tlID)+2)%6)
    
    # actions performed at each step of the simulation
    # Here, if we are not in improvement 2 and I am not giving the green light to a direction, nothing is done and the default action of the XML is maintained.
    def performStep(self):
        if 2 in self.enhancements:
            # If we are at the end of the yellow phase, try to skip the red-only phase if it is safe to do so.
            if libsumo.trafficlight.getPhase(self.tlID) in [1,4] and 2 <= libsumo.trafficlight.getSpentDuration(self.tlID) < 3:
                self.tryToSkipRed()

        # maximum 180 seconds of green for one flow
        if libsumo.trafficlight.getSpentDuration(self.tlID) >= 180.0: # implicitly refers to a green phase because yellow and caution last 3 seconds
            self.switchTrafficLight()
            return
        
        # minimum 10 seconds of green for a flow and check that you are not in a phase with yellow or red only
        if libsumo.trafficlight.getSpentDuration(self.tlID) > 10 and libsumo.trafficlight.getPhase(self.tlID) not in [1,2,4,5]: # I am giving the green light for at least 10 seconds to one of the two directions
            costH, costV = self.getFlowCosts() 
            if (self.movingFlow == 'HORIZONTAL' and costH < costV) or (self.movingFlow == 'VERTICAL' and costV < costH): # consider whether it is appropriate to switch
                self.switchTrafficLight()
