import libsumo

J = 100
K = 1
Ke = 5

# Implementation of Denny Ciccia from: https://github.com/dennyciccia/sumo-simulations
class TrafficLight:
    def __init__(self, tlID, enhancements):
        self.__tlID = tlID
        self.__enhancements = enhancements  # lista dei miglioramenti dell'algoritmo
    
    @property
    def tlID(self):
        return self.__tlID

    @property
    def enhancements(self):
        return self.__enhancements
    
    @property
    def movingFlow(self):
        if libsumo.trafficlight.getPhase(self.tlID) in [3,4,5]:   # flusso orizzontale
            return 'HORIZONTAL'
        elif libsumo.trafficlight.getPhase(self.tlID) in [0,1,2]: # flusso verticale
            return 'VERTICAL'

    # switch del semaforo per cambiare il flusso che si muove
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
    
    # calcolo dei costi dei flussi
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

        # se i veicoli sono fermi o non ci sono vai alla fase verde
        if (self.movingFlow == 'HORIZONTAL' and (meanSpeedH < 1.0 or vehicleNumberH == 0)) or (self.movingFlow == 'VERTICAL' and (meanSpeedV < 1.0 or vehicleNumberV == 0)):
            libsumo.trafficlight.setPhase(self.tlID, (libsumo.trafficlight.getPhase(self.tlID)+2)%6)
    
    # azioni eseguite a ogni step della simulazione
    # Qua se non siamo nell'improvment 2 e non sto dando il verde ad una direzione non viene fatto nulla e viene tenuta l'azione di defuault dell'xml
    def performStep(self):
        if 2 in self.enhancements:
            # se siamo alla fine della fase giallo prova a saltare la fase di solo rosso se è sicuro farlo
            if libsumo.trafficlight.getPhase(self.tlID) in [1,4] and 2 <= libsumo.trafficlight.getSpentDuration(self.tlID) < 3:
                self.tryToSkipRed()

        # massimo 180s di verde per un flusso
        if libsumo.trafficlight.getSpentDuration(self.tlID) >= 180.0: #implicitamente riguarda una fase di verde perché giallo e salvaguardia durano 3s
            self.switchTrafficLight()
            return
        
        # minimo 10s di verde per un flusso e controllo di non essere in una fase con giallo o solo rosso
        if libsumo.trafficlight.getSpentDuration(self.tlID) > 10 and libsumo.trafficlight.getPhase(self.tlID) not in [1,2,4,5]: # sto dando il verde per almeno di 10 secondi ad una delle due direzioni
            costH, costV = self.getFlowCosts() 
            if (self.movingFlow == 'HORIZONTAL' and costH < costV) or (self.movingFlow == 'VERTICAL' and costV < costH): # valuta se è il caso di switchare
                self.switchTrafficLight()
