import numpy as np

class DriverProfile:
    def __init__(self, tau, sigma, aggressivity, speedLimitComplianceFactor):
        # tau: regola la distanza in movimento
        self.tau = tau

        # sigma: imperfezione del guidatore
        self.sigma = sigma

        self.aggressivity = aggressivity

        # speedLimitComplianceFactor: moltiplicatore del limite 
        self.speedLimitComplianceFactor = speedLimitComplianceFactor

    @staticmethod
    def _clamp(val, min_val, max_val):
        return max(min_val, min(max_val, val))

    @staticmethod
    def generateRandom():
        # aggressitività del guidatore da 0 a 1
        raw_agg = np.random.normal(loc=0.5, scale=0.2, size=1)[0]
        aggressivity = DriverProfile._clamp(raw_agg, 0.0, 1.0)

        # tau basato su aggressività e regola 2 secondi 
        # Rule 126
        # https://www.gov.uk/guidance/the-highway-code/general-rules-techniques-and-advice-for-all-drivers-and-riders-103-to-158 
        tau = 2.0 - (aggressivity * 1.45) 
        # Nota: in simulazioni da step di 0.5s il minimo deve essere 0.55 (tenendo un po' di margine), altrimenti si creano incidenti

        # si deriva speedLimitComplianceFactor dall'aggressività
        # più un guidatore è aggressivo, più va forte
        speedLimitComplianceFactor = 0.95 + (aggressivity * 0.30)
        # partiamo da 0.95 per simulare UN ECE Regulation 39 (la velocità mostrata dal tachimetro è sempre inferiore a quella reale)

        # imperfezione del guidatore: sigma
        # default 0.5 per sumo, creo una distribuzione normale attorno a 0.5
        sigma = round(DriverProfile._clamp(np.random.normal(0.5, 0.1), 0.0, 1.0), 2)

        return DriverProfile(tau, sigma, aggressivity, speedLimitComplianceFactor)