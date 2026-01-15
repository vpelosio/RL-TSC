import numpy as np

class DriverProfile:
    def __init__(self, tau, sigma, aggressivity, speedLimitComplianceFactor):
        # tau: adjust the distance while moving
        self.tau = tau

        # sigma: driver imperfection
        self.sigma = sigma

        self.aggressivity = aggressivity

        # speedLimitComplianceFactor: speed limit multiplier
        self.speedLimitComplianceFactor = speedLimitComplianceFactor

    @staticmethod
    def _clamp(val, min_val, max_val):
        return max(min_val, min(max_val, val))

    @staticmethod
    def generateRandom():
        # driver aggressiveness from 0 to 1
        raw_agg = np.random.normal(loc=0.5, scale=0.2, size=1)[0]
        aggressivity = DriverProfile._clamp(raw_agg, 0.0, 1.0)

        #  tau based on aggressiveness and 2-second rule 
        # Rule 126
        # https://www.gov.uk/guidance/the-highway-code/general-rules-techniques-and-advice-for-all-drivers-and-riders-103-to-158 
        tau = 2.0 - (aggressivity * 1.45) 
        # Note: in simulations with 0.5s steps, the minimum must be 0.55 (keeping a little margin), otherwise accidents will occur.

        # speedLimitComplianceFactor is derived from aggressiveness
        # The more aggressive a driver is, the faster they go.
        speedLimitComplianceFactor = 0.95 + (aggressivity * 0.30)
        # Let's start from 0.95 to simulate ECE Regulation 39 (the speed shown on the speedometer is always lower than the actual speed).

        # driver imperfection: sigma
        # default 0.5 for sumo, I create a normal distribution around 0.5
        sigma = round(DriverProfile._clamp(np.random.normal(0.5, 0.1), 0.0, 1.0), 2)

        return DriverProfile(tau, sigma, aggressivity, speedLimitComplianceFactor)