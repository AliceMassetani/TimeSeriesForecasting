import numpy as np

class KalmanFilter:
    """
    Semplice Filtro di Kalman Unidimensionale per la stima del Bias di un modello.
    Mantiene uno stato (il bias stimato) e un'incertezza (varianza).
    """
    def __init__(self, process_variance: float = 0.5, measurement_variance: float = 0.001):
        # Parametri per una reazione ultra-rapida
        self.q = process_variance
        self.r = measurement_variance
        
        self.state = 0.0      
        self.covariance = 1.0 

    def update(self, measurement: float) -> float:
        # LOGICA 'FLOOR PROTECTION'
        # Impediamo alla correzione di 'affondare' il modello originale
        
        # Se mancano risorse, resettiamo immediatamente qualsiasi bias negativo
        if measurement > 0 and self.state < 0:
            self.state = 0.0

        if measurement > self.state:
            # SOTTODIMENSIONAMENTO: Reazione massima (1.0) per annullare il lag
            kalman_gain = 1.0  
        else:
            # SOVRADIMENSIONAMENTO: Rientro controllato (0.5)
            kalman_gain = 0.5 

        # Aggiornamento dello stato
        self.state = self.state + kalman_gain * (measurement - self.state)
        
        # PROTEZIONE FINALE: Il bias non può mai scendere sotto -0.5
        # Questo garantisce che la curva corretta non sia MAI significativamente 
        # più bassa della fascia P90 scelta dall'utente.
        self.state = max(self.state, -0.5)
        
        return self.state

    def reset(self, initial_state: float = 0.0):
        self.state = initial_state
        self.covariance = 1.0
