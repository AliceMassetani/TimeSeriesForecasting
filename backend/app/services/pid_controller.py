import numpy as np

class NonLinearPID:
    """
    Implementazione di un controller PID Non-lineare.
    Supporta:
    - Penalizzazione asimmetrica per lo scale-down.
    - Derivata non lineare per reagire con forza a trend significativi (picchi).
    """
    def __init__(
        self, 
        kp: float = 0.5, 
        ki: float = 0.1, 
        kd: float = 0.2, 
        derivative_exp: float = 1.0, 
        scale_down_penalty: float = 1.0
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.derivative_exp = derivative_exp
        self.scale_down_penalty = scale_down_penalty
        
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, setpoint: float, current_value: float, dt: float = 1.0, min_val: float = 0.0, max_val: float = 2000.0) -> float:
        error = setpoint - current_value
        
        # Penalizzazione asimmetrica (rallenta lo scale-down se < 1)
        if error < 0:
            effective_error = error * self.scale_down_penalty
        else:
            effective_error = error
            
        # Proporzionale
        p_term = self.kp * effective_error
        
        # Integrale
        self.integral += effective_error * dt
        i_term = self.ki * self.integral
        
        # Derivata non lineare (potenziata sui picchi in salita)
        derivative = (effective_error - self.prev_error) / dt
        if derivative > 0:
            # Protezione overflow: limitiamo la derivata prima dell'elevamento a potenza
            safe_derivative = min(derivative, 100.0) # Tetto massimo di variazione
            nl_derivative = safe_derivative ** (self.derivative_exp * 1.2)
        else:
            safe_derivative = max(derivative, -100.0)
            nl_derivative = np.sign(safe_derivative) * (np.abs(safe_derivative) ** self.derivative_exp)
            
        d_term = self.kd * nl_derivative
        
        # Output delta
        correction = p_term + i_term + d_term
        
        # Aggiornamento stato
        self.prev_error = effective_error
        
        # Suggerimento basato sul setpoint (Forecast) + correzione
        # Questo permette di "anticipare" il forecast se il trend è positivo
        suggested_value = setpoint + correction
        return float(np.clip(suggested_value, min_val, max_val)) # Assicura che il risultato finale non sia mai negativo e non superi mai un tetto massimo di sicurezza
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
