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
        scale_down_penalty: float = 1.0,
        max_derivative: float = 100.0,
        acceleration_factor: float = 1.2
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.derivative_exp = derivative_exp
        self.scale_down_penalty = scale_down_penalty
        self.max_derivative = max_derivative
        self.acceleration_factor = acceleration_factor
        
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, setpoint: float, current_value: float, dt: float = 1.0, min_val: float = 0.0, max_val: float = 2000.0) -> float:
        # Errore per autoscaler: quanto la realtà (PV) supera la previsione (Setpoint)
        # Se realtà > forecast -> errore positivo -> il PID aggiunge capacità
        error = current_value - setpoint
        
        # Penalizzazione asimmetrica (rallenta lo scale-down se < 1)
        if error < 0:
            effective_error = error * self.scale_down_penalty
        else:
            effective_error = error
            
        # Proporzionale
        p_term = self.kp * effective_error
        
        # Integrale con Anti-windup (Clamping)
        self.integral += effective_error * dt
        
        # Limitiamo l'integrale affinché non superi il range totale di manovra
        # Questo evita che il termine I accumuli drift infiniti in saturazione
        i_limit = (max_val - min_val)
        self.integral = float(np.clip(self.integral, -i_limit, i_limit))
        
        i_term = self.ki * self.integral
        
        # Derivata non lineare
        # L'esponente (derivative_exp) trasforma la risposta:
        # - exp = 1: risposta lineare standard
        # - exp > 1: sopprime il rumore (piccole derivate) e amplifica i picchi (grandi derivate)
        derivative = (effective_error - self.prev_error) / dt
        
        if derivative > 0:
            # In salita applichiamo l'acceleration_factor per una reattività massima
            safe_derivative = min(derivative, self.max_derivative) 
            nl_derivative = safe_derivative ** (self.derivative_exp * self.acceleration_factor)
        else:
            # In discesa usiamo l'esponente standard per uno scale-down più controllato
            safe_derivative = max(derivative, -self.max_derivative)
            nl_derivative = np.sign(safe_derivative) * (np.abs(safe_derivative) ** self.derivative_exp)
            
        d_term = self.kd * nl_derivative
        
        # Output delta
        correction = p_term + i_term + d_term
        
        # Aggiornamento stato
        self.prev_error = effective_error
        
        # Suggerimento basato sul setpoint (Forecast) + correzione
        suggested_value = setpoint + correction
        return float(np.clip(suggested_value, min_val, max_val))
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
