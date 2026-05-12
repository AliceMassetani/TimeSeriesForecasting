/**
 * Contratti per i dati che arrivano dal Backend FastAPI
 */

export interface BacktestMetrics {
  rmse: number;
  mse: number;
  mae: number;
  r2: number;
}

export interface ForecastChart {
  labels: string[];
  prediction: number[];
  prediction_p10: number[];
  prediction_p90: number[];
  prediction_rounded: number[];
  prediction_p10_rounded: number[];
  prediction_p90_rounded: number[];
  prediction_pid?: number[];
  prediction_pid_rounded?: number[];
  prediction_kalman?: number[];
  prediction_kalman_rounded?: number[];
  actual: (number | null)[];
  actual_rounded?: (number | null)[];
}

export interface BacktestChart {
  labels: string[];
  actual: number[];
  prediction: number[];
  prediction_p10: number[];
  prediction_p90: number[];
  diff_instances: number[];
  prediction_rounded: number[];
  prediction_p10_rounded: number[];
  prediction_p90_rounded: number[];
  actual_rounded: number[];
  diff_rounded_instances: number[];
  metrics?: BacktestMetrics;
}

export interface TrainingResult {
  status: string;
  message: string;
  records_trained?: number;
  metrics?: BacktestMetrics;
}
