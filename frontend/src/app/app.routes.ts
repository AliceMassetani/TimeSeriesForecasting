import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', redirectTo: 'forecast', pathMatch: 'full' },
  { 
    path: 'forecast', 
    loadComponent: () => import('./pages/forecast/forecast.component').then(m => m.ForecastComponent) 
  },
  { 
    path: 'backtest', 
    loadComponent: () => import('./pages/backtest/backtest.component').then(m => m.BacktestComponent) 
  },
  { 
    path: 'training', 
    loadComponent: () => import('./pages/training/training.component').then(m => m.TrainingComponent) 
  }
];
