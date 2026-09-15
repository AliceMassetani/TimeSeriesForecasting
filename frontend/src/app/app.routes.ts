import { Routes } from '@angular/router';
import { authGuard } from './guards/auth.guard';

export const routes: Routes = [
  { path: '', redirectTo: 'forecast', pathMatch: 'full' },
  { 
    path: 'login', 
    loadComponent: () => import('./pages/auth/auth.component').then(m => m.AuthComponent) 
  },
  { 
    path: 'forecast', 
    canActivate: [authGuard],
    loadComponent: () => import('./pages/forecast/forecast.component').then(m => m.ForecastComponent) 
  },
  { 
    path: 'backtest', 
    canActivate: [authGuard],
    loadComponent: () => import('./pages/backtest/backtest.component').then(m => m.BacktestComponent) 
  },
  { 
    path: 'training', 
    canActivate: [authGuard],
    loadComponent: () => import('./pages/training/training.component').then(m => m.TrainingComponent) 
  }
];
