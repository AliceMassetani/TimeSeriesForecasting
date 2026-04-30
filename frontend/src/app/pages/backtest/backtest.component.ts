import { Component, ViewChild, ElementRef, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { Chart } from 'chart.js/auto';

@Component({
  selector: 'app-backtest',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="backtest-container">
      <div class="page-header">
        <h2>Analisi Storica (Backtest)</h2>
      <p class="subtitle">Carica un file CSV storico per validare le prestazioni del modello sui dati passati.</p>
      </div>
    
      <div class="upload-section">
        <label class="file-upload">
          <input type="file" (change)="onFileSelected($event)" accept=".csv" >
          <button class="primary-btn" (click)="runBacktest()" [disabled]="!selectedFile">
            Avvia Backtest
          </button>
        </label>
        
        <div class="filters-row">
          <input type="number" [(ngModel)]="limit" placeholder="Numero di punti">
          <input type="datetime-local" [(ngModel)]="startDate">
          <input type="datetime-local" [(ngModel)]="endDate">
          <button class="secondary-btn" (click)="loadBacktestChart()">Vedi grafico</button>
        </div>      
      </div>

      <div class="legend-custom" *ngIf="hasData">
        <div class="legend-item"><span class="dot actual"></span> Reale</div>
        <div class="legend-item"><span class="dot p50"></span> Previsione</div>
        <div class="legend-item"><span class="dot p70"></span> Previsione P70</div>
        <div class="legend-item"><span class="dot diff"></span> Differenza</div>
      </div>

      <div class="chart-container" [hidden]="!hasData">
        <div class="chart-wrapper" [hidden]="!hasData">
          <canvas #backtestChart></canvas>
        </div>
      </div>
      
      <div class="metrics-grid" *ngIf="metrics">
        <div class="metrics-card">
          <span class="label">MSE</span>
          <span class="value">{{ metrics.mse | number: '1.3-3' }}</span>
        </div>        
        <div class="metrics-card">
          <span class="label">RMSE</span>
          <span class="value">{{ metrics.rmse | number: '1.3-3' }}</span>
        </div> 
        <div class="metrics-card"> 
          <span class="label">MAE</span>
          <span class="value">{{ metrics.mae | number: '1.3-3' }}</span>
        </div>
        <div class="metrics-card">
          <span class="label">R²</span>
          <span class="value">{{ metrics.r2 | number: '1.3-3' }}</span>
        </div>        
      </div>
    </div>
  `,
  styles: [`
    .backtest-container { color: #fff; padding: 2rem; }
    .upload-section { background: #1a1d24; padding: 2rem; border-radius: 8px; border: 1px solid #334155; }
    
    .primary-btn { 
      margin-left: 1rem; padding: 0.5rem 1rem; background: #3b82f6; 
      color: white; border: none; border-radius: 4px; cursor: pointer;
    }
    .secondary-btn { 
      padding: 0.5rem 1rem; background: transparent; color: #3b82f6; 
      border: 1px solid #3b82f6; border-radius: 4px; cursor: pointer;
    }
    
    .filters-row {
      margin-top: 1.5rem; display: flex; gap: 1rem; align-items: center; flex-wrap: wrap;
      padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.1);
    }
    .filters-row input {
      background: #0f172a; border: 1px solid #334155; color: white; padding: 0.5rem; border-radius: 4px;
    }
    .dot.p50 { background: rgba(153, 102, 255, 1); }
    .dot.p70 { background: rgba(255, 99, 132, 1); }
    .dot.actual { background: rgba(75, 192, 192, 1); }
    .dot.diff { background: rgba(255, 159, 64, 1); }
  `]
})
export class BacktestComponent implements OnInit {
  selectedFile: File | null = null;
  metrics: any = null;
  limit?: number;
  startDate?: string;
  endDate?: string;

  @ViewChild('backtestChart') backtestChart!: ElementRef;
  chart: any;
  hasData: boolean = false;

  constructor(private apiService: ApiService) { }

  ngOnInit() {
    this.loadBacktestChart();
  }

  onFileSelected(event: Event) {
    const file = event.target as HTMLInputElement;
    this.selectedFile = file.files?.[0] || null;
    console.log('file selezionato:', this.selectedFile);
  }

  runBacktest() {
    if (!this.selectedFile) return;
    console.log('Avvio backtest con file:', this.selectedFile?.name);
    this.apiService.runBacktest(this.selectedFile).subscribe({
      next: (response: any) => {
        this.metrics = response.metrics;
        this.loadBacktestChart();
        console.log('Backtest completato con successo:', response);
        },
        error: (error: any) => {
          console.error('Errore durante il backtest:', error);
      }
      })
  }

  loadBacktestChart() {
    this.apiService.getBacktestHistory(this.limit, this.startDate, this.endDate).subscribe({
      next: (data: any) => {
        this.metrics = data.metrics;
        this.createChart(data);
        console.log('Dati backtest caricati con successo:', data);
      },
      error: (error: any) => {
        console.error('Errore durante il caricamento del backtest:', error);
      }
    })
  }

  private createChart(data: any) {
    this.hasData = true;
    const context = this.backtestChart.nativeElement.getContext('2d');

    if (this.chart) this.chart.destroy(); // distrugge il vecchio grafico prima di farne uno nuovo

    this.chart = new Chart(context, {
      type: 'line',
      data: {
        labels: data.labels,
        datasets: [
          {
            label: 'Actual',
            data: data.actual_rounded,
            borderColor: 'rgba(75, 192, 192, 1)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            fill: false,
            tension: 0.4,
            cubicInterpolationMode: 'monotone',
            pointRadius: 4
          },
          {
            label: 'Prediction',
            data: data.prediction_rounded,
            borderColor: 'rgba(153, 102, 255, 1)',
            backgroundColor: 'rgba(153, 102, 255, 0.2)',
            fill: false,
            tension: 0.4,
            cubicInterpolationMode: 'monotone',
            pointRadius: 2
          },
          {
            label: 'Diff',
            data: data.diff_rounded_instances,
            borderColor: 'rgba(255, 159, 64, 1)',
            backgroundColor: 'rgba(255, 159, 64, 0.2)',
            fill: true,
            tension: 0.4,
            cubicInterpolationMode: 'monotone',
            pointRadius: 2
          },
          {
            label: 'Prediction P70',
            data: data.prediction_p70_rounded,
            borderColor: 'rgba(255, 99, 132, 1)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            //borderDash: [5, 5], // La facciamo tratteggiata per non confonderla
            fill: false,
            tension: 0.4,
            cubicInterpolationMode: 'monotone',
            pointRadius: 2
          }
        ]
      },
      options: {
        devicePixelRatio: window.devicePixelRatio || 2,
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          title: { display: true, text: 'Analisi Backtest', color: '#ffffff' }
        },
        scales: {
          x: { 
            ticks: { color: '#cccccc', maxRotation: 45, minRotation: 45 },
            grid: { color: 'rgba(255, 255, 255, 0.05)' }
          },
          y: { 
            ticks: { color: '#cccccc' },
            grid: { color: 'rgba(255, 255, 255, 0.05)' }
          }
        }
      }
    });
  }
}
