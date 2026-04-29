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
      <h2>Analisi Storica (Backtest)</h2>
      <p class="subtitle">Carica un file CSV storico per validare le prestazioni del modello sui dati passati.</p>
    
      <div class="upload-section">
        <label class="file-upload">
          <input type="file" (change)="onFileSelected($event)" accept=".csv" >
          <button (click)="runBacktest()" [disabled]="!selectedFile">
            Avvia Backtest
          </button>
        </label>
        <div class="filters-row">
          <input type="number" [(ngModel)]="limit" placeholder="Limit">
          <input type="date" [(ngModel)]="startDate" placeholder="Start Date">
          <input type="date" [(ngModel)]="endDate" placeholder="End Date">
          <button (click)="loadBacktestChart()">
            Carica Grafico
          </button>
        </div>      
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
    .subtitle { color: #94a3b8; margin-bottom: 2rem; }
    .upload-section { background: #1a1d24; padding: 2rem; border-radius: 8px; }
    button { 
      margin-left: 1rem; 
      padding: 0.5rem 1rem; 
      background: #3b82f6; 
      color: white; 
      border: none; 
      border-radius: 4px; 
      cursor: pointer;
    }
    button:disabled { background: #475569; cursor: not-allowed; }
    .chart-container {
      margin-top: 2rem;
      background: #1a1d24;
      padding: 1.5rem;
      border-radius: 8px;
      height: 650px; 
      width: 100%;
      overflow-x: auto;  /* Permette lo scroll orizzontale */
    }
    .chart-wrapper {
      width: 7000px;
      height: 600px;
    }
        .metrics-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr); /* 4 colonne pulite */
      gap: 1.5rem;
      margin-top: 2rem;
      padding: 0; /* Togliamo il padding del container */
    }

    .metrics-card {
      background: rgba(30, 41, 59, 0.5); /* Effetto vetro scuro */
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 1.25rem;
      display: flex;
      flex-direction: column; /* Label sopra, valore sotto per più eleganza */
      align-items: flex-start;
      transition: all 0.3s ease;
      position: relative;
      overflow: hidden;
    }

    /* Una piccola linea luminosa in cima alla card */
    .metrics-card::before {
      content: "";
      position: absolute;
      top: 0; left: 0; width: 100%; height: 3px;
      background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }

    .metrics-card:hover {
      transform: translateY(-5px);
      background: rgba(30, 41, 59, 0.8);
      border-color: rgba(59, 130, 246, 0.5);
      box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
    }

    .metrics-card .label {
      color: #64748b; /* Grigio bluastro discreto */
      font-size: 0.75rem;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      font-weight: 700;
      margin-bottom: 0.5rem;
    }

    .metrics-card .value {
      color: #f8fafc;
      font-size: 1.75rem;
      font-weight: 800;
      font-family: 'Inter', sans-serif; /* Se hai un font moderno */
      background: linear-gradient(to right, #fff, #94a3b8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .filters-row {
      margin-top: 1rem; 
      padding-top: 1rem; 
      border-top: 1px solid rgba(255,255,255,0.1); 
      display: flex; 
      gap: 1rem; 
      align-items: center;
      flex-wrap: wrap;
    }
    .filters-row input {
      background: rgba(30, 41, 59, 0.5); /* Stessa estetica del resto */
      border: 1px solid #334155; 
      color: #fff; 
      padding: 0.5rem; 
      border-radius: 4px; 
      min-width: 150px;
      flex: 1;
      max-width: 200px;
    }

  `]
})
export class BacktestComponent {
  selectedFile: File | null = null;
  metrics: any = null;
  limit?: number;
  startDate?: string;
  endDate?: string;

  @ViewChild('backtestChart')
  backtestChart!: ElementRef;
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

    this.apiService.runBacktest(this.selectedFile)
      .subscribe({
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
            data: data.actual_rounded, // Usiamo i nomi di Postman!
            borderColor: 'rgba(75, 192, 192, 1)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            fill: false,
            tension: 0.4,
            cubicInterpolationMode: 'monotone'
          },
          {
            label: 'Prediction',
            data: data.prediction_rounded,
            borderColor: 'rgba(153, 102, 255, 1)',
            backgroundColor: 'rgba(153, 102, 255, 0.2)',
            fill: false,
            tension: 0.4,
            cubicInterpolationMode: 'monotone'
          },
          {
            label: 'Diff',
            data: data.diff_rounded_instances,
            borderColor: 'rgba(255, 159, 64, 1)',
            backgroundColor: 'rgba(255, 159, 64, 0.2)',
            fill: true,
            tension: 0.4,
            cubicInterpolationMode: 'monotone'
          },
          {
            label: 'Prediction P70',
            data: data.prediction_p70_rounded,
            borderColor: 'rgba(255, 99, 132, 1)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            borderDash: [5, 5], // La facciamo tratteggiata per non confonderla
            fill: false,
            tension: 0.4,
            cubicInterpolationMode: 'monotone'
          }
        ]
      },
      options: {
        devicePixelRatio: window.devicePixelRatio || 2,
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: true, labels: { color: '#ffffff' } },
          title: { display: true, text: 'Actual vs Prediction', color: '#ffffff' }
        },
        scales: {
          x: {
            display: true,
            title: { display: true, text: 'Time', color: '#ffffff' },
            ticks: { color: '#cccccc', maxRotation: 45, minRotation: 45 },
            grid: {
              color: 'rgba(255, 255, 255, 0.1)'
            }
          },
          y: {
            display: true,
            title: { display: true, text: 'Value', color: '#ffffff' },
            ticks: { color: '#cccccc' },
            grid: { color: 'rgba(255, 255, 255, 0.1)' }
          }
        },
        elements: {
          line: { tension: 0.4 } // Questo riporta le "curve" morbide del tuo grafico originale
        }
      }

    });
  }
} 
