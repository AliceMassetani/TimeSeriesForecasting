import { Component, ViewChild, ElementRef, OnInit} from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../services/api.service';
import { Chart } from 'chart.js/auto';

@Component({
  selector: 'app-backtest',
  standalone: true,
  imports: [CommonModule],
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
        <button (click)="loadBacktestChart()">
          Carica Grafico\
        </button>
      </div>

      <div class="chart-container">
        <div class="chart-wrapper" [hidden]="!hasData">
          <canvas #backtestChart></canvas>
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
      width: 10000px;
      height: 600px;
    }
  `]
})
export class BacktestComponent {
  selectedFile: File | null = null;

  @ViewChild('backtestChart') 
  backtestChart!: ElementRef;
  chart: any;
  hasData: boolean = false;
  constructor(private apiService: ApiService) {}

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
          console.log('Backtest completato con successo:', response);
          this.loadBacktestChart()
        },
        error: (error: any) => {
          console.error('Errore durante il backtest:', error);
        }
      })
  }

  loadBacktestChart() {
    this.apiService.getBacktestHistory().subscribe({
      next: (data: any) => {
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
        
    if(this.chart) this.chart.destroy(); // distrugge il vecchio grafico prima di farne uno nuovo

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
                grid: { color: 'rgba(255, 255, 255, 0.1)'
                }
            },
            y: { 
                display: true,
                title: { display: true, text: 'Value', color: '#ffffff'},
                ticks: { color: '#cccccc' },
                grid: { color: 'rgba(255, 255, 255, 0.1)' }
            }
        },
        elements: {
            line: {tension: 0.4 } // Questo riporta le "curve" morbide del tuo grafico originale
        }
    }

    });
  }
}
