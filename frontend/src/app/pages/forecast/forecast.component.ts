import { Component, OnInit, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../services/api.service';
import { ForecastChart } from '../../models/api-data.model';
import { Chart, registerables } from 'chart.js';

Chart.register(...registerables);

@Component({
  selector: 'app-forecast',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="forecast-container">
      <div class="card-header">
        <h2>Live Forecast</h2>
        <p class="subtitle">Previsioni della capacità In-Use per le prossime ore</p>
      </div>

      <div class="chart-wrapper">
        <canvas #forecastChart></canvas>
      </div>

      <div class="legend-custom">
        <div class="legend-item"><span class="dot p50"></span> Previsione Media (P50)</div>
        <div class="legend-item"><span class="dot p70"></span> Previsione Cautelativa (P70)</div>
        <div class="legend-item"><span class="dot actual"></span> Dati Reali</div>
      </div>
    </div>
  `,
  styles: [`
    .forecast-container {
      background: #1a1d24;
      border-radius: 12px;
      padding: 2rem;
      box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .card-header { margin-bottom: 2rem; }
    .subtitle { color: #94a3b8; font-size: 0.9rem; }
    .chart-wrapper {
      position: relative;
      height: 400px;
      width: 100%;
    }
    .legend-custom {
      display: flex;
      gap: 2rem;
      margin-top: 1.5rem;
      justify-content: center;
      font-size: 0.85rem;
      color: #94a3b8;
    }
    .legend-item { display: flex; align-items: center; gap: 0.5rem; }
    .dot { width: 10px; height: 10px; border-radius: 2px; }
    .dot.p50 { background: #3b82f6; }
    .dot.p70 { background: #60a5fa; border: 1px dashed #fff; }
    .dot.actual { background: #ffffff; }
  `]
})
export class ForecastComponent implements OnInit, AfterViewInit {
  @ViewChild('forecastChart') chartCanvas!: ElementRef;
  chart: any;

  constructor(private apiService: ApiService) {}

  ngOnInit() { }

  ngAfterViewInit() {
    this.apiService.getForecastHistory().subscribe({
      next: (data: ForecastChart) => this.createChart(data),
      error: (err: any) => console.error('Errore dati forecast:', err)
    });
  }

  createChart(data: ForecastChart) {
    const ctx = this.chartCanvas.nativeElement.getContext('2d');
    
    if (this.chart) {
      this.chart.destroy();
    }

    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.labels,
        datasets: [
          {
            label: 'Reale',
            data: data.actual,
            borderColor: '#ffffff',
            borderWidth: 2,
            pointRadius: 3,
            fill: false,
            tension: 0.3
          },
          {
            label: 'Previsione (P50)',
            data: data.prediction,
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            borderWidth: 3,
            pointRadius: 0,
            fill: true,
            tension: 0.3
          },
          {
            label: 'Previsione Cautelativa (P70)',
            data: data.prediction_p70,
            borderColor: '#60a5fa',
            borderDash: [5, 5],
            borderWidth: 2,
            pointRadius: 0,
            fill: false,
            tension: 0.3
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8' }
          },
          x: {
            grid: { display: false },
            ticks: { color: '#94a3b8', autoSkip: true, maxTicksLimit: 10 }
          }
        }
      }
    });
  }
}
