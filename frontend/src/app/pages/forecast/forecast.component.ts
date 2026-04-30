import { Component, OnInit, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../services/api.service';
import { ForecastChart } from '../../models/api-data.model';
import { Chart } from 'chart.js/auto';

@Component({
  selector: 'app-forecast',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="forecast-container">
      <div class="page-header">
        <h2>Previsione Live (Forecast)</h2>
        <p class="subtitle">Capacità In-Use prevista per le prossime ore rispetto ai dati reali recenti.</p>
      </div>

      <div class="legend-custom">
        <div class="legend-item"><span class="dot actual"></span> Dati Reali</div>
        <div class="legend-item"><span class="dot p50"></span> Previsione (P50)</div>
        <div class="legend-item"><span class="dot p70"></span> Previsione Cautelativa (P70)</div>
      </div>

      <div class="chart-container">
        <div class="chart-wrapper">
          <canvas #forecastChart></canvas>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .forecast-container { padding: 1rem; }
    .dot.p50 { background: rgba(153, 102, 255, 1); }
    .dot.p70 { background: rgba(255, 99, 132, 1); }
    .dot.actual { background: rgba(75, 192, 192, 1); }
  `]
})
export class ForecastComponent implements OnInit, AfterViewInit {
  @ViewChild('forecastChart') chartCanvas!: ElementRef;
  chart: any;

  constructor(private apiService: ApiService) { }

  ngOnInit() { }

  ngAfterViewInit() {
    this.apiService.getForecastHistory().subscribe({
      next: (data: ForecastChart) => this.createChart(data),
      error: (err: any) => console.error('Errore dati forecast:', err)
    });
  }

  private createChart(data: ForecastChart) {
    const ctx = this.chartCanvas.nativeElement.getContext('2d');
    if (this.chart) this.chart.destroy();

    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.labels,
        datasets: [
          {
            label: 'Dati Reali',
            data: data.actual_rounded || data.actual,
            borderColor: 'rgba(75, 192, 192, 1)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            fill: false,
            tension: 0.4,
            cubicInterpolationMode: 'monotone',
            pointRadius: 4,
            pointBackgroundColor: 'rgba(75, 192, 192, 1)'
          },
          {
            label: 'Previsione (P50)',
            data: data.prediction_rounded || data.prediction,
            borderColor: 'rgba(153, 102, 255, 1)',
            backgroundColor: 'rgba(153, 102, 255, 0.2)',
            fill: false,
            tension: 0.4,
            cubicInterpolationMode: 'monotone',
            pointRadius: 0
          },
          {
            label: 'Previsione Cautelativa (P70)',
            data: data.prediction_p70_rounded || data.prediction_p70,
            borderColor: 'rgba(255, 99, 132, 1)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            fill: false,
            tension: 0.4,
            cubicInterpolationMode: 'monotone',
            pointRadius: 0
          }
        ]
      },
      options: {
        devicePixelRatio: window.devicePixelRatio || 2,
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8' },
            title: { display: true, text: 'Capacità (Istanze)', color: '#ffffff' }
          },
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', maxRotation: 45, minRotation: 45 },
            title: { display: true, text: 'Tempo', color: '#ffffff' }
          }
        }
      }
    });
  }
}
