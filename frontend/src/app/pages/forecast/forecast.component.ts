import { Component, OnInit, ElementRef, signal, viewChild } from '@angular/core';
import { ApiService } from '../../services/api.service';
import { ForecastChart } from '../../models/api-data.model';
import { Chart } from 'chart.js/auto';

@Component({
  selector: 'app-forecast',
  standalone: true,
  imports: [],
  template: `
    <div class="page-container">
      <div class="page-header">
        <h2>Previsione Live (Forecast)</h2>
        <p class="subtitle">Capacità In-Use prevista per le prossime ore rispetto ai dati reali recenti.</p>
      </div>

      @if (hasData()) {
        <div class="legend-custom">
          <div class="legend-item"><span class="dot actual"></span> Dati Reali</div>
          <div class="legend-item"><span class="dot p50"></span> Previsione (P50)</div>
          <div class="legend-item"><span class="dot p70"></span> Previsione Cautelativa (P70)</div>
        </div>
      }

      <div class="chart-container">
        @if (isLoading()) {
          <div class="flex-row" style="height: 100%; justify-content: center;">
             <div class="spinner"></div>
             <span class="text-primary">Caricamento previsioni...</span>
          </div>
        }
        
        <div class="chart-wrapper" [hidden]="isLoading() || !hasData()">
          <canvas #forecastChart></canvas>
        </div>
      </div>
    </div>
  `,
  styles: []
})
export class ForecastComponent implements OnInit {
  isLoading = signal(true);
  hasData = signal(false);
  
  chartCanvas = viewChild<ElementRef<HTMLCanvasElement>>('forecastChart');
  chart: any;

  constructor(private apiService: ApiService) { }

  ngOnInit() {
    this.fetchForecast();
  }

  fetchForecast() {
    this.isLoading.set(true);
    this.apiService.getForecastHistory().subscribe({
      next: (data: ForecastChart) => {
        this.hasData.set(true);
        this.isLoading.set(false);
        setTimeout(() => this.createChart(data), 0);
      },
      error: (err: any) => {
        console.error('Errore dati forecast:', err);
        this.isLoading.set(false);
      }
    });
  }

  private createChart(data: ForecastChart) {
    const canvas = this.chartCanvas()?.nativeElement;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

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
