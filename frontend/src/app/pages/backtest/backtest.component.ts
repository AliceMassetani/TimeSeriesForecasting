import { Component } from '@angular/core';

@Component({
  selector: 'app-backtest',
  standalone: true,
  template: `
    <div class="page-container">
      <h2>Analisi Backtest</h2>
      <p>Sezione dedicata al caricamento dei file CSV e alla validazione storica.</p>
    </div>
  `,
  styles: [`
    .page-container {
      color: #fff;
    }
    h2 { margin-bottom: 1rem; }
  `]
})
export class BacktestComponent {}
