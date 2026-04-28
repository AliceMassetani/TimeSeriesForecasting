import { Component } from '@angular/core';

@Component({
  selector: 'app-training',
  standalone: true,
  template: `
    <div class="page-container">
      <h2>Gestione Modelli</h2>
      <p>Pannello di controllo per addestramento, promozione e rollback dei modelli ML.</p>
    </div>
  `,
  styles: [`
    .page-container {
      color: #fff;
    }
    h2 { margin-bottom: 1rem; }
  `]
})
export class TrainingComponent {}
