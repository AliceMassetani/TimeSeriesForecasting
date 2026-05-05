import { Component, OnInit, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { TrainingResult } from '../../models/api-data.model';
import { DecimalPipe } from '@angular/common';

@Component({
  selector: 'app-training',
  standalone: true,
  imports: [FormsModule, DecimalPipe],
  template: `
    <div class="page-container">
      <div class="page-header">
        <h2>Gestione e Addestramento Modelli</h2>
        <p class="subtitle">Controlla il ciclo di vita dei modelli predittivi: addestramento, validazione e rilascio.</p>
      </div>

      <!-- SEZIONE 1: ADDESTRAMENTO -->
      <div class="upload-section">
        <h3>1. Addestra Nuovo Modello</h3>
        <p class="subtitle mb-1-5">Crea una nuova versione "Candidato" del modello utilizzando dati storici aggiornati.</p>
        
        <div class="flex-row flex-gap-1-5 flex-wrap">
          <label class="file-upload">
            <input type="file" (change)="onFileSelected($event)" accept=".csv" style="display: none;">
            <span class="upload-btn">Sfoglia CSV</span>
          </label>
          
          @if (selectedFile) {
            <span class="subtitle" style="margin-top: 0; align-self: center;">File: {{ selectedFile.name }}</span>
          }

          <button 
            class="primary-btn" 
            (click)="startTraining()" 
            [disabled]="!selectedFile || isTraining() || (params.n_epochs !== null && (params.n_epochs > 200 || params.n_epochs < 1))"
            style="margin-left: auto;">
            {{ isTraining() ? 'In corso...' : 'Avvia Training' }}
          </button>
        </div>

        <!-- PARAMETRI AVANZATI -->
        <details class="mt-2">
          <summary class="text-primary" style="cursor: pointer; font-weight: 600; font-size: 0.9rem;">
            Parametri Avanzati
          </summary>
          <div class="filters-row mt-1" style="background: rgba(255,255,255,0.02); padding: 1.5rem; border-radius: 8px;">
            <div class="filter-group">
              <label>Input Chunk</label>
              <select [(ngModel)]="params.input_chunk_len" class="sleek-field">
                <option [ngValue]="null">Default (168)</option>
                @for (p of powersOf2; track p) {
                  <option [ngValue]="p">{{ p }}</option>
                }
              </select>
            </div>

            <div class="filter-group">
              <label>Output Chunk</label>
              <select [(ngModel)]="params.output_chunk_len" class="sleek-field">
                <option [ngValue]="null">Default (8)</option>
                @for (p of powersOf2Small; track p) {
                  <option [ngValue]="p">{{ p }}</option>
                }
              </select>
            </div>

            <div class="filter-group">
              <label>Epoche</label>
              <input type="number" [(ngModel)]="params.n_epochs" 
                     min="0" max="200" step="10" 
                     (keydown.arrowup)="onEpochsArrow($event, 10)"
                     (keydown.arrowdown)="onEpochsArrow($event, -10)"
                     placeholder="Default (50)" style="width: 120px;">
              @if (params.n_epochs > 200) {
                <span class="text-danger" style="font-size: 0.7rem; font-weight: 600;">Max 200 consentite</span>
              } @else if (params.n_epochs !== null && params.n_epochs < 1) {
                <span class="text-danger" style="font-size: 0.7rem; font-weight: 600;">Minimo 1 epoca richiesta</span>
              }
            </div>

            <div class="filter-group">
              <label>Batch Size</label>
              <select [(ngModel)]="params.batch_size" class="sleek-field">
                <option [ngValue]="null">Default (32)</option>
                <option [ngValue]="16">16</option>
                <option [ngValue]="32">32</option>
                <option [ngValue]="64">64</option>
                <option [ngValue]="128">128</option>
              </select>
            </div>

            <div class="filter-group">
              <label>Hidden Size</label>
              <select [(ngModel)]="params.hidden_size" class="sleek-field">
                <option [ngValue]="null">Default (64)</option>
                <option [ngValue]="32">32</option>
                <option [ngValue]="64">64</option>
                <option [ngValue]="128">128</option>
              </select>
            </div>

            <div class="filter-group">
              <label>Num Blocks</label>
              <input type="number" [(ngModel)]="params.num_blocks" min="1" max="10" placeholder="Default (2)" style="width: 100px;">
            </div>

            <div class="filter-group">
              <label>Dropout</label>
              <input type="number" [(ngModel)]="params.dropout" min="0" max="0.9" step="0.05" placeholder="Default (0.1)" style="width: 100px;">
            </div>

            <div class="filter-group">
              <label>Learning Rate</label>
              <input type="number" [(ngModel)]="params.learning_rate" step="0.0001" placeholder="Default (0.001)" style="width: 140px;">
            </div>
          </div>
        </details>

        @if (isTraining()) {
          <div class="flex-row mt-1-5 text-primary">
            <div class="spinner"></div>
            <span style="font-weight: 500;">Addestramento TSMixer in corso... L'operazione può richiedere alcuni secondi.</span>
          </div>
        }

        <!-- RISULTATO TRAINING (CONTESTUALE) -->
        @if (trainingResult()) {
          <div class="alert-box success" style="margin-top: 1.5rem;">
            <span class="alert-icon">✓</span>
            <div class="alert-content">
              <strong>Addestramento Completato!</strong>
              <p>{{ trainingResult()?.message }}</p>
              
              <div class="metrics-grid" style="margin-top: 1rem; grid-template-columns: repeat(5, 1fr);">
                <div class="metrics-card" style="padding: 0.75rem; background: rgba(0,0,0,0.2);">
                  <span class="label" style="font-size: 0.6rem;">MAE</span>
                  <span class="value" style="font-size: 1.2rem;">{{ trainingResult()?.metrics?.mae | number:'1.3-3' }}</span>
                </div>
                <div class="metrics-card" style="padding: 0.75rem; background: rgba(0,0,0,0.2);">
                  <span class="label" style="font-size: 0.6rem;">MSE</span>
                  <span class="value" style="font-size: 1.2rem;">{{ trainingResult()?.metrics?.mse | number:'1.3-3' }}</span>
                </div>
                <div class="metrics-card" style="padding: 0.75rem; background: rgba(0,0,0,0.2);">
                  <span class="label" style="font-size: 0.6rem;">RMSE</span>
                  <span class="value" style="font-size: 1.2rem;">{{ trainingResult()?.metrics?.rmse | number:'1.3-3' }}</span>
                </div>
                <div class="metrics-card" style="padding: 0.75rem; background: rgba(0,0,0,0.2);">
                  <span class="label" style="font-size: 0.6rem;">R²</span>
                  <span class="value" style="font-size: 1.2rem;">{{ trainingResult()?.metrics?.r2 | number:'1.3-3' }}</span>
                </div>
                <div class="metrics-card" style="padding: 0.75rem; background: rgba(0,0,0,0.2);">
                  <span class="label" style="font-size: 0.6rem;">Record</span>
                  <span class="value" style="font-size: 1.2rem;">{{ trainingResult()?.records_trained }}</span>
                </div>
              </div>
            </div>
          </div>
        }

        <!-- ERRORE TRAINING (CONTESTUALE) -->
        @if (errorMessage()) {
          <div class="alert-box error" style="margin-top: 1.5rem;">
            <span class="alert-icon">✕</span>
            <div class="alert-content">
              <strong>Errore durante l'addestramento</strong>
              <p>{{ errorMessage() }}</p>
            </div>
          </div>
        }
      </div>

      <!-- SEZIONE 2: SELEZIONE MODELLO (TEST) -->
      <div class="upload-section">
        <h3>2. Selezione Modello per Test</h3>
        <p class="subtitle mb-1-5">Cambia il modello attivo in memoria per eseguire confronti e validazioni nelle pagine di Backtest.</p>
        
        <div class="flex-row flex-gap-1-5 flex-wrap">
          <button 
            class="secondary-btn" 
            [class.active]="activeModel() === 'champion'"
            (click)="switchToChampion()">
            Attiva Champion (Produzione)
          </button>
          
          <button 
            class="secondary-btn" 
            [class.active]="activeModel() === 'candidate'"
            (click)="switchToCandidate()">
            Attiva Candidato (Sfidante)
          </button>

          <div class="status-badge">
            <span class="status-label">In uso:</span>
            <span class="status-value" [class.text-primary]="activeModel() === 'champion'" [class.text-success]="activeModel() === 'candidate'">
              {{ activeModel() === 'champion' ? 'CHAMPION' : 'CANDIDATO' }}
            </span>
          </div>
        </div>

        @if (selectionError()) {
          <div class="alert-box error mt-1">
            <span class="alert-icon">✕</span>
            <div class="alert-content">
              <p>{{ selectionError() }}</p>
            </div>
          </div>
        }
      </div>

      <!-- SEZIONE 3: PROMOZIONE E ROLLBACK -->
      <div class="upload-section">
        <h3>3. Gestione della Produzione</h3>
        <p class="subtitle mb-1-5">Azioni definitive per il rilascio o il ripristino dei modelli.</p>
        
        <div class="explanation-box">
          <ul>
            <li>
              <strong>Promozione Nuovo Champion:</strong> 
              Questa azione rende il modello "Candidato" (l'ultimo addestrato) il modello ufficiale per la produzione. 
              Il vecchio modello Champion verrà conservato automaticamente come backup per sicurezza.
            </li>
            <li>
              <strong>Ripristino Versione Precedente (Rollback):</strong> 
              Annulla l'ultima promozione ripristinando il modello Champion dal backup. 
              Il modello attuale verrà declassato a "Candidato", permettendoti di analizzare cosa non ha funzionato.
            </li>
          </ul>
        </div>

        <div class="card-footer">
          <button class="secondary-btn border-danger text-danger" (click)="rollbackModel()" [disabled]="isPromoting()">
            Esegui Rollback
          </button>

          <button class="primary-btn" style="background: #10b981;" (click)="promoteModel()" [disabled]="isPromoting()">
            {{ isPromoting() ? 'Promozione...' : 'Conferma Promozione Champion' }}
          </button>
        </div>

        @if (promotionSuccess()) {
          <div class="alert-box success mt-1">
            <span class="alert-icon">✓</span>
            <div class="alert-content">
              <strong>Operazione eseguita con successo!</strong>
              <p>Le impostazioni dei modelli sono state aggiornate correttamente.</p>
            </div>
          </div>
        }

        @if (promotionError()) {
          <div class="alert-box error mt-1">
            <span class="alert-icon">✕</span>
            <div class="alert-content">
              <strong>Errore nell'operazione</strong>
              <p>{{ promotionError() }}</p>
            </div>
          </div>
        }
      </div>
  `,
  styles: []
})
export class TrainingComponent implements OnInit {
  selectedFile: File | null = null;
  
  isTraining = signal(false);
  isPromoting = signal(false);
  trainingResult = signal<any | null>(null);
  promotionSuccess = signal(false);
  errorMessage = signal<string | null>(null);
  promotionError = signal<string | null>(null);
  selectionError = signal<string | null>(null);
  activeModel = signal<'champion' | 'candidate'>('champion');

  // Parametri di training
  powersOf2 = [32, 64, 128, 256, 512, 1024];
  powersOf2Small = [4, 8, 16, 24, 32, 48, 64]; // Non tutti potenze di 2 strette, ma seguono i tuoi step tipici
  
  params: any = {
    input_chunk_len: null,
    output_chunk_len: null,
    n_epochs: null,
    batch_size: null,
    hidden_size: null,
    num_blocks: null,
    dropout: null,
    learning_rate: null
  };

  constructor(private apiService: ApiService) {}

  ngOnInit() {}

  onFileSelected(event: any) {
    this.selectedFile = event.target.files[0];
  }

  startTraining() {
    if (!this.selectedFile) return;
    console.log('--- AVVIO TRAINING ---');
    this.isTraining.set(true);
    this.trainingResult.set(null);
    this.promotionSuccess.set(false);
    this.errorMessage.set(null);

    this.apiService.trainModel(this.selectedFile, this.params).subscribe({
      next: (res: any) => {
        console.log('--- TRAINING COMPLETATO ---', res);
        this.trainingResult.set(res);
        this.isTraining.set(false);
      },
      error: (err: any) => {
        console.error('--- ERRORE TRAINING ---', err);
        this.isTraining.set(false);
        const detail = err.error?.detail || 'Errore imprevisto durante l\'addestramento.';
        this.errorMessage.set(detail);
      }
    });
  }

  switchToChampion() {
    this.selectionError.set(null);
    this.apiService.loadChampion().subscribe({
      next: () => {
        this.activeModel.set('champion');
        console.log('Champion attivato');
      },
      error: (err: any) => {
        const detail = err.error?.detail || 'Errore durante il caricamento del Champion.';
        this.selectionError.set(detail);
      }
    });
  }

  switchToCandidate() {
    this.selectionError.set(null);
    this.apiService.loadCandidate().subscribe({
      next: () => {
        this.activeModel.set('candidate');
        console.log('Candidato attivato');
      },
      error: (err: any) => {
        const detail = err.error?.detail || 'Errore durante il caricamento del Candidato.';
        this.selectionError.set(detail);
      }
    });
  }

  promoteModel() {
    if (!confirm('Confermi la promozione?')) return;
    this.isPromoting.set(true);
    this.promotionError.set(null);
    this.promotionSuccess.set(false);
    
    this.apiService.promoteModel().subscribe({
      next: () => {
        this.promotionSuccess.set(true);
        this.activeModel.set('champion');
        this.trainingResult.set(null);
        this.isPromoting.set(false);
      },
      error: (err: any) => {
        console.error(err);
        this.isPromoting.set(false);
        const detail = err.error?.detail || 'Errore durante la promozione del modello.';
        this.promotionError.set(detail);
      }
    });
  }

  onEpochsChange(val: number | null) {
    if (val !== null && (val < 1 || val > 200)) {
      this.errorMessage.set('Il numero di epoche deve essere compreso tra 1 e 200.');
    } else {
      this.errorMessage.set(null);
    }
  }

  rollbackModel() {
    if (!confirm('Vuoi davvero ripristinare?')) return;
    this.isPromoting.set(true);
    this.promotionError.set(null);
    this.promotionSuccess.set(false);
    
    this.apiService.rollbackModel().subscribe({
      next: () => {
        this.promotionSuccess.set(true);
        this.activeModel.set('champion');
        this.isPromoting.set(false);
      },
      error: (err: any) => {
        console.error(err);
        this.isPromoting.set(false);
        const detail = err.error?.detail || 'Errore durante il rollback del modello.';
        this.promotionError.set(detail);
      }
    });
  }

  onEpochsArrow(event: any, delta: number) {
    event.preventDefault();
    let current = this.params.n_epochs;
    
    // Se è null, partiamo dal default 50
    if (current === null || current === undefined) {
      current = 50;
    }

    let next: number;
    if (delta > 0) {
      // Arrotonda alla decina superiore
      next = Math.floor(current / 10) * 10 + 10;
    } else {
      // Arrotonda alla decina inferiore
      if (current % 10 === 0) {
        next = current - 10;
      } else {
        next = Math.floor(current / 10) * 10;
      }
    }

    // Applica limiti
    if (next <= 0) next = 1;
    this.params.n_epochs = Math.min(200, next);
  }
}
