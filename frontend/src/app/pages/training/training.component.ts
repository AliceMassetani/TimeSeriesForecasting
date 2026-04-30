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
        
        <div class="flex-row flex-gap-1-5">
          <label class="file-upload">
            <input type="file" (change)="onFileSelected($event)" accept=".csv" style="display: none;">
            <span class="upload-btn">Sfoglia CSV</span>
          </label>
          <button 
            class="primary-btn" 
            (click)="startTraining()" 
            [disabled]="!selectedFile || isTraining()">
            {{ isTraining() ? 'In corso...' : 'Avvia Training' }}
          </button>
          
          @if (selectedFile) {
            <span class="subtitle" style="margin-top: 0;">File: {{ selectedFile.name }}</span>
          }
        </div>

        @if (isTraining()) {
          <div class="flex-row mt-1-5 text-primary">
            <div class="spinner"></div>
            <span style="font-weight: 500;">Addestramento TSMixer in corso... L'operazione può richiedere alcuni secondi.</span>
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
      </div>

      @if (promotionSuccess()) {
        <div class="success-alert">
          <span class="success-icon">✓</span>
          <div>
            <strong>Operazione eseguita con successo!</strong>
            <p>Le impostazioni dei modelli sono state aggiornate correttamente.</p>
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
  trainingResult = signal<TrainingResult | null>(null);
  promotionSuccess = signal(false);
  activeModel = signal<'champion' | 'candidate'>('champion');

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

    this.apiService.trainModel(this.selectedFile).subscribe({
      next: (res: any) => {
        console.log('--- TRAINING COMPLETATO ---', res);
        this.trainingResult.set(res);
        this.isTraining.set(false);
      },
      error: (err: any) => {
        console.error('--- ERRORE TRAINING ---', err);
        this.isTraining.set(false);
        alert('Errore training.');
      }
    });
  }

  switchToChampion() {
    this.apiService.loadChampion().subscribe({
      next: () => {
        this.activeModel.set('champion');
        console.log('Champion attivato');
      },
      error: (err: any) => console.error(err)
    });
  }

  switchToCandidate() {
    this.apiService.loadCandidate().subscribe({
      next: () => {
        this.activeModel.set('candidate');
        console.log('Candidato attivato');
      },
      error: (err: any) => console.error(err)
    });
  }

  promoteModel() {
    if (!confirm('Confermi la promozione?')) return;
    this.isPromoting.set(true);
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
      }
    });
  }

  rollbackModel() {
    if (!confirm('Vuoi davvero ripristinare?')) return;
    this.isPromoting.set(true);
    this.apiService.rollbackModel().subscribe({
      next: () => {
        this.promotionSuccess.set(true);
        this.activeModel.set('champion');
        this.isPromoting.set(false);
      },
      error: (err: any) => {
        console.error(err);
        this.isPromoting.set(false);
      }
    });
  }
}
