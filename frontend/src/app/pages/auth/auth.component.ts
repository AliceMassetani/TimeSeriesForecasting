import { Component, inject, signal } from '@angular/core';
import { FormBuilder, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { AuthService } from '../../services/auth.service';

@Component({
  selector: 'app-auth',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './auth.component.html',
  styleUrl: './auth.component.css'
})
export class AuthComponent {
  private fb = inject(FormBuilder);
  private authService = inject(AuthService);
  private router = inject(Router);

  isLoginMode = signal(true);
  isLoading = signal(false);
  errorMessage = signal<string | null>(null);

  authForm: FormGroup = this.fb.group({
    username: ['', [Validators.required, Validators.minLength(3)]],
    password: ['', [Validators.required, Validators.minLength(6)]]
  });

  toggleMode() {
    this.isLoginMode.set(!this.isLoginMode());
    this.errorMessage.set(null);
    this.authForm.reset();
  }

  onSubmit() {
    if (this.authForm.invalid) return;

    this.isLoading.set(true);
    this.errorMessage.set(null);
    
    const { username, password } = this.authForm.value;

    if (this.isLoginMode()) {
      // LOGIN
      this.authService.login({ username, password }).subscribe({
        next: () => {
          this.isLoading.set(false);
          this.router.navigate(['/forecast']);
        },
        error: (err) => {
          this.isLoading.set(false);
          this.errorMessage.set(err.error?.detail || 'Credenziali non valide.');
        }
      });
    } else {
      // REGISTER
      this.authService.register({ username, password }).subscribe({
        next: () => {
          // Auto-login dopo registrazione con successo
          this.authService.login({ username, password }).subscribe({
            next: () => {
              this.isLoading.set(false);
              this.router.navigate(['/forecast']);
            },
            error: () => {
              this.isLoading.set(false);
              this.errorMessage.set('Registrazione completata, ma il login automatico ha fallito. Effettua il login manualmente.');
              this.isLoginMode.set(true);
            }
          });
        },
        error: (err) => {
          this.isLoading.set(false);
          this.errorMessage.set(err.error?.detail || 'Errore durante la registrazione.');
        }
      });
    }
  }
}
