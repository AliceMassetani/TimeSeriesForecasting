import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable, tap } from 'rxjs';
import { Router } from '@angular/router';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private apiUrl = '/api/auth';
  private tokenKey = 'jwt_token';
  
  // BehaviorSubject per gestire lo stato in modo reattivo
  private loggedInSubject = new BehaviorSubject<boolean>(this.hasToken());
  isLoggedIn$ = this.loggedInSubject.asObservable();

  constructor(private http: HttpClient, private router: Router) { }

  private hasToken(): boolean {
    return !!localStorage.getItem(this.tokenKey);
  }

  isLoggedIn(): boolean {
    return this.loggedInSubject.value;
  }

  getToken(): string | null {
    return localStorage.getItem(this.tokenKey);
  }

  register(userData: any): Observable<any> {
    return this.http.post(`${this.apiUrl}/register`, userData).pipe(
      // Dopo la registrazione, logghiamo automaticamente l'utente sfruttando la password appena inserita
      tap(() => {
        console.log('Registrazione completata con successo');
      })
    );
  }

  login(userData: any): Observable<any> {
    return this.http.post<{ access_token: string, token_type: string }>(`${this.apiUrl}/login`, userData).pipe(
      tap(response => {
        if (response.access_token) {
          localStorage.setItem(this.tokenKey, response.access_token);
          this.loggedInSubject.next(true);
        }
      })
    );
  }

  logout(): void {
    localStorage.removeItem(this.tokenKey);
    this.loggedInSubject.next(false);
    this.router.navigate(['/login']);
  }
}
