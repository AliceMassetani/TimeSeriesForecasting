import { HttpInterceptorFn, HttpErrorResponse } from '@angular/common/http';
import { inject } from '@angular/core';
import { AuthService } from '../services/auth.service';
import { catchError } from 'rxjs/operators';
import { throwError } from 'rxjs';

export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const authService = inject(AuthService);
  const token = authService.getToken();

  // Non aggiungiamo il token alle richieste di login/register (anche se il backend le rifiuterebbe solo se non valide)
  const isAuthRoute = req.url.includes('/api/auth');

  let clonedRequest = req;

  if (token && !isAuthRoute) {
    clonedRequest = req.clone({
      setHeaders: {
        Authorization: `Bearer ${token}`
      }
    });
  }

  return next(clonedRequest).pipe(
    catchError((error: HttpErrorResponse) => {
      // Se riceviamo 401 (Unauthorized) o 403 (Forbidden), forziamo il logout
      if (error.status === 401 || error.status === 403) {
        authService.logout();
      }
      return throwError(() => error);
    })
  );
};
