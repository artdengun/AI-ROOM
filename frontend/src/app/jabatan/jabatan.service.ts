import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { catchError, Observable, throwError } from 'rxjs';
import { Jabatan } from './jabatan';

@Injectable({
  providedIn: 'root'
})
export class JabatanService {

  // api
  private api = "https://jsonplaceholder.typicode.com";

  // ini header
  httpOptions =  {
    headers: new HttpHeaders({
      'Content-Type':'application/json'
    })
  }

  // buat constructor
  constructor(private http: HttpClient) { }

  // get semua data
  getSemuaData(): Observable<any> {
    return this.http.get(this.api + '/posts/')
    .pipe(catchError(this.errorHandler))
  }

  create(jabatan:Jabatan): Observable<any> {
    return this.http.post(this.api + '/posts/', JSON.stringify(jabatan), this.httpOptions)
    .pipe(catchError(this.errorHandler))
  }

  cariData(id:number): Observable<any> {
    return this.http.get(this.api + '/posts/' + id)
    .pipe(catchError(this.errorHandler))

  }

  updateData(id:number, jabatan:Jabatan): Observable<any>{
    return this.http.put(this.api + '/posts/' + id, JSON.stringify(jabatan), this.httpOptions)
    .pipe(catchError(this.errorHandler))
  }
  delete(id:number){
    return this.http.delete(this.api + '/posts/' + id, this.httpOptions)
  }


  // untuk handle error
  errorHandler(error:any){
    let message = '';
    if(error.error instanceof ErrorEvent){
      message = error.error.message;
    } else {
      message = `Error Code: ${error.status}\nMessage: ${error.message}`;
    }
    return throwError(message);
  }

}
