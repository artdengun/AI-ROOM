import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { RouterModule } from '@angular/router';
import { Jabatan } from '../jabatan';
import { HttpClientModule } from '@angular/common/http';
import { JabatanService } from '../jabatan.service';

@Component({
  selector: 'app-index',
  imports: [CommonModule, RouterModule, HttpClientModule],
  templateUrl: './index.component.html',
  styleUrl: './index.component.scss'
})
export class IndexComponent {
  jabatans: Jabatan[] = [];
  constructor(public jabatanService: JabatanService){}

  ngOnInit(): void { 
    this.jabatanService.getSemuaData().subscribe((data: Jabatan[])=>{
      this.jabatans = data;
      console.log(this.jabatans)
    })
  }

  deleteJabatan(id:number){
    this.jabatanService.delete(id).subscribe(x => {
      this.jabatans = this.jabatans.filter( item => item.id != id);
      console.log('Jabatan Berhasil Di Delete');
    })
  }
}
