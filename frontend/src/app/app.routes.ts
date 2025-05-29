import { Routes } from '@angular/router';
import { IndexComponent } from './jabatan/index/index.component';
import { ViewComponent } from './jabatan/view/view.component';
import { CreateComponent } from './jabatan/create/create.component';
import { EditComponent } from './jabatan/edit/edit.component';

export const routes: Routes = [
  { path: '', redirectTo: 'jabatan/index', pathMatch: 'full' },
  { path: 'jabatan/index', component: IndexComponent },
  { path: 'jabatan/create', component: CreateComponent },
  { path: 'jabatan/:jabatanId/view', component: ViewComponent },
  { path: 'jabatan/:jabatanId/edit', component: EditComponent },
];
