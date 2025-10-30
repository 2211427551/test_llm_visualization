'use client';

import { ReactNode } from 'react';
import { Header } from './Header';
import { Sidebar } from './Sidebar';

interface AppLayoutProps {
  children: ReactNode;
  showSidebar?: boolean;
}

export function AppLayout({ children, showSidebar = true }: AppLayoutProps) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <Header />
      
      <div className="flex">
        {showSidebar && (
          <aside className="w-80 h-[calc(100vh-4rem)] sticky top-16 overflow-y-auto">
            <Sidebar />
          </aside>
        )}
        
        <main className={`flex-1 p-6 ${showSidebar ? 'max-w-[calc(100vw-20rem)]' : 'max-w-full'}`}>
          {children}
        </main>
      </div>
    </div>
  );
}
