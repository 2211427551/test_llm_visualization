'use client';

import { Brain, Github, BookOpen } from 'lucide-react';
import { Button } from '@/components/ui/button';
import Link from 'next/link';

export function Header() {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-slate-800 bg-slate-950/80 backdrop-blur supports-[backdrop-filter]:bg-slate-950/60">
      <div className="container flex h-16 items-center px-6">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-3 mr-8">
          <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">Transformer Visualizer</h1>
            <p className="text-xs text-slate-400">Interactive computation visualization</p>
          </div>
        </Link>
        
        {/* Navigation */}
        <nav className="flex items-center gap-6 flex-1">
          <Link 
            href="/" 
            className="text-sm text-slate-300 hover:text-white transition"
          >
            Home
          </Link>
          <Link 
            href="/demo" 
            className="text-sm text-slate-300 hover:text-white transition"
          >
            Demo
          </Link>
          <Link 
            href="/examples" 
            className="text-sm text-slate-300 hover:text-white transition"
          >
            Examples
          </Link>
        </nav>
        
        {/* Actions */}
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="icon" asChild>
            <a 
              href="https://github.com" 
              target="_blank" 
              rel="noopener noreferrer"
              title="View on GitHub"
            >
              <Github className="w-5 h-5" />
            </a>
          </Button>
          
          <Button variant="ghost" size="icon" title="Documentation">
            <BookOpen className="w-5 h-5" />
          </Button>
        </div>
      </div>
    </header>
  );
}
