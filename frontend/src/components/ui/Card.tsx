import { ReactNode } from 'react';

interface CardProps {
  children: ReactNode;
  className?: string;
  glow?: boolean;
  hover?: boolean;
}

export const Card = ({ children, className = '', glow = false, hover = true }: CardProps) => {
  return (
    <div
      className={`
        bg-slate-800/50 backdrop-blur-sm 
        border border-slate-700/50 
        rounded-xl p-6 
        transition-all duration-300
        ${hover ? 'hover:border-purple-500/50 hover:shadow-xl hover:shadow-purple-500/10' : ''}
        ${glow ? 'shadow-2xl shadow-purple-500/20' : ''}
        ${className}
      `}
    >
      {children}
    </div>
  );
};
