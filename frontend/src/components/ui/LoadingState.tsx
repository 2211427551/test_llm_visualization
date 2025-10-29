import { Loader2 } from 'lucide-react';

interface LoadingStateProps {
  message?: string;
}

export const LoadingState = ({ message = '正在初始化可视化...' }: LoadingStateProps) => {
  return (
    <div className="flex flex-col items-center justify-center h-64">
      <div className="relative w-16 h-16">
        <div className="absolute inset-0 border-4 border-purple-500/30 rounded-full" />
        <div className="absolute inset-0 border-4 border-transparent border-t-purple-500 rounded-full animate-spin" />
      </div>
      <p className="mt-4 text-slate-400">{message}</p>
    </div>
  );
};

export const LoadingSpinner = ({ className = '' }: { className?: string }) => {
  return <Loader2 className={`animate-spin ${className}`} />;
};
