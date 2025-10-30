import React from 'react';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  text?: string;
  className?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'md',
  text,
  className = '',
}) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
  };

  return (
    <div className={`flex flex-col items-center justify-center ${className}`}>
      <div
        className={`${sizeClasses[size]} animate-spin rounded-full border-4 border-purple-500/30 border-t-purple-500`}
        role="status"
        aria-label="Loading"
      />
      {text && <p className="mt-2 text-sm text-slate-400">{text}</p>}
    </div>
  );
};

interface ProgressBarProps {
  progress: number;
  text?: string;
  showPercentage?: boolean;
  className?: string;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({
  progress,
  text,
  showPercentage = true,
  className = '',
}) => {
  const percentage = Math.min(Math.max(progress, 0), 100);

  return (
    <div className={`w-full ${className}`}>
      {(text || showPercentage) && (
        <div className="flex justify-between items-center mb-2">
          {text && <span className="text-sm text-slate-300">{text}</span>}
          {showPercentage && (
            <span className="text-sm font-medium text-white">{Math.round(percentage)}%</span>
          )}
        </div>
      )}
      <div className="w-full bg-slate-700 rounded-full h-2.5 overflow-hidden">
        <div
          className="bg-gradient-to-r from-purple-500 to-pink-500 h-2.5 rounded-full transition-all duration-300 ease-out"
          style={{ width: `${percentage}%` }}
          role="progressbar"
          aria-valuenow={percentage}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>
    </div>
  );
};

interface SkeletonProps {
  className?: string;
  variant?: 'text' | 'circular' | 'rectangular';
}

export const Skeleton: React.FC<SkeletonProps> = ({
  className = '',
  variant = 'rectangular',
}) => {
  const variantClasses = {
    text: 'h-4 rounded',
    circular: 'rounded-full',
    rectangular: 'rounded-lg',
  };

  return (
    <div
      className={`animate-pulse bg-slate-700/50 ${variantClasses[variant]} ${className}`}
      aria-hidden="true"
    />
  );
};

export const SkeletonScreen: React.FC = () => {
  return (
    <div className="space-y-4 p-6">
      <Skeleton className="h-8 w-3/4" variant="text" />
      <Skeleton className="h-4 w-1/2" variant="text" />
      <div className="space-y-2">
        <Skeleton className="h-4 w-full" variant="text" />
        <Skeleton className="h-4 w-full" variant="text" />
        <Skeleton className="h-4 w-3/4" variant="text" />
      </div>
      <Skeleton className="h-64 w-full" variant="rectangular" />
    </div>
  );
};
