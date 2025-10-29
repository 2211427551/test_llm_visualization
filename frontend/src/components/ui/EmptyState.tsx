import { Sparkles, Play } from 'lucide-react';
import { Button } from './Button';

interface EmptyStateProps {
  title?: string;
  description?: string;
  onStart?: () => void;
}

export const EmptyState = ({
  title = '开始探索',
  description = '输入文本，开启您的Transformer架构可视化之旅',
  onStart,
}: EmptyStateProps) => {
  return (
    <div className="flex flex-col items-center justify-center h-64 text-center px-4">
      <div className="w-20 h-20 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-2xl flex items-center justify-center mb-4">
        <Sparkles className="w-10 h-10 text-purple-400" />
      </div>
      <h3 className="text-xl font-semibold text-white mb-2">{title}</h3>
      <p className="text-slate-400 mb-6 max-w-sm">{description}</p>
      {onStart && (
        <Button variant="primary" onClick={onStart}>
          <Play className="w-4 h-4" />
          开始演示
        </Button>
      )}
    </div>
  );
};
