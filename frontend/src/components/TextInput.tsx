import { useState, useCallback } from 'react';
import { AlertCircle } from 'lucide-react';

interface TextInputProps {
  onSubmit: (text: string) => void;
  isLoading: boolean;
}

const MAX_LENGTH = 500;
const WARNING_THRESHOLD = 450;

export const TextInput: React.FC<TextInputProps> = ({ onSubmit, isLoading }) => {
  const [text, setText] = useState('');
  const [tokenCount, setTokenCount] = useState(0);

  const estimateTokenCount = useCallback((input: string) => {
    const count = Math.ceil(input.trim().split(/\s+/).filter(Boolean).length * 1.3);
    return count;
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newText = e.target.value;
    if (newText.length <= MAX_LENGTH) {
      setText(newText);
      setTokenCount(estimateTokenCount(newText));
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (text.trim() && !isLoading) {
      onSubmit(text.trim());
    }
  };

  const isNearLimit = text.length >= WARNING_THRESHOLD;
  const isValid = text.trim().length > 0;

  return (
    <form onSubmit={handleSubmit} className="w-full space-y-4">
      <div className="relative">
        <label htmlFor="text-input" className="block text-sm font-medium text-gray-700 mb-2">
          Input Text
        </label>
        <textarea
          id="text-input"
          value={text}
          onChange={handleChange}
          placeholder="Enter text to run through the model..."
          disabled={isLoading}
          className={`w-full h-32 px-4 py-3 border rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 outline-none resize-none transition-colors ${
            isNearLimit ? 'border-yellow-500' : 'border-gray-300'
          } ${isLoading ? 'bg-gray-50 cursor-not-allowed' : 'bg-white'}`}
          aria-describedby="char-count token-count"
        />
        <div className="absolute bottom-3 right-3 flex items-center gap-2 text-xs">
          {isNearLimit && (
            <span className="flex items-center gap-1 text-yellow-600">
              <AlertCircle size={14} />
              Near limit
            </span>
          )}
        </div>
      </div>

      <div className="flex items-center justify-between text-sm">
        <div className="flex gap-4">
          <span id="char-count" className={text.length >= WARNING_THRESHOLD ? 'text-yellow-600 font-medium' : 'text-gray-600'}>
            {text.length} / {MAX_LENGTH} characters
          </span>
          <span id="token-count" className="text-gray-600">
            ~{tokenCount} tokens
          </span>
        </div>

        <button
          type="submit"
          disabled={!isValid || isLoading}
          className={`px-6 py-2 rounded-lg font-medium transition-all ${
            isValid && !isLoading
              ? 'bg-primary-600 text-white hover:bg-primary-700 active:scale-95'
              : 'bg-gray-300 text-gray-500 cursor-not-allowed'
          }`}
          aria-label="Run model"
        >
          {isLoading ? 'Running...' : 'Run'}
        </button>
      </div>

      {isNearLimit && (
        <p className="text-xs text-yellow-600 flex items-center gap-1">
          <AlertCircle size={14} />
          Input is approaching the maximum length. Backend may truncate your input.
        </p>
      )}
    </form>
  );
};
