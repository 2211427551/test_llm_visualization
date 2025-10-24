import { useState } from 'react';
import { TextInput } from './components/TextInput';
import { ExecutionControls } from './components/ExecutionControls';
import { MacroView } from './components/MacroView';
import { MicroView } from './components/MicroView';
import { SummaryPanel } from './components/SummaryPanel';
import { ErrorDisplay } from './components/ErrorDisplay';
import { useExecutionStore } from './store/executionStore';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
import { usePlayback } from './hooks/usePlayback';
import { runModelForward } from './api/client';

function App() {
  const { status, error, setStatus, setData, setError } = useExecutionStore();
  const [showError, setShowError] = useState(true);

  useKeyboardShortcuts();
  usePlayback();

  const handleSubmit = async (text: string) => {
    setStatus('loading');
    setError(null);
    setShowError(true);

    try {
      const response = await runModelForward(text);
      setData(response);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An unexpected error occurred';
      setError(errorMessage);
    }
  };

  const isLoading = status === 'loading';

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <header className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Model Execution Visualizer
          </h1>
          <p className="text-gray-600">
            Enter text to see step-by-step model computation with interactive controls
          </p>
        </header>

        {error && showError && (
          <div className="mb-6">
            <ErrorDisplay error={error} onDismiss={() => setShowError(false)} />
          </div>
        )}

        <div className="space-y-6">
          <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
            <TextInput onSubmit={handleSubmit} isLoading={isLoading} />
          </div>

          {status === 'success' && (
            <>
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 space-y-6">
                  <MacroView />
                  <MicroView />
                </div>
                <div className="space-y-6">
                  <ExecutionControls />
                  <SummaryPanel />
                </div>
              </div>
            </>
          )}

          {isLoading && (
            <div className="flex items-center justify-center h-64 bg-white border border-gray-200 rounded-lg shadow-sm">
              <div className="text-center">
                <div className="inline-block w-12 h-12 border-4 border-primary-500 border-t-transparent rounded-full animate-spin mb-4"></div>
                <p className="text-gray-600">Running model computation...</p>
              </div>
            </div>
          )}
        </div>

        <footer className="mt-12 pt-8 border-t border-gray-200 text-center text-sm text-gray-600">
          <p>Use keyboard shortcuts for quick navigation: ← → (step), Space (play/pause), Home/End (jump)</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
