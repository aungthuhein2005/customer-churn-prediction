import { Outlet, useLocation } from 'react-router';

export function Layout() {
  const location = useLocation();
  
  const steps = [
    { path: '/', label: 'Upload', number: 1 },
    { path: '/mapping', label: 'Map Columns', number: 2 },
    { path: '/preprocessing', label: 'Preprocessing', number: 3 },
    { path: '/results', label: 'Results', number: 4 },
  ];

  const currentStepIndex = steps.findIndex(step => 
    step.path === location.pathname
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        <header className="mb-8 text-center">
          <h1 className="text-indigo-600 mb-2">Churn Predictor</h1>
          <p className="text-gray-600">AI-powered customer churn analysis</p>
        </header>

        {/* Progress Stepper */}
        {currentStepIndex >= 0 && (
          <div className="mb-8">
            <div className="flex justify-between items-center max-w-3xl mx-auto">
              {steps.map((step, index) => (
                <div key={step.path} className="flex items-center flex-1">
                  <div className="flex flex-col items-center flex-1">
                    <div className={`w-10 h-10 rounded-full flex items-center justify-center border-2 transition-colors ${
                      index <= currentStepIndex
                        ? 'bg-indigo-600 border-indigo-600 text-white'
                        : 'bg-white border-gray-300 text-gray-400'
                    }`}>
                      {step.number}
                    </div>
                    <span className={`mt-2 text-sm ${
                      index <= currentStepIndex ? 'text-indigo-600' : 'text-gray-400'
                    }`}>
                      {step.label}
                    </span>
                  </div>
                  {index < steps.length - 1 && (
                    <div className={`h-0.5 flex-1 -mt-6 transition-colors ${
                      index < currentStepIndex ? 'bg-indigo-600' : 'bg-gray-300'
                    }`} />
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        <main>
          <Outlet />
        </main>
      </div>
    </div>
  );
}
