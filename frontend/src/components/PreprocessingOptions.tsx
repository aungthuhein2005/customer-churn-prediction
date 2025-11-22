import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Label } from './ui/label';
import { Switch } from './ui/switch';
import { ArrowRight, Loader2 } from 'lucide-react';
import { dataStore } from '../utils/dataStore';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import { HelpCircle } from 'lucide-react';

export function PreprocessingOptions() {
  const navigate = useNavigate();
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [options, setOptions] = useState(dataStore.preprocessingOptions);

  useEffect(() => {
    if (!dataStore.csvData) {
      navigate('/');
    }
  }, [navigate]);

  const handleAnalyze = async () => {
    dataStore.preprocessingOptions = options;
    setIsAnalyzing(true);
    
    try {
      const response = await fetch('http://localhost:8000/predict/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          csvData: dataStore.csvData,
          fileName: dataStore.fileName,
          columnMapping: dataStore.columnMapping,
          preprocessingOptions: dataStore.preprocessingOptions,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Prediction failed: ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Prediction result:', data);

      // Store results in dataStore
      if (data.error) {
        throw new Error(data.error);
      }

      dataStore.predictionResults = data;
      setIsAnalyzing(false);
      navigate('/results');
    } catch (error) {
      console.error('Error during prediction:', error);
      setIsAnalyzing(false);
      alert(error instanceof Error ? error.message : 'An error occurred during prediction');
    }
  };

  const handleOptionChange = (key: keyof typeof options, value: boolean) => {
    setOptions(prev => ({ ...prev, [key]: value }));
  };

  if (!dataStore.csvData) {
    return null;
  }

  return (
    <div className="max-w-2xl mx-auto">
      <Card className="shadow-lg">
        <CardHeader>
          <CardTitle>Step 3: Preprocessing Options ⚙️</CardTitle>
          <CardDescription>
            Configure how your data should be prepared for analysis. These options help improve prediction accuracy.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 flex-1">
                <Label htmlFor="handle-missing">Handle missing values automatically?</Label>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <HelpCircle className="h-4 w-4 text-gray-400 cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="max-w-xs">Automatically fill in or remove rows with missing data using statistical methods</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <Switch
                id="handle-missing"
                checked={options.handleMissing}
                onCheckedChange={(checked) => handleOptionChange('handleMissing', checked)}
              />
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 flex-1">
                <Label htmlFor="encode-categorical">Encode categorical columns?</Label>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <HelpCircle className="h-4 w-4 text-gray-400 cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="max-w-xs">Convert text categories (e.g., "Premium", "Basic") into numbers for ML processing</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <Switch
                id="encode-categorical"
                checked={options.encodeCategorical}
                onCheckedChange={(checked) => handleOptionChange('encodeCategorical', checked)}
              />
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 flex-1">
                <Label htmlFor="apply-scaling">Apply feature scaling?</Label>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <HelpCircle className="h-4 w-4 text-gray-400 cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="max-w-xs">Normalize numeric values to the same range for better model performance</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <Switch
                id="apply-scaling"
                checked={options.applyScaling}
                onCheckedChange={(checked) => handleOptionChange('applyScaling', checked)}
              />
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 flex-1">
                <Label htmlFor="generate-sentiment">Generate sentiment score from text?</Label>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <HelpCircle className="h-4 w-4 text-gray-400 cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="max-w-xs">Analyze text columns (like support tickets) to detect positive or negative sentiment</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <Switch
                id="generate-sentiment"
                checked={options.generateSentiment}
                onCheckedChange={(checked) => handleOptionChange('generateSentiment', checked)}
                disabled={!dataStore.columnMapping.supportTextColumn}
              />
            </div>
          </div>

          {!dataStore.columnMapping.supportTextColumn && (
            <p className="text-sm text-gray-500 bg-gray-50 p-3 rounded-lg">
              💡 Sentiment analysis is disabled because no support text column was selected in the previous step.
            </p>
          )}

          <div className="flex justify-end pt-4">
            <Button onClick={handleAnalyze} size="lg" disabled={isAnalyzing}>
              {isAnalyzing ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  Start Analysis
                  <ArrowRight className="ml-2 h-4 w-4" />
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
