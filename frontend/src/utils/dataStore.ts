import { CSVData } from './csvParser';

interface PredictionResult {
  customerId: string;
  churnProbability: number;
  prediction: string;
  confidence: number;
  risk: string;
  reasons: string[];
  recommendedActions: string;
}

interface DataStore {
  csvData: CSVData | null;
  fileName: string;
  columnMapping: {
    targetColumn: string;
    customerIdColumn: string;
    supportTextColumn: string;
    dateColumn: string;
  };
  preprocessingOptions: {
    handleMissing: boolean;
    encodeCategorical: boolean;
    applyScaling: boolean;
    generateSentiment: boolean;
  };
  predictionResults: {
    success: boolean;
    summary?: {
      totalCustomers: number;
      predictedChurn: number;
      averageChurnProbability: number;
      modelAccuracy: number;
    };
    predictions?: PredictionResult[];
    featureImportance?: Record<string, number>;
  } | null;
}

// Simple in-memory store for the session
export const dataStore: DataStore = {
  csvData: null,
  fileName: '',
  columnMapping: {
    targetColumn: '',
    customerIdColumn: '',
    supportTextColumn: '',
    dateColumn: '',
  },
  preprocessingOptions: {
    handleMissing: true,
    encodeCategorical: true,
    applyScaling: true,
    generateSentiment: false,
  },
  predictionResults: null,
};
