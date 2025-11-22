import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { Badge } from './ui/badge';
import { Download, TrendingUp, TrendingDown, AlertTriangle } from 'lucide-react';
import { dataStore } from '../utils/dataStore';

export function ResultsDashboard() {
  const navigate = useNavigate();
  const [showDownloadMessage, setShowDownloadMessage] = useState(false);

  useEffect(() => {
    if (!dataStore.csvData || !dataStore.predictionResults) {
      navigate('/');
    }
  }, [navigate]);

  const handleDownload = () => {
    if (!dataStore.predictionResults?.predictions) {
      return;
    }

    const csvContent = [
      ['Customer ID', 'Churn Probability', 'Prediction', 'Risk Level', 'Top Reasons', 'Recommended Actions'],
      ...dataStore.predictionResults.predictions.map(p => [
        p.customerId,
        p.churnProbability.toFixed(4),
        p.prediction,
        p.risk,
        p.reasons.join('; '),
        p.recommendedActions,
      ])
    ].map(row => row.map(cell => `"${cell}"`).join(',')).join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `churn_predictions_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);

    setShowDownloadMessage(true);
    setTimeout(() => setShowDownloadMessage(false), 3000);
  };

  if (!dataStore.csvData || !dataStore.predictionResults) {
    return null;
  }

  const predictions = dataStore.predictionResults.predictions || [];
  const summary = dataStore.predictionResults.summary;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2>Customer Churn Analysis Results 📊</h2>
          <p className="text-gray-600">Analysis completed successfully</p>
        </div>
        <Button onClick={handleDownload} size="lg">
          <Download className="mr-2 h-4 w-4" />
          Download Predictions CSV
        </Button>
      </div>

      {showDownloadMessage && (
        <div className="bg-green-50 border border-green-200 text-green-800 px-4 py-3 rounded-lg">
          ✓ Predictions CSV downloaded successfully!
        </div>
      )}

      {/* Summary Statistics */}
      {summary && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Total Customers</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{summary.totalCustomers}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Predicted Churn</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-red-600">{summary.predictedChurn}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Avg Churn Probability</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{(summary.averageChurnProbability * 100).toFixed(1)}%</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardDescription>Model Accuracy</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{(summary.modelAccuracy * 100).toFixed(1)}%</div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Top Churn Predictions Table */}
      <Card>
        <CardHeader>
          <CardTitle>Customer Predictions</CardTitle>
          <CardDescription>All customers with churn predictions and recommended actions</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Customer ID</TableHead>
                <TableHead>Churn Probability</TableHead>
                <TableHead>Top Churn Reasons</TableHead>
                <TableHead>Recommended Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {predictions.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={4} className="text-center text-gray-500">
                    No predictions available
                  </TableCell>
                </TableRow>
              ) : (
                predictions
                  .sort((a, b) => b.churnProbability - a.churnProbability)
                  .map((prediction) => (
                    <TableRow key={prediction.customerId}>
                      <TableCell>{prediction.customerId}</TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          {prediction.risk === 'high' ? (
                            <TrendingUp className="h-4 w-4 text-red-500" />
                          ) : prediction.risk === 'medium' ? (
                            <AlertTriangle className="h-4 w-4 text-orange-500" />
                          ) : (
                            <TrendingDown className="h-4 w-4 text-green-500" />
                          )}
                          <span>{(prediction.churnProbability * 100).toFixed(1)}%</span>
                          <Badge variant={prediction.risk === 'high' ? 'destructive' : prediction.risk === 'medium' ? 'default' : 'secondary'}>
                            {prediction.risk}
                          </Badge>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex flex-wrap gap-1">
                          {prediction.reasons.slice(0, 3).map((reason, idx) => (
                            <Badge key={idx} variant="outline" className="text-xs">
                              {reason}
                            </Badge>
                          ))}
                        </div>
                      </TableCell>
                      <TableCell className="text-sm">{prediction.recommendedActions}</TableCell>
                    </TableRow>
                  ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <div className="flex justify-center">
        <Button variant="outline" onClick={() => navigate('/')}>
          Analyze Another Dataset
        </Button>
      </div>
    </div>
  );
}
