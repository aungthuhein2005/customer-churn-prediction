import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { AlertCircle, ArrowRight } from 'lucide-react';
import { dataStore } from '../utils/dataStore';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import { HelpCircle } from 'lucide-react';

const NONE_VALUE = '__none__';

export function ColumnMapping() {
  const navigate = useNavigate();
  const [targetColumn, setTargetColumn] = useState('');
  const [customerIdColumn, setCustomerIdColumn] = useState('');
  const [supportTextColumn, setSupportTextColumn] = useState('');
  const [dateColumn, setDateColumn] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    if (!dataStore.csvData) {
      navigate('/');
    }
  }, [navigate]);

  if (!dataStore.csvData) {
    return null;
  }

  const { headers, rows, detectedTypes } = dataStore.csvData;
  const previewRows = rows.slice(0, 5);

  const handleConfirm = () => {
    if (!targetColumn) {
      setError('Please select a Target Column (Churn)');
      return;
    }
    if (!customerIdColumn) {
      setError('Please select a Customer ID Column');
      return;
    }

    dataStore.columnMapping = {
      targetColumn,
      customerIdColumn,
      supportTextColumn,
      dateColumn,
    };

    navigate('/preprocessing');
  };

  return (
    <div className="space-y-6">
      <Card className="shadow-lg">
        <CardHeader>
          <CardTitle>CSV Preview</CardTitle>
          <CardDescription>
            First 5 rows of <strong>{dataStore.fileName}</strong>
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  {headers.map((header, index) => (
                    <TableHead key={index}>
                      <div className="flex flex-col gap-1">
                        <span>{header}</span>
                        <Badge variant="secondary" className="w-fit text-xs">
                          {detectedTypes[header]}
                        </Badge>
                      </div>
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {previewRows.map((row, rowIndex) => (
                  <TableRow key={rowIndex}>
                    {row.map((cell, cellIndex) => (
                      <TableCell key={cellIndex}>{cell}</TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      <Card className="shadow-lg">
        <CardHeader>
          <CardTitle>Step 2: Map Columns 🔗</CardTitle>
          <CardDescription>
            Please tell us which columns represent the following. Your CSV may have different column names. 
            Mapping ensures the system knows which column to use for prediction.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <div className="space-y-4">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Label htmlFor="target-column">Target Column (Churn) *</Label>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <HelpCircle className="h-4 w-4 text-gray-400 cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="max-w-xs">The column that indicates whether a customer churned (Yes/No or 1/0)</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <Select value={targetColumn} onValueChange={setTargetColumn}>
                <SelectTrigger id="target-column">
                  <SelectValue placeholder="Select target column" />
                </SelectTrigger>
                <SelectContent>
                  {headers.map((header) => (
                    <SelectItem key={header} value={header}>
                      {header} ({detectedTypes[header]})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Label htmlFor="customer-id-column">Customer ID Column *</Label>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <HelpCircle className="h-4 w-4 text-gray-400 cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="max-w-xs">A unique identifier for each customer in your dataset</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <Select value={customerIdColumn} onValueChange={setCustomerIdColumn}>
                <SelectTrigger id="customer-id-column">
                  <SelectValue placeholder="Select customer ID column" />
                </SelectTrigger>
                <SelectContent>
                  {headers.map((header) => (
                    <SelectItem key={header} value={header}>
                      {header} ({detectedTypes[header]})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Label htmlFor="support-text-column">Support Text Column (Optional)</Label>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <HelpCircle className="h-4 w-4 text-gray-400 cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="max-w-xs">Text data like support tickets that can be analyzed for sentiment</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <Select 
                value={supportTextColumn || NONE_VALUE} 
                onValueChange={(value) => setSupportTextColumn(value === NONE_VALUE ? '' : value)}
              >
                <SelectTrigger id="support-text-column">
                  <SelectValue placeholder="Select support text column (optional)" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={NONE_VALUE}>None</SelectItem>
                  {headers.map((header) => (
                    <SelectItem key={header} value={header}>
                      {header} ({detectedTypes[header]})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Label htmlFor="date-column">Date Column (Optional)</Label>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <HelpCircle className="h-4 w-4 text-gray-400 cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="max-w-xs">Date information like last login or account creation date</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
              <Select 
                value={dateColumn || NONE_VALUE} 
                onValueChange={(value) => setDateColumn(value === NONE_VALUE ? '' : value)}
              >
                <SelectTrigger id="date-column">
                  <SelectValue placeholder="Select date column (optional)" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={NONE_VALUE}>None</SelectItem>
                  {headers.map((header) => (
                    <SelectItem key={header} value={header}>
                      {header} ({detectedTypes[header]})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="flex justify-end">
            <Button onClick={handleConfirm} size="lg">
              Confirm Mapping
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
