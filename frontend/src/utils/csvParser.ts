export interface CSVData {
  headers: string[];
  rows: any[][];
  detectedTypes: Record<string, string>;
}

export function parseCSV(csvText: string): CSVData {
  const lines = csvText.trim().split('\n');
  const headers = lines[0].split(',').map(h => h.trim());
  
  const rows = lines.slice(1).map(line => {
    return line.split(',').map(cell => cell.trim());
  });

  // Detect column types
  const detectedTypes: Record<string, string> = {};
  headers.forEach((header, index) => {
    const sampleValue = rows[0]?.[index];
    if (!sampleValue) {
      detectedTypes[header] = 'unknown';
    } else if (!isNaN(Number(sampleValue))) {
      detectedTypes[header] = 'number';
    } else if (sampleValue.toLowerCase() === 'yes' || sampleValue.toLowerCase() === 'no') {
      detectedTypes[header] = 'boolean';
    } else {
      detectedTypes[header] = 'text';
    }
  });

  return { headers, rows, detectedTypes };
}
