import axios, {
  AxiosError,
  AxiosResponse,
  AxiosRequestConfig,
  AxiosProgressEvent,
} from 'axios';

const uploadUrl: string = 'http://localhost:8000';

export interface Config extends AxiosRequestConfig {
  onUploadProgress?: (progressEvent: AxiosProgressEvent) => void;
  headers?: Record<string, string>;
}

export interface fileUploadResponse {
  data: {
    filename: string;
    status: string;
  };
  error?: AxiosError | null;
}

export const uploadDocument = async (
  endpoint: string,
  files: FileList,
  file_type: string, 
  config: Config
) => {
  const formData = new FormData();

  for (let i = 0; i < files.length; i++) {
    formData.append('files', files[i]);
  }

  const result: fileUploadResponse = await axios.post(
    `${uploadUrl}/${endpoint}`,
    formData,
    {
      ...config,
      params: { file_type },  // Pass the file_type as a parameter
    }
  ).then((response: AxiosResponse) => {
    const data = response.data;
    return { data };
  });

  return result;
};

export interface UserQuery {
  query: string;
}

export interface Document {
  content: string;
  metadata?: {
    source: string;
  };
}

export interface QueryResponse {
  data?: {
    response: string;
    relevantDocs: Document[];
  };
  error?: AxiosError | null;
}


export const getAnswer = async (
  inputQuery: string,
  namespace: string,
  dateRange: Array<string> | null
) => {
  const inputData = { 
    input_query: { query: inputQuery }, 
    dateRange: { dateRange }
  };
  const result: QueryResponse = await axios
    .post(
      `${uploadUrl}/query`,
      inputData,
      {
        params: { namespace },
      }
    )
    .then((response: AxiosResponse) => {
      const data = response.data;
      return { data };
    })
    .catch((error: AxiosError) => {
      return { error };
    });

  return result;
};

export interface DocumentSummary {

  summary:{
    date: string;
    name: string;
    summary: string;
    followUpIns: string | null;
    organization: string;
    diagnosis : string | null;
    medications : string | null;
    medicationsinstructions : string ;
    instructions : string | null;
    ad : string | null;
    snf : string | null;
    cost : string | null;
    diabetesCoverage : string | null;
    contact : string | null;
  };
  error?: AxiosError | null;
}

export const summarizeDocument = async (
  selectedFileForSummarization: string | null
): Promise<DocumentSummary | {error: any}> => {
  const result: DocumentSummary | {error: any} = await axios
    .post(
      `${uploadUrl}/file-summary`, 
      {}, // empty body
      {
        params: {
          filename: selectedFileForSummarization, // 'filename' as a query parameter
        },
      }
    )
    .then((response: AxiosResponse) => {
      if (response.data[0]) {
        const summaryData = response.data[0];
        return {
          summary: {
            date: summaryData.date,
            name: summaryData.name,
            summary: summaryData.summary,
            followUpIns: summaryData.followUpIns || null,
            organization: summaryData.organization,
            diagnosis: summaryData.diagnosis || null,
            medications: summaryData.medications || null,
            medicationsinstructions: summaryData.medicationsinstructions,
            instructions: summaryData.instructions || null,
            ad: summaryData.ad || null,
            snf: summaryData.snf || null,
            cost: summaryData.cost || null,
            diabetesCoverage: summaryData.diabetesCoverage || null,
            contact: summaryData.contact || null
          }
        };
      } else {
        return { error: new Error("Summary data not found in response") };
      }
    })
    
    .catch((error: AxiosError) => {
      return { error };
    });

  return result;
};

export const getChartData = async () => {
  try {
    const response = await axios.post(`${uploadUrl}/report-data`);
    return response.data;
  } catch (error) {
    console.error('API call error:', error);
    return { error: 'Failed to get chart data from the server.' };
  }
};
