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
  files: FileList, // Changed from File to FileList
  config: Config
) => {
  const formData = new FormData();

  // Loop over each file and append it to the formData
  for(let i = 0; i < files.length; i++) {
    formData.append('files', files[i]); // The 'files' here should match with the backend parameter name
  }

  const result: fileUploadResponse = await axios
    .post(`${uploadUrl}/${endpoint}`, formData, config)
    .then((response: AxiosResponse) => {
      const data = response.data;
      return { data };
    });

  // .catch((error: AxiosError) => {
  //   console.error('File upload failed');
  //   return { data: { filename: '', status: '' }, error: error };
  // });

  return result;
};

export interface UserQuery {
  query: string;
}

export interface QueryResponse {
  response: string;
  relevantDocs: [
    {
      content: string;
      metadata?: {
        source: string;
      };
    }
  ];
}

export const getAnswer = async (inputQuery: string) => {
  const inputData = { query: inputQuery };
  const result = await axios.post(`${uploadUrl}/query`, inputData);
  // console.log(result);
  const data = {
    response: result.data.response,
    relevantDocs: result.data.relevant_docs,
  };
  return data;
};

export interface DocumentSummary {
  response: string;
}

export const summarizeDocument = async (selectedFileForSummarization: string | null): Promise<DocumentSummary> => {
  const result = await axios.post(`${uploadUrl}/summarize`, { document: selectedFileForSummarization });
  const data = {
    response: result.data.response
  };
  return data;
};


