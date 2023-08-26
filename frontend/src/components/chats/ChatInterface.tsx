import React, { useState, useEffect, useRef, useContext } from 'react';
import { Avatar, Button, Box, Grid, Typography, LinearProgress, Snackbar  } from '@mui/material';
import { deepOrange, deepPurple } from '@mui/material/colors';
import { getAnswer, summarizeDocument, getChartData  } from '../../apis/api';
import { AppBar, TextField, MenuItem, Select } from '@mui/material';
import { OptionContext } from './../../OptionContext';
import './ChatInterface.css';
import 'react-date-range/dist/styles.css'; // main style file
import 'react-date-range/dist/theme/default.css'; // theme css file
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { DateRangePicker, Range } from 'react-date-range';
import { eachDayOfInterval, format } from 'date-fns';
import FileUploader from '../uploader/fileUploader';
import Plot from 'react-plotly.js';


import axios, {
  AxiosError,
  AxiosResponse,
  AxiosRequestConfig,
  AxiosProgressEvent,
} from 'axios';

const uploadUrl: string = 'http://localhost:8000';


interface ChatMessage {
  id: string;
  message: string;
  isUser: boolean;
  avatar: string;
}

interface MessageProps {
  id: string;
  message: string;
  isUser: boolean;
  avatar: string;
}

interface DateRange {
  startDate: Date | undefined;
  endDate: Date | undefined;
  key: string | undefined;
}


const Message = ({ message }: { message: MessageProps }) => {

  const avatarStyles = {
    gradient: {
      background: `radial-gradient(circle, ${message.isUser ? deepPurple[500] : deepOrange[500]}, ${message.isUser ? deepPurple[900] : deepOrange[900]})`
    }
  };

  return (
    <div className={`chat-message ${message.isUser ? 'user-message' : 'bot-message'}`} key={message.id}>
      <Avatar style={avatarStyles.gradient} variant="rounded">{message.isUser ? 'U' : 'AI'}</Avatar>
      <Typography sx={{ margin: '0 10px', borderRadius: '10px', backgroundColor: '#f3f3f3', padding: '5px 10px' }} variant="body1">{message.message}</Typography>
    </div>
  );
};

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [selectedFileForSummarization, setSelectedFileForSummarization] = useState<string | null>(null);
  const chatMessageRef = useRef<HTMLDivElement>(null);
  const { selectedOption, setSelectedOption } = React.useContext(OptionContext);
  const [isPickerVisible, setIsPickerVisible] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [isChatVisible, setChatVisible] = useState(false);
  const [summaryData, setSummaryData] = useState<string | null>(null);
  const [chartData, setChartData] = useState<any | null>(null);

  const toggleChatVisibility = () => {
    setChatVisible(!isChatVisible);
  };

  const [dateRange, setDateRange] = useState<{ startDate: Date, endDate: Date }>({
    startDate: new Date(),
    endDate: new Date()
  });


  useEffect(() => {
    const chatMessageLog = chatMessageRef.current;
    if (chatMessageLog) {
      chatMessageLog.scrollTop = chatMessageLog.scrollHeight;
    }
  }, [messages]);

  const handleDateRangeChange = (ranges: { [key: string]: Range }) => {
    const range = ranges['selection'];
    if (range.startDate && range.endDate) {
      setDateRange({
        startDate: range.startDate,
        endDate: range.endDate
      });
    }
  };
  
  
  const formatDateRange = (range: { startDate: Date, endDate: Date }): string[] => {
    const interval = { start: range.startDate, end: range.endDate };
    const datesInRange = eachDayOfInterval(interval);
    
    return datesInRange.map(date => format(date, 'yyyy-MM-dd'));
  };

  const handleOptionChange = (event: React.ChangeEvent<{ value: unknown }>) => {
    setSelectedOption(event.target.value as string);
    console.log(event.target.value);
  };

  const handleInputMessageChange = (
    event: React.ChangeEvent<HTMLTextAreaElement>
  ) => {
    setInputMessage(event.target.value);
  };

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      handleInputMessageSubmit();
    }
  };

  const handleInputMessageSubmit = async () => {
    if (inputMessage.trim() === '') {
      return;
    }

    const newMessage: ChatMessage = {
      id: Math.random().toString(36).substring(7),
      message: inputMessage,
      isUser: true,
      avatar: 'https://example.com/user-avatar.png',
    };

    setMessages((messages) => [
      ...messages,
      newMessage,
      {
        id: Math.random().toString(36).substring(7),
        message: 'Loading...',
        isUser: false,
        avatar: 'https://example.com/chatbot-avatar.png',
      },
    ]);

    setInputMessage('');

    // const dateRange = null; 

    const result = await getAnswer(inputMessage, selectedOption, formatDateRange(dateRange));

    if (result.data) {
      const chatbotMessage: ChatMessage = {
        id: Math.random().toString(36).substring(7),
        message: result.data.response,
        isUser: false,
        avatar: 'https://example.com/chatbot-avatar.png',
      };

      // Add message to chat
      setMessages((prevMessages) => [
        ...prevMessages.slice(0, -1),
        chatbotMessage,
      ]);
    } else {
      // handle error here, maybe add an error message to the chat
      console.error(result.error);
    }
  };

  const handleSummarize = async () => {

     
    const result = await summarizeDocument(selectedFileForSummarization);
    console.log('Summarize document result:', result);
    let summaryMessage: ChatMessage;

    if ('summary' in result) {
        const summaryText = `
            Date: ${result.summary.date}
            Name: ${result.summary.name}
            if(result.summary.summary){
              Summary: ${result.summary.summary.replace(/\n/g, ' ')}
            }
            Follow-up Instructions: ${result.summary.followUpIns || 'None'}
            if(result.summary.organization){
              Organization: ${result.summary.organization.replace(/\n/g, ' ')}
            }
            Diagnosis: ${result.summary.diagnosis || 'None'}
            Medications: ${result.summary.medications || 'None'}
            if(result.summary.medicationsinstructions){
              Medications Instructions: ${result.summary.medicationsinstructions.replace(/\n/g, ' ') || 'None'}
            }
            Instructions: ${result.summary.instructions || 'None'}
            Ad: ${result.summary.ad || 'None'}
            Snf: ${result.summary.snf || 'None'}
            Cost: ${result.summary.cost || 'None'}
            DiabetesCoverage: ${result.summary.diabetesCoverage || 'None'}
            Contact: ${result.summary.contact || 'None'}
        `;
        setSummaryData(summaryText);
        // ... rest of the code 
        // ... handle error case
        
    }else {
      setSummaryData(`Error: ${result.error.message}`);
    }
  };
  

  useEffect(() => {
    const fetchUploadedFiles = async (namespace: string) => {
      try {
        const response = await axios.get(`${uploadUrl}/uploaded-files`, {
          params: { namespace },
        });
        setUploadedFiles(response.data.files);
      } catch (error) {
        console.error('Error fetching uploaded files:', error);
      }
    };

    fetchUploadedFiles(selectedOption);
  }, [selectedOption]);

  useEffect(() => {
    async function fetchData() {
      const data = await getChartData();
      setChartData(data);
    }

    fetchData();
  }, []);

  if (!chartData) {
    return <div>Loading...</div>;
  }

  const plotData = [
    {
      type: 'scatter' as const,  
      mode: 'lines+markers',
      x: Array.isArray(chartData) ? chartData.map((_, index) => index) : [],
      y: chartData,
    }
  ];

return (
  <>
    {/* Always show the dropdowns, date picker, and file uploader on the left */}
    <div style={{ position: 'fixed', left: '20px', top: '20px', zIndex: 1000 }}>
      
    <FileUploader />

      {/* Existing code */}
      <Button variant="contained" color="primary" onClick={() => setIsPickerVisible(!isPickerVisible)}>
        Date Picker
      </Button>
      {isPickerVisible && (
        <DateRangePicker
          onChange={handleDateRangeChange}
          showPreview={true}
          moveRangeOnFirstSelection={false}
          months={2}
          ranges={[
            {
              startDate: dateRange.startDate,
              endDate: dateRange.endDate,
              key: 'selection',
            },
          ]}
          direction="horizontal"
        />
      )}
      
      <Select
          value={selectedOption}
          onChange={(event) =>{
            setSelectedOption(event.target.value as string)
            setSelectedFile(null);
          }}
          
          variant="outlined"
          fullWidth
        >
          <MenuItem value="Lab reports">Lab Reports</MenuItem>
          <MenuItem value="Prescriptions">Prescriptions</MenuItem>
          <MenuItem value="Imaging reports">Imaging reports</MenuItem>
          <MenuItem value="Genetic reports">Genetic reports</MenuItem>
          <MenuItem value="Histopathology reports">Histopathology reports</MenuItem>
          <MenuItem value="Pill bottle photos">Pill bottle photos</MenuItem>
          <MenuItem value="Clinician notes">Clinician notes</MenuItem>
          <MenuItem value="Discharge notes">Discharge notes</MenuItem>
          <MenuItem value="End of Visit Summaries">End of Visit Summaries</MenuItem>
          <MenuItem value="Healthcare bills">Healthcare bills</MenuItem>
          <MenuItem value="Legal documents">Legal documents</MenuItem>
          <MenuItem value="Insurance documents">Insurance documents</MenuItem>
          <MenuItem value="Advance care planning">Advance care planning</MenuItem>
          <MenuItem value="Financial documents">Financial documents</MenuItem>
          <MenuItem value="Miscellaneous">Miscellaneous</MenuItem>
      </Select>

      {uploadedFiles.length > 0 && (
        <Select
          value={selectedFile || ''}
          onChange={(event) => {
            setSelectedFile(event.target.value as string);
            setSelectedFileForSummarization(event.target.value as string);
          }}
          fullWidth
        >
          {uploadedFiles.map((file) => (
            <MenuItem key={file} value={file}>
              {file}
            </MenuItem>
          ))}
        </Select>
      )}

    </div>

    {!isChatVisible ? (
      <div 
        style={{
          position: 'fixed',
          bottom: '20px',
          right: '20px',
          cursor: 'pointer',
          zIndex: 1000
        }}
        onClick={toggleChatVisibility}
      >
        <Avatar sx={{ bgcolor: deepOrange[500] }}>AI</Avatar>
      </div>
    ) : (
      <div className="chatbox" style={{ 
        position: 'fixed', 
        bottom: '0', 
        right: '0', 
        width: '400px', 
        maxHeight: '70vh', 
        zIndex: 1000, 
        overflowY: 'scroll', 
        boxShadow: '-2px 0px 15px rgba(0, 0, 0, 0.2)' 
    }}>
        <div className="chat-logs" ref={chatMessageRef}>
          {messages.map((message, index) => (
            <Message key={index} message={message} />
          ))}
        </div>
        <div className="input-container">
        <Box display="flex" justifyContent="flex-end" alignItems="center" p={1}>
            <TextField
              value={inputMessage}
              className="chat-input"
              variant="outlined"
              placeholder="Type your message..."
              onKeyDown={handleKeyDown}
              onChange={handleInputMessageChange}
              style={{ flex: 1, marginRight: '10px' }} 
            />
            <Button
              className="chat-input-btn"
              variant="contained"
              color="secondary"
              onClick={handleInputMessageSubmit}
            >
              <i
                className="fa fa-paper-plane chat-input-send-icon"
                aria-hidden="true"
              ></i>
            </Button>
            <Button
              className="summarize-btn"
              variant="contained"
              color="secondary"
              onClick={handleSummarize}
              disabled={!selectedFileForSummarization}
              style={{ marginLeft: '10px' }}  
            >
              <i
                className="fa fa-book chat-input-send-icon"
                aria-hidden="true"
              ></i>
            </Button>
          </Box>
        </div>

        <div className="chat-content">
          <Button
            style={{ position: 'absolute', top: '10px', right: '0px' }}
            onClick={toggleChatVisibility}
          >
            X
          </Button>
        </div>
      </div>
    )}

    {summaryData && (
      <div
        style={{
          position: 'fixed',
          top: '45%',
          left: '70%',
          transform: 'translate(-50%, -50%)',
          padding: '20px',
          border: '1px solid #ddd',
          borderRadius: '8px',
          backgroundColor: '#fff7',
          zIndex: 999,
          width: '60%',
          overflowY: 'auto',  
          maxHeight: '70vh',  
        }}
      >
        <Typography variant="h6" gutterBottom>
          Data:
        </Typography>
        
        <div style={{
          display: 'flex', 
          flexWrap: 'wrap', 
          gap: '10px'  
        }}>
          {summaryData.split('\n').map((line, index) => {
            const colonIndex = line.indexOf(':');
            const key = line.slice(0, colonIndex);
            const value = line.slice(colonIndex + 1);
            return (
              value && value.trim() !== 'None' && (
                <div key={index} 
                    style={{ 
                      padding: '8px',
                      border: '1px solid #ccc',
                      borderRadius: '4px',
                      flex: '1 0 calc(33.333% - 10px)',  // This makes the box take up roughly 1/3 of the available width, minus the gap
                      boxSizing: 'border-box'  // This ensures padding and border are included in the width
                    }}
                >
                  <Typography variant="subtitle1"><h1>{key}</h1></Typography>
                  <Typography variant="body1">{value}</Typography>
                </div>
              )
            );
          })}
        </div>
      </div>
    )}



  </>
);

};

export default ChatInterface;
