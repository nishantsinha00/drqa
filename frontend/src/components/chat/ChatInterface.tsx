import React, { useState, useEffect, useRef } from 'react';
import { Avatar, Button, Box, Grid, Typography } from '@mui/material';
import { deepOrange, deepPurple } from '@mui/material/colors';
import { getAnswer, summarizeDocument } from '../../apis/api';
import { AppBar, TextField } from '@mui/material';

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

const Message = ({ message }: { message: MessageProps }) => {
  return (
    <div
      className="chat-message"
      style={{ backgroundColor: message.isUser ? '#fff' : '#f6f6f8' }}
      key={message.id}
    >
      <Avatar
        sx={{
          bgcolor: message.isUser ? deepPurple[500] : deepOrange[500],
        }}
        variant="rounded"
      >
        {message.isUser ? 'You' : 'AI'}
      </Avatar>

      <Typography
        sx={{ margin: '10px', paddingLeft: '10px', paddingRight: '10px' }}
        variant="body1"
      >
        {message.message}
      </Typography>
    </div>
  );
};

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [selectedFileForSummarization, setSelectedFileForSummarization] = useState<string | null>(null);
  const chatMessageRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const chatMessageLog = chatMessageRef.current;
    if (chatMessageLog) {
      chatMessageLog.scrollTop = chatMessageLog.scrollHeight;
    }
  }, [messages]);

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

    const result = await getAnswer(inputMessage);

    const chatbotMessage: ChatMessage = {
      id: Math.random().toString(36).substring(7),
      message: result.response,
      isUser: false,
      avatar: 'https://example.com/chatbot-avatar.png',
    };

    setMessages((prevMessages) => [
      ...prevMessages.slice(0, -1),
      chatbotMessage,
    ]);
  };

  const handleSummarize = async () => {
    // if (selectedFileForSummarization === null) {
    //   return;
    // }
  
    const newMessage: ChatMessage = {
      id: Math.random().toString(36).substring(7),
      message: 'Summarizing document...',
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
  
    const result = await summarizeDocument(selectedFileForSummarization);
  
    const summaryMessage: ChatMessage = {
      id: Math.random().toString(36).substring(7),
      message: result.response,
      isUser: false,
      avatar: 'https://example.com/chatbot-avatar.png',
    };
  
    setMessages((prevMessages) => [
      ...prevMessages.slice(0, -1),
      summaryMessage,
    ]);
  };
  
  
  


  return (
    <div className="chatbox">
      <div className="chat-logs" ref={chatMessageRef}>
        {messages.map((message, index) => (
          <Message key={index} message={message} />
        ))}
      </div>
      <AppBar position="fixed" color="primary" style={{ top: 'auto', bottom: 0 }}>
        <Box display="flex" alignItems="center" p={1}>
          <TextField
            value={inputMessage}
            className="chat-input"
            fullWidth
            variant="outlined"
            placeholder="Type your message..."
            onKeyDown={handleKeyDown}
            onChange={handleInputMessageChange}
          />
  
          <Button
            className="chat-input-btn"
            variant="contained"
            color="secondary"
            onClick={handleInputMessageSubmit}
            style={{ marginLeft: '10px' }}
          >
            <i
              className="fa fa-paper-plane chat-input-send-icon"
              aria-hidden="true"
            ></i>
            Send
          </Button>
  
          <Button
            className="chat-input-btn"
            variant="contained"
            color="secondary"
            onClick={handleSummarize}
            style={{ marginLeft: '10px' }}
          >
            <i
              className="fa fa-book chat-input-send-icon"
              aria-hidden="true"
            ></i>
            Summarize
          </Button>
        </Box>
      </AppBar>
    </div>
  );
  
  
};

export default ChatInterface;
