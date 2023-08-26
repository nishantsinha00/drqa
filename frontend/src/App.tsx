import { useState } from 'react';
import './App.css';
import { Box, Container, Typography } from '@mui/material';
import { OptionContext } from './OptionContext';
import FileUploader from './components/uploader/fileUploader';
import ChatInterface from './components/chat/ChatInterface';
import Login from './components/login/Login';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import ChatIcon from './components/chat/ChatIcon';

// Import the logo
import logo from './logo.jpg';

function App() {
  const [selectedOption, setSelectedOption] = useState('Lab Report');
  const [isChatVisible, setIsChatVisible] = useState(false);
  
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route
          path="/home"
          element={
            <div className="main-holder">
              {/* Add the logo */}
              <img src={logo} alt="Website Logo" style={{ position: 'absolute', top: '10px', left: '10px', width: '40px' }} />
              
              <Typography margin="auto" variant="h4" gutterBottom>
                {/* Content here */}
              </Typography>
              <Box my={3} mx={5}>
                <Typography variant="body1">
                  {/* Content here */}
                </Typography>
              </Box>
              <OptionContext.Provider value={{ selectedOption, setSelectedOption }}>
                
                {isChatVisible ? <ChatInterface /> : <ChatIcon onClick={() => setIsChatVisible(true)} />}
              </OptionContext.Provider>
            </div>
          }
        />
      </Routes>
    </Router>
  );
}

export default App;
