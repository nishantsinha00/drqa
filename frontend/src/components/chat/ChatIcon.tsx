// ChatIcon.tsx
import React from 'react';
import './ChatIcon.css';

interface Props {
  onClick: () => void;
}

const ChatIcon: React.FC<Props> = ({ onClick }) => {
  return (
    <div className="chat-icon-container" onClick={onClick}>
      <i className="fa fa-commenting" aria-hidden="true"></i>
    </div>
  );
};

export default ChatIcon;

// ChatIcon.css

