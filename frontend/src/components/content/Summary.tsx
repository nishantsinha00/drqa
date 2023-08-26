interface SummaryProps {
    summaryText: string;
  }
  
  const Summary: React.FC<SummaryProps> = ({ summaryText }) => {
    return (
      <div className="summary">
        <h2>Summary:</h2>
        <p>{summaryText}</p>
      </div>
    );
  }

  export default Summary;
  