interface GeneralPropertiesProps {
    date: string;
    name: string;
    organization: string;
    followUpIns?: string;
  }
  
  const GeneralProperties: React.FC<GeneralPropertiesProps> = ({ date, name, organization, followUpIns }) => {
    return (
      <div className="general-properties">
        <div><strong>Date:</strong> {date}</div>
        <div><strong>Name:</strong> {name}</div>
        <div><strong>Organization:</strong> {organization}</div>
        {followUpIns && <div><strong>Follow-Up Instructions:</strong> {followUpIns}</div>}
      </div>
    );
  }
  

  export default GeneralProperties;
  
