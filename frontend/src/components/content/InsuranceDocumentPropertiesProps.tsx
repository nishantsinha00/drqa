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


  interface InsuranceDocumentPropertiesProps extends GeneralPropertiesProps {
    ad: string;
    snf: string;
    cost: string;
    diabetesCoverage: string;
    contact: string;
  }
  
  const InsuranceDocumentProperties: React.FC<InsuranceDocumentPropertiesProps> = (props) => {
    return (
      <div className="insurance-document-properties">
        <GeneralProperties {...props} />
        <div><strong>Annual Deductible:</strong> {props.ad}</div>
        <div><strong>Skilled Nursing Facility:</strong> {props.snf}</div>
        <div><strong>Cost for PT Visits:</strong> {props.cost}</div>
        <div><strong>Diabetes Coverage:</strong> {props.diabetesCoverage}</div>
        <div><strong>Contact:</strong> {props.contact}</div>
      </div>
    );
  }
  

  export default InsuranceDocumentProperties;
  