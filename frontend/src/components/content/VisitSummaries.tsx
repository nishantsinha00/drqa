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

interface VisitSummariesProps extends GeneralPropertiesProps {
    diagnosis?: string;
    medications: string;
    medicationsinstructions: string;
    instructions: string;
  }
  
  const VisitSummaries: React.FC<VisitSummariesProps> = (props) => {
    return (
      <div className="visit-summaries">
        <GeneralProperties {...props} />
        {props.diagnosis && <div><strong>Diagnosis:</strong> {props.diagnosis}</div>}
        <div><strong>Medications:</strong> {props.medications}</div>
        <div><strong>Medications Instructions:</strong> {props.medicationsinstructions}</div>
        <div><strong>General Instructions:</strong> {props.instructions}</div>
      </div>
    );
  }

  export default VisitSummaries;
  