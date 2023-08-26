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
  
  interface PrescriptionPropertiesProps extends GeneralPropertiesProps {
    medications: string;
    medicationsinstructions: string;
  }
  
  const PrescriptionProperties: React.FC<PrescriptionPropertiesProps> = (props) => {
    return (
      <div className="prescription-properties">
        <GeneralProperties {...props} />
        <div><strong>Medications:</strong> {props.medications}</div>
        <div><strong>Medications Instructions:</strong> {props.medicationsinstructions}</div>
      </div>
    );
  }

  export default PrescriptionProperties;