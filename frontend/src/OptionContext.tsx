import React from 'react';

export const OptionContext = React.createContext({
  selectedOption: 'Lab Report',
  setSelectedOption: (_: string) => {}
});
