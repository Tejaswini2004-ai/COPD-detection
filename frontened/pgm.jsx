// import React, { useState } from 'react';

// const COPDForm = () => {
//   const [formData, setFormData] = useState({
//     age: '',
//     smoke: '',
//     rs10007052: '',
//     rs8192288: '',
//     rs20541: '',
//     alcoholConsumption: false,
//     exerciseRegularly: false,
//   });

//   const [predictionResult, setPredictionResult] = useState(null);

//   // Handle input field changes
//   const handleChange = (e) => {
//     setFormData({
//       ...formData,
//       [e.target.name]: e.target.type === 'checkbox' ? e.target.checked : e.target.value,
//     });
//   };

//   // Handle form submission to send data to backend
//   const handleSubmit = async (e) => {
//     e.preventDefault();
    
//     // Sending data to Flask backend for prediction
//     const response = await fetch('http://127.0.0.1:5000/predict', {
//       method: 'POST',
//       headers: { 'Content-Type': 'application/json' },
//       body: JSON.stringify(formData),
//     });

//     const result = await response.json();
//     setPredictionResult(result); // Updating the result state with prediction
//   };

//   return (
//     <div>
//       <h1>COPD Prediction Form</h1>
//       <form onSubmit={handleSubmit}>
//         <label>Age:
//           <input
//             type="number"
//             name="age"
//             value={formData.age}
//             onChange={handleChange}
//             placeholder="Enter your age"
//             required
//           />
//         </label>
//         <br />

//         <label>Smoking Status:
//           <select name="smoke" value={formData.smoke} onChange={handleChange} required>
//             <option value="">Select</option>
//             <option value="Yes">Yes</option>
//             <option value="No">No</option>
//           </select>
//         </label>
//         <br />

//         <label>Genetic Marker rs10007052:
//           <input
//             type="text"
//             name="rs10007052"
//             value={formData.rs10007052}
//             onChange={handleChange}
//             placeholder="Enter genetic marker value"
//             required
//           />
//         </label>
//         <br />

//         <label>Genetic Marker rs8192288:
//           <input
//             type="text"
//             name="rs8192288"
//             value={formData.rs8192288}
//             onChange={handleChange}
//             placeholder="Enter genetic marker value"
//             required
//           />
//         </label>
//         <br />

//         <label>Genetic Marker rs20541:
//           <input
//             type="text"
//             name="rs20541"
//             value={formData.rs20541}
//             onChange={handleChange}
//             placeholder="Enter genetic marker value"
//             required
//           />
//         </label>
//         <br />

//         {/* Additional habits */}
//         <label>Alcohol Consumption:
//           <input
//             type="checkbox"
//             name="alcoholConsumption"
//             checked={formData.alcoholConsumption}
//             onChange={handleChange}
//           />
//         </label>
//         <br />

//         <label>Exercise Regularly:
//           <input
//             type="checkbox"
//             name="exerciseRegularly"
//             checked={formData.exerciseRegularly}
//             onChange={handleChange}
//           />
//         </label>
//         <br />

//         <button type="submit">Submit</button>
//       </form>

//       {predictionResult && (
//         <div>
//           <h2>Prediction Result: {predictionResult.class}</h2>
//         </div>
//       )}
//     </div>
//   );
// };

// export default COPDForm;
import React from 'react';
import PatientForm from './components/PatientForm';

function App() {
  return (
    <div className="app">
      <PatientForm />
    </div>
  );
}

export default App;
