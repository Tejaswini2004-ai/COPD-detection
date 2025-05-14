// import React, { useState } from 'react';

// function PatientForm() {
//     const [formData, setFormData] = useState({
//         sex: 1, // Default Male
//         age: '',
//         bmi: '',
//         smoke: 0, // Default No Smoke
//         rs10007052: '',
//         rs8192288: '',
//         rs20541: '',
//         rs12922394: '',
//         rs2910164: '',
//         rs161976: '',
//         rs473892: '',
//         rs159497: '',
//         rs9296092: ''
//     });

//     const [prediction, setPrediction] = useState('');
//     const [probability, setProbability] = useState('');

//     const handleChange = (e) => {
//         setFormData({
//             ...formData,
//             [e.target.name]: e.target.value
//         });
//     };

//     const handleSubmit = async (e) => {
//         e.preventDefault();
        
//         // Send POST request to Flask backend
//         const response = await fetch('http://127.0.0.1:5000/predict', {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json'
//             },
//             body: JSON.stringify(formData)
//         });
        
//         const data = await response.json();
//         if (data.prediction) {
//             setPrediction(data.prediction);
//             setProbability(data.probability);
//         } else {
//             alert('Error: ' + data.error);
//         }
//     };

//     return (
//         <div>
//             <h2>Enter Patient Details</h2>
//             <form onSubmit={handleSubmit}>
//                 <label>
//                     Sex (1=Male, 2=Female):
//                     <input type="number" name="sex" value={formData.sex} onChange={handleChange} />
//                 </label><br />

//                 <label>
//                     Age:
//                     <input type="number" name="age" value={formData.age} onChange={handleChange} />
//                 </label><br />

//                 <label>
//                     BMI:
//                     <input type="number" name="bmi" value={formData.bmi} onChange={handleChange} />
//                 </label><br />

//                 <label>
//                     Smoke (0=No, 1=Yes):
//                     <input type="number" name="smoke" value={formData.smoke} onChange={handleChange} />
//                 </label><br />

//                 {/* Add other input fields for rs10007052, rs8192288, etc. */}
//                 <label>
//                     rs10007052:
//                     <input type="number" name="rs10007052" value={formData.rs10007052} onChange={handleChange} />
//                 </label><br />

//                 {/* Repeat for other genetic factors... */}

//                 <button type="submit">Predict COPD Risk</button>
//             </form>

//             {prediction && (
//                 <div>
//                     <h3>Prediction: {prediction}</h3>
//                     <p>Probability: {probability.toFixed(2)}%</p>
//                 </div>
//             )}
//         </div>
//     );
// }

// export default PatientForm;





// last used code
// import React, { useState } from 'react';

// function PatientForm() {
//     const [formData, setFormData] = useState({
//         sex: 1,
//         age: '',
//         bmi: '',
//         smoke: 0,
//         rs10007052: '',
//         rs8192288: '',
//         rs20541: '',
//         rs12922394: '',
//         rs2910164: '',
//         rs161976: '',
//         rs473892: '',
//         rs159497: '',
//         rs9296092: ''
//     });

//     const [prediction, setPrediction] = useState('');
//     const [probability, setProbability] = useState('');

//     const handleChange = (e) => {
//         setFormData({
//             ...formData,
//             [e.target.name]: e.target.value
//         });
//     };

//     const handleSubmit = async (e) => {
//         e.preventDefault();

//         try {
//             const response = await fetch('http://127.0.0.1:5000/predict', {
//                 method: 'POST',
//                 headers: {
//                     'Content-Type': 'application/json'
//                 },
//                 body: JSON.stringify(formData)
//             });

//             const data = await response.json();
//             console.log("Response from backend:", data);

//             if (data.prediction !== undefined) {
//                 setPrediction(data.prediction);
//                 setProbability(data.probability);
//             } else {
//                 alert('Error in response: ' + JSON.stringify(data));
//             }
//         } catch (err) {
//             alert('Request failed: ' + err.message);
//         }
//     };

//     return (
//         <div>
//             <h2>Enter Patient Details</h2>
//             <form onSubmit={handleSubmit}>
//                 {Object.keys(formData).map((key) => (
//                     <div key={key}>
//                         <label>
//                             {key}:&nbsp;
//                             <input
//                                 type="number"
//                                 name={key}
//                                 value={formData[key]}
//                                 onChange={handleChange}
//                             />
//                         </label><br />
//                     </div>
//                 ))}
//                 <button type="submit">Predict COPD Risk</button>
//             </form>

//             {prediction !== '' && (
//                 <div>
//                     <h3>Prediction: {prediction === 1 ? 'At Risk' : 'Not at Risk'}</h3>
//                     <p>Probability: {Number(probability).toFixed(2)}%</p>
//                 </div>
//             )}
//         </div>
//     );
// }

// export default PatientForm;





// import React, { useState } from 'react';

// function PatientForm() {
   
//     const [formData, setFormData] = useState({
//         sex: 1,
//         age: '',
//         bmi: '',
//         smoke: 0,
//         rs10007052: '',
//         rs8192288: '',
//         rs20541: '',
//         rs12922394: '',
//         rs2910164: '',
//         rs161976: '',
//         rs473892: '',
//         rs159497: '',
//         rs9296092: '',
//         alcohol_consumption: '',  // New field
//         exercise_regularly: ''    // New field
//     });
    

//     const [prediction, setPrediction] = useState('');
//     const [probability, setProbability] = useState('');

//     const handleChange = (e) => {
//         setFormData({
//             ...formData,
//             [e.target.name]: e.target.value
//         });
//     };

//     const handleSubmit = async (e) => {
//         e.preventDefault();

//         try {
//             const response = await fetch('http://127.0.0.1:5000/predict', {
//                 method: 'POST',
//                 headers: {
//                     'Content-Type': 'application/json'
//                 },
//                 body: JSON.stringify(formData)
//             });

//             const data = await response.json();
//             console.log("Response from backend:", data);

//             if (data.prediction !== undefined) {
//                 setPrediction(data.prediction);  // either 'COPD' or 'No COPD'
//                 setProbability(data.probability);
//             } else {
//                 alert('Error in response: ' + JSON.stringify(data));
//             }
//         } catch (err) {
//             alert('Request failed: ' + err.message);
//         }
//     };

//     return (
//         <div>
//             <h2>Enter Patient Details</h2>
//             <form onSubmit={handleSubmit}>
//                 {Object.keys(formData).map((key) => (
//                     <div key={key}>
//                         <label>
//                             {key}:
//                             <input
//                                 type="number"
//                                 name={key}
//                                 value={formData[key]}
//                                 onChange={handleChange}
//                             />
//                         </label><br />
//                     </div>
//                 ))}
//                 <button type="submit">Predict COPD Risk</button>
//             </form>

//             {prediction && (
//                 <div>
//                     <h3>Prediction: {prediction === 'COPD' ? 'At Risk' : 'Not at Risk'}</h3>
//                     <p>Probability: {Number(probability).toFixed(2)}%</p>
//                 </div>
//             )}
//         </div>
//     );
// }

// export default PatientForm;

//correct code
// 



import React, { useState } from 'react';

function PatientForm() {
    const [formData, setFormData] = useState({
        sex: 1,
        age: '',
        bmi: '',
        smoke: 0,
        rs10007052: '',
        rs8192288: '',
        rs20541: '',
        rs12922394: '',
        rs2910164: '',
        rs161976: '',
        rs473892: '',
        rs159497: '',
        rs9296092: '',
        alcohol_consumption: '',
        exercise_regularly: ''
    });

    const [prediction, setPrediction] = useState('');
    const [probability, setProbability] = useState('');

    const handleChange = (e) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const data = await response.json();
            console.log("Response from backend:", data);

            if (data.prediction !== undefined) {
                setPrediction(data.prediction);
                setProbability(data.probability);
            } else {
                alert('Error in response: ' + JSON.stringify(data));
            }
        } catch (err) {
            alert('Request failed: ' + err.message);
        }
    };

    return (
        <div>
            <h2>Enter Patient Details</h2>
            <form onSubmit={handleSubmit}>
                {Object.keys(formData).map((key) => (
                    <div key={key}>
                        <label>
                            {key}:&nbsp;
                            <input
                                type="number"
                                name={key}
                                value={formData[key]}
                                onChange={handleChange}
                            />
                        </label><br />
                    </div>
                ))}
                <button type="submit">Predict COPD Risk</button>
            </form>

            {prediction !== '' && (
                <div>
                    <h3>Prediction: {prediction === 1 ? 'At Risk' : 'Not at Risk'}</h3>
                    <p>Probability: {Number(probability).toFixed(2)}%</p>
                </div>
            )}
        </div>
    );
}

export default PatientForm;
