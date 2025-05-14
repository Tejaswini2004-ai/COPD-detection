// document.getElementById('copd-form').onsubmit = async function(e) {
//     e.preventDefault();
    
//     // Show loading state
//     const predictBtn = document.querySelector('button[type="submit"]');
//     predictBtn.disabled = true;
//     predictBtn.textContent = 'Predicting...';
    
//     // Get form data
//     const formData = {
//         age: document.querySelector('input[name="age"]').value,
//         smoke: document.querySelector('input[name="smoke"]').value,
//         rs10007052: document.querySelector('input[name="rs10007052"]').value,
//         rs8192288: document.querySelector('input[name="rs8192288"]').value,
//         rs20541: document.querySelector('input[name="rs20541"]').value,
//         alcoholConsumption: document.querySelector('input[name="alcoholConsumption"]').value,
//         exerciseRegularly: document.querySelector('input[name="exerciseRegularly"]').value
//     };

//     try {
//         const response = await fetch('http://localhost:5000/predict', {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json',
//             },
//             body: JSON.stringify(formData)
//         });
        
//         const result = await response.json();
//         console.log("API Response:", result);  // Debugging
        
//         if (result.status === "success") {
//             document.getElementById('result').innerHTML = `
//                 <strong>Prediction:</strong> ${result.class}<br>
//                 <small>Raw prediction value: ${result.prediction}</small>
//             `;
//         } else {
//             document.getElementById('result').innerText = `Error: ${result.error}`;
//         }
//     } catch (error) {
//         console.error('Fetch Error:', error);
//         document.getElementById('result').innerText = 'Failed to get prediction';
//     } finally {
//         // Reset button
//         predictBtn.disabled = false;
//         predictBtn.textContent = 'Predict';
//     }
// };

// document.getElementById('predict-btn').addEventListener('click', async () => {
//     const age = document.getElementById('age').value;
//     const smoking = document.getElementById('smoking').value;
//     const genetics = document.getElementById('genetics').value;
//     const alcohol = document.getElementById('alcohol').value;
//     const diet = document.getElementById('diet').value;
//     const pollution = document.getElementById('pollution').value;

//     const response = await fetch('/predict', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json'  // REQUIRED!
//         },
//         body: JSON.stringify({
//             age: parseInt(age),
//             smoking: parseInt(smoking),
//             genetics: parseInt(genetics),
//             alcohol: parseInt(alcohol),
//             diet: parseInt(diet),
//             pollution: parseInt(pollution)
//         })
//     });

//     const result = await response.json();
//     alert("Prediction: " + result.prediction);
// });
document.getElementById('predict-btn').addEventListener('click', async () => {
    const form = document.getElementById('prediction-form');
    const formData = new FormData(form);
    const resultDiv = document.getElementById('result');
    const predictionText = document.getElementById('prediction-text');
    const accuracyText = document.getElementById('accuracy-text');
    
    // Convert form data to object
    const data = {};
    formData.forEach((value, key) => {
        data[key] = value;
    });
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        const result = await response.json();
        
        // Display results
        predictionText.textContent = result.message;
        accuracyText.textContent = `Model Confidence: ${(result.accuracy * 100).toFixed(2)}%`;
        resultDiv.style.display = 'block';
        
    } catch (error) {
        predictionText.textContent = `Error: ${error.message}`;
        accuracyText.textContent = '';
        resultDiv.style.display = 'block';
        console.error('Error:', error);
    }
});