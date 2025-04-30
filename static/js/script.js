document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        
        if (result.error) {
            document.getElementById('results').innerHTML = `<p style="color: red;">Error: ${result.error}</p>`;
            return;
        }
        
        let output = '<h3>Predictions:</h3><ul>';
        for (const [model, prediction] of Object.entries(result)) {
            output += `<li>${model}: ${prediction}</li>`;
        }
        output += '</ul>';
        document.getElementById('results').innerHTML = output;
    } catch (error) {
        document.getElementById('results').innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
    }
});