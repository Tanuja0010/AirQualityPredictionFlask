fetch('/predict', {
    method: 'POST',
    body: formData
})
.then(response => {
    if (!response.ok) {
        throw new Error('Network response was not ok');
    }
    return response.text();
})
.then(data => {
    // Clear input fields
    this.reset();

    // Display prediction result
    document.getElementById('prediction-result').innerHTML = data;
})
.catch(error => {
    console.error('Error:', error);
    // Display error message to the user
    document.getElementById('prediction-result').innerHTML = '<p>Error fetching prediction. Please try again.</p>';
});