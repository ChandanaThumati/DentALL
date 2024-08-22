// Load the JSON data
fetch('/static/information.json')

    .then(response => response.json())
    .then(data => {
        // Get the information section element
        const infoSection = document.getElementById('information-content');

        // Function to update the information section
        function updateInfo(label) {
            // Get the information for the label from the data
            const info = data[label];
            console.log('info:', info);
            // Update the content of the information section
            
            if (info) {
                infoSection.innerHTML = `
                  <h2>${info.title}</h2>
                  <p>${info.description}</p>
                  <p>${info.remedies}</p>
                  <p>${info.conclusion}</p>
                `;
              } else {
                infoSection.innerHTML = '<p>No information available for this label.</p>';
              }
        }

        // Call the updateInfo function with the initial label
        const defaultLabel = label;
        console.log('defaultLabel:', defaultLabel);
        updateInfo(defaultLabel);

        // Add click event listeners to the labels
        const labels = document.querySelectorAll('.label');
        labels.forEach(label => {
            label.addEventListener('click', () => {
                const selectedLabel = label.dataset.label;
                updateInfo(selectedLabel);
                document.querySelector('.active').classList.remove('active');
                label.classList.add('active');
            });
        });
    })
    .catch(error => {
        console.error('Error loading JSON data:', error);
    });
