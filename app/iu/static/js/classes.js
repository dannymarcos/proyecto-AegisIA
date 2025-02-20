document.addEventListener('DOMContentLoaded', () => {
    const uploadClassForm = document.getElementById('upload-class-form');
    const generateLinkForm = document.getElementById('generate-link-form');
    const accessForm = document.getElementById('access-form');
    const generatedLinkDiv = document.getElementById('generated-link');
    const accessLinkInput = document.getElementById('access-link');
    const classesList = document.getElementById('classes-list');
    
    if (uploadClassForm) {
        uploadClassForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(uploadClassForm);
            
            try {
                const response = await fetch('/upload_class', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                if (data.status === 'success') {
                    alert('Class uploaded successfully!');
                    uploadClassForm.reset();
                } else {
                    alert(data.message || 'Failed to upload class');
                }
            } catch (error) {
                console.error('Error uploading class:', error);
                alert('An error occurred while uploading the class');
            }
        });
    }
    
    if (generateLinkForm) {
        generateLinkForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(generateLinkForm);
            
            try {
                const response = await fetch('/generate_access_link', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                if (data.status === 'success') {
                    generatedLinkDiv.classList.remove('hidden');
                    accessLinkInput.value = data.access_link;
                    generateLinkForm.reset();
                } else {
                    alert(data.message || 'Failed to generate access link');
                }
            } catch (error) {
                console.error('Error generating access link:', error);
                alert('An error occurred while generating the access link');
            }
        });
    }
    
    const requestAccessForm = document.getElementById('request-access-form');
    if (requestAccessForm) {
        requestAccessForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const reason = document.getElementById('request-reason').value;
            
            try {
                const response = await fetch('/request_teleclass_access', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ reason })
                });
                
                const data = await response.json();
                if (data.status === 'success') {
                    alert('Your request has been submitted successfully. You will be notified when access is granted.');
                    requestAccessForm.reset();
                } else {
                    alert(data.message || 'Failed to submit request');
                }
            } catch (error) {
                console.error('Error submitting access request:', error);
                alert('An error occurred while submitting your request');
            }
        });
    }
});

function copyLink() {
    const accessLink = document.getElementById('access-link');
    accessLink.select();
    document.execCommand('copy');
    alert('Link copied to clipboard!');
}

function toggleReferralMenu() {
    const menu = document.getElementById('referral-menu');
    menu.classList.toggle('hidden');
}

function loadClasses(classes) {
    const classesList = document.getElementById('classes-list');
    classesList.innerHTML = '';
    
    classes.forEach(classItem => {
        const classCard = document.createElement('div');
        classCard.className = 'class-card bg-white rounded-lg shadow-md p-4';
        classCard.innerHTML = `
            <h3 class="font-medium mb-2">${classItem.title}</h3>
            <p class="text-gray-600 mb-4">${classItem.description}</p>
            <video controls class="w-full mb-4">
                <source src="${classItem.video_url}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        `;
        classesList.appendChild(classCard);
    });
}