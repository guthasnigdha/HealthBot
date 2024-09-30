document.getElementById('register-button').addEventListener('click', register);
document.getElementById('login-button').addEventListener('click', login);
document.getElementById('send-button').addEventListener('click', sendMessage);
document.getElementById('logout-button').addEventListener('click', logout);
document.getElementById('go-to-register-button').addEventListener('click', showRegisterPage);
document.getElementById('go-to-login-button').addEventListener('click', showLoginPage);
document.getElementById('reg-username').addEventListener('keypress', handleKeyPressRegister);
document.getElementById('reg-password').addEventListener('keypress', handleKeyPressRegister);
document.getElementById('username').addEventListener('keypress', handleKeyPressLogin);
document.getElementById('password').addEventListener('keypress', handleKeyPressLogin);
document.getElementById('user-input').addEventListener('keypress', handleKeyPressSend);

let diseaseDatabase = null;
let pendingSymptomConfirmation = null;

function loadDatabase() {
    fetch('disease_data.json')
        .then(response => response.json())
        .then(data => {
            diseaseDatabase = data.diseases;
        })
        .catch(error => console.error('Error loading database:', error));
}

function handleKeyPressRegister(event) {
    if (event.key === 'Enter') {
        register();
    }
}

function handleKeyPressLogin(event) {
    if (event.key === 'Enter') {
        login();
    }
}

function handleKeyPressSend(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

function validatePassword(password) {
    const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/;
    return passwordRegex.test(password);
}

function register() {
    const username = document.getElementById('reg-username').value;
    const password = document.getElementById('reg-password').value;

    if (username && password) {
        if (!validatePassword(password)) {
            alert('Password must be at least 8 characters long and include at least one uppercase letter, one lowercase letter, one digit, and one special character.');
            return;
        }
        const users = JSON.parse(localStorage.getItem('users')) || {};
        if (users[username]) {
            alert('Username already exists. Please choose a different username.');
        } else {
            users[username] = password;
            localStorage.setItem('users', JSON.stringify(users));
            alert('Registration successful! You can now log in.');
            showLoginPage();
        }
    } else {
        alert('Please enter both username and password.');
    }
}

function login() {
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    const users = JSON.parse(localStorage.getItem('users')) || {};

    if (users[username] && users[username] === password) {
        document.getElementById('login-page').classList.add('hidden');
        document.getElementById('chat-page').classList.remove('hidden');
    } else {
        alert('Invalid username or password.');
    }
}

function logout() {
    document.getElementById('login-page').classList.remove('hidden');
    document.getElementById('chat-page').classList.add('hidden');
}

function showRegisterPage() {
    document.getElementById('login-page').classList.add('hidden');
    document.getElementById('register-page').classList.remove('hidden');
}

function showLoginPage() {
    document.getElementById('register-page').classList.add('hidden');
    document.getElementById('login-page').classList.remove('hidden');
}

function sendMessage() {
    const userInput = document.getElementById('user-input');
    const userMessage = userInput.value.trim();

    if (userMessage) {
        appendMessage(userMessage, 'user-message');
        userInput.value = '';

        if (pendingSymptomConfirmation) {
            if (userMessage.toLowerCase() === 'yes' || userMessage.toLowerCase() === 'y') {
                appendMessage(`Confirmed symptom: ${pendingSymptomConfirmation}`, 'bot-message');
                getBotResponse(pendingSymptomConfirmation, true);
            } else {
                appendMessage(`Did not confirm the symptom. Please re-enter your symptom.`, 'bot-message');
            }
            pendingSymptomConfirmation = null;
        } else {
            getBotResponse(userMessage, false);
        }
    }
}

function appendMessage(message, className) {
    const chatBox = document.getElementById('chat-box');
    const messageElement = document.createElement('div');
    messageElement.className = 'message ' + className;
    messageElement.textContent = message;
    chatBox.appendChild(messageElement);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function getBotResponse(userMessage, isConfirmed) {
    let symptoms = [];
    if (isConfirmed) {
        symptoms = [userMessage];
    } else {
        symptoms = parseUserInput(userMessage);
    }
    
    let possibleDiagnoses = [];
    let doctors = '';
    let unrecognizedSymptoms = [];

    for (const symptom of symptoms) {
        if (!diseaseDatabase.some(disease => disease.symptoms.includes(symptom))) {
            unrecognizedSymptoms.push(symptom);
        }
    }

    if (unrecognizedSymptoms.length > 0 && !isConfirmed) {
        let suggestionMessages = [];
        for (const unrecognizedSymptom of unrecognizedSymptoms) {
            const closestSymptom = getClosestSymptom(unrecognizedSymptom);
            pendingSymptomConfirmation = closestSymptom;
            suggestionMessages.push(`Did you mean "${closestSymptom}" for "${unrecognizedSymptom}"?`);
        }
        appendMessage(suggestionMessages.join(' '), 'bot-message');
        return;
    }

    for (const disease of diseaseDatabase) {
        if (symptoms.every(symptom => disease.symptoms.includes(symptom))) {
            possibleDiagnoses.push(disease.name);
            if (doctors === '') {
                doctors = `Doctors to consult: ${disease.doctors.join(', ')}`;
            }
        }
    }

    let diagnosisMessage = 'Unable to determine the disease based on provided symptoms. Please consult a healthcare provider.';
    if (possibleDiagnoses.length > 0) {
        diagnosisMessage = `Possible Diagnoses: ${possibleDiagnoses.join(', ')}`;
    }

    setTimeout(() => {
        appendMessage(diagnosisMessage, 'bot-message');
        if (doctors) {
            appendMessage(doctors, 'bot-message');
        }
    }, 1000);
}

function parseUserInput(userMessage) {
    const symptomRegex = /(?:^|\W)i\s+have\s+(.*?)(?:\W|$)/gi;
    const matches = symptomRegex.exec(userMessage.toLowerCase());
    if (matches && matches.length > 1) {
        const symptom = matches[1].trim();
        return [symptom];
    }
    return userMessage.toLowerCase().split(',').map(symptom => symptom.trim());
}

function levenshtein(a, b) {
    const matrix = [];

    for (let i = 0; i <= b.length; i++) {
        matrix[i] = [i];
    }

    for (let j = 0; j <= a.length; j++) {
        matrix[0][j] = j;
    }

    for (let i = 1; i <= b.length; i++) {
        for (let j = 1; j <= a.length; j++) {
            if (b.charAt(i - 1) === a.charAt(j - 1)) {
                matrix[i][j] = matrix[i - 1][j - 1];
            } else {
                matrix[i][j] = Math.min(matrix[i - 1][j - 1] + 1, Math.min(matrix[i][j - 1] + 1, matrix[i - 1][j] + 1));
            }
        }
    }

    return matrix[b.length][a.length];
}

function getClosestSymptom(symptom) {
    const allSymptoms = new Set(diseaseDatabase.flatMap(disease => disease.symptoms));
    let closestSymptom = '';
    let closestDistance = Infinity;

    for (const s of allSymptoms) {
        const distance = levenshtein(symptom, s);
        if (distance < closestDistance) {
            closestDistance = distance;
            closestSymptom = s;
        }
    }

    return closestSymptom;
}

// Load the database when the page loads
window.onload = loadDatabase;
