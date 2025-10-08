// Login functionality
document.getElementById('loginForm')?.addEventListener('submit', function(event) {
    event.preventDefault();
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    if (username === 'admin' && password === 'admin') {
        window.location.href = 'dashboard.html';
    } else {
        alert('Invalid username or password');
    }
});

// Logout functionality
document.getElementById('logoutButton')?.addEventListener('click', function() {
    window.location.href = 'index.html';
});

// Waste data management
const wasteTypeForm = document.getElementById('wasteTypeForm');
const wasteList = document.getElementById('wasteList');
const fillLevel = document.getElementById('fillLevel');
const fillProgress = document.getElementById('fillProgress');
const totalWaste = document.getElementById('totalWaste');
const wasteGaugeCanvas = document.getElementById('wasteGaugeChart');

let wasteEntries = JSON.parse(localStorage.getItem('wasteEntries')) || [];
let wasteGaugeChart;

function renderWasteEntries() {
    wasteList.innerHTML = '';
    let totalVolume = 0;
    const typeCounts = { organic: 0, plastic: 0, paper: 0 };

    wasteEntries.forEach((entry, index) => {
        const listItem = document.createElement('li');
        listItem.className = 'flex justify-between items-center bg-gray-50 p-2 rounded-lg shadow-md';

        const entryText = document.createElement('span');
        entryText.textContent = `${entry.type} - ${entry.volume}L - ${new Date(entry.timestamp).toLocaleString()}`;
        listItem.appendChild(entryText);

        const deleteButton = document.createElement('button');
        deleteButton.textContent = 'Delete';
        deleteButton.className = 'bg-red-500 text-white px-2 py-1 rounded-lg hover:bg-red-600';
        deleteButton.addEventListener('click', () => {
            wasteEntries.splice(index, 1);
            saveWasteEntries();
            renderWasteEntries();
        });
        listItem.appendChild(deleteButton);

        wasteList.appendChild(listItem);

        totalVolume += entry.volume; // Ensure totalVolume is updated correctly
        typeCounts[entry.type] += entry.volume;
    });

    const fillPercentage = Math.min((totalVolume / 100) * 100, 100).toFixed(2);
    fillLevel.textContent = `${fillPercentage}%`;
    fillProgress.style.width = `${fillPercentage}%`;
    totalWaste.textContent = `${totalVolume}L`; // Update total waste display

    if (fillPercentage >= 80) {
        alert('Bin fill level is 80% or more!');
    }

    updateWasteGauge(fillPercentage);
}

function updateWasteGauge() {
    const organicGauge = document.getElementById('organicGauge');
    const plasticGauge = document.getElementById('plasticGauge');
    const paperGauge = document.getElementById('paperGauge');

    const typeCounts = { organic: 0, plastic: 0, paper: 0 };
    wasteEntries.forEach(entry => {
        typeCounts[entry.type] += entry.volume;
    });

    const totalVolume = Object.values(typeCounts).reduce((a, b) => a + b, 0);
    const totalPercentage = Math.min((totalVolume / 100) * 100, 100);

    const organicPercentage = (typeCounts.organic / totalVolume) * totalPercentage || 0;
    const plasticPercentage = (typeCounts.plastic / totalVolume) * totalPercentage || 0;
    const paperPercentage = (typeCounts.paper / totalVolume) * totalPercentage || 0;

    organicGauge.style.width = `${organicPercentage}%`;
    organicGauge.style.left = `0%`;
    organicGauge.setAttribute('title', 'Organic');

    plasticGauge.style.width = `${plasticPercentage}%`;
    plasticGauge.style.left = `${organicPercentage}%`;
    plasticGauge.setAttribute('title', 'Plastic');

    paperGauge.style.width = `${paperPercentage}%`;
    paperGauge.style.left = `${organicPercentage + plasticPercentage}%`;
    paperGauge.setAttribute('title', 'Paper');

    const wasteStats = document.getElementById('wasteStats');
    wasteStats.textContent = `Organic: ${typeCounts.organic}L, Plastic: ${typeCounts.plastic}L, Paper: ${typeCounts.paper}L`;
}

function saveWasteEntries() {
    localStorage.setItem('wasteEntries', JSON.stringify(wasteEntries));
}

wasteTypeForm?.addEventListener('submit', (event) => {
    event.preventDefault();
    const type = document.getElementById('wasteType').value;

    // Simulate bin unlocking and weight calculation
    alert(`Bin unlocked for ${type} waste. Please throw your trash.`);

    const simulatedWeight = Math.floor(Math.random() * 5) + 1; // Random weight between 1-5 liters
    wasteEntries.push({ type, volume: simulatedWeight, timestamp: Date.now() });
    saveWasteEntries();
    renderWasteEntries();
});

// Initial render
renderWasteEntries();