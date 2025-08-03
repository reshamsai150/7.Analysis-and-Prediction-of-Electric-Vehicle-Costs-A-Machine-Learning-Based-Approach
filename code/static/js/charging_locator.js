// code/static/js/charging_locator.js

// Initialize the map
function initMap(lat = 28.6139, lon = 77.2090) { // Default: Delhi
    const map = L.map('map').setView([lat, lon], 12);

    // Tile Layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);

    return map;
}

// Plot stations on the map
function plotStations(map, stations) {
    stations.forEach(station => {
        const marker = L.marker([station.lat, station.lon]).addTo(map);
        marker.bindPopup(`
            <b>${station.title}</b><br>
            ${station.address || ''}<br>
            ${station.town || ''}, ${station.state || ''} ${station.postcode || ''}
        `);
    });
}

// Fetch stations dynamically (Flask route returns JSON)
async function fetchStations(lat, lon, distance) {
    const response = await fetch(`/api/get-stations?lat=${lat}&lon=${lon}&distance=${distance}`);
    const data = await response.json();
    return data;
}
