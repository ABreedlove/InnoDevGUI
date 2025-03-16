import { Node } from 'vis-network';

interface DeviceState {
    temperature: number;
    humidity: number;
    doorStatus: string;
    tempWarningHigh: number;
    tempWarningLow: number;
    tempCriticalHigh: number;
    tempCriticalLow: number;
    humidityWarningHigh: number;
    humidityWarningLow: number;
    humidityCriticalHigh: number;
    humidityCriticalLow: number;
    doorTimeWarning: number;
}

export const createDeviceSVG = (node: Node, state: DeviceState): string => {
    // Calculate bar heights based on values
    const tempHeight = calculateBarHeight(state.temperature, 0, 50); // 0-50°C range
    const humidHeight = calculateBarHeight(state.humidity, 0, 100); // 0-100% range
    const doorHeight = calculateBarHeight(state.doorStatus === 'OPEN' ? 100 : 0, 0, 100);

    // Calculate warning marker positions
    const tempWarningHighY = 200 - calculateBarHeight(state.tempWarningHigh, 0, 50);
    const tempWarningLowY = 200 - calculateBarHeight(state.tempWarningLow, 0, 50);
    const tempCriticalHighY = 200 - calculateBarHeight(state.tempCriticalHigh, 0, 50);
    const tempCriticalLowY = 200 - calculateBarHeight(state.tempCriticalLow, 0, 50);
    const humidWarningHighY = 200 - calculateBarHeight(state.humidityWarningHigh, 0, 100);
    const humidWarningLowY = 200 - calculateBarHeight(state.humidityWarningLow, 0, 100);
    const humidCriticalHighY = 200 - calculateBarHeight(state.humidityCriticalHigh, 0, 100);
    const humidCriticalLowY = 200 - calculateBarHeight(state.humidityCriticalLow, 0, 100);
    const doorTimeY = 200 - calculateBarHeight(state.doorTimeWarning, 0, 100);

    // Determine status colors
    const tempColor = getStatusColor(
        state.temperature,
        state.tempWarningHigh,
        state.tempWarningLow,
        state.tempCriticalHigh,
        state.tempCriticalLow
    );
    const humidColor = getStatusColor(
        state.humidity,
        state.humidityWarningHigh,
        state.humidityWarningLow,
        state.humidityCriticalHigh,
        state.humidityCriticalLow
    );
    const doorColor = state.doorStatus === 'OPEN' ? '#ff4757' : '#26de81';

    return `
    <svg viewBox="0 0 240 340" xmlns="http://www.w3.org/2000/svg">
        <defs>
            ${getGradientsAndFilters()}
        </defs>

        <!-- Main device body -->
        <rect x="10" y="10" width="220" height="320" rx="15"
              fill="url(#metallic)" stroke="#555" stroke-width="2"
              filter="url(#shadow)"/>

        <!-- Panels -->
        <rect x="20" y="20" width="200" height="220" rx="10"
              fill="url(#panelGradient)"/>
        <rect x="20" y="250" width="200" height="70" rx="10"
              fill="url(#panelGradient)"/>

        <!-- Bars -->
        <g class="bars">
            <!-- Temperature Bar -->
            <g transform="translate(35,25)">
                <rect class="bar-bg" x="0" y="0" width="40" height="200" rx="3"
                      fill="#f0f0f0" stroke="#666" stroke-width="2"/>
                <rect class="bar-fill" x="2" y="${200 - tempHeight}" width="36" height="${tempHeight}" rx="2"
                      fill="${tempColor}"/>
                <!-- High warning/critical markers -->
                <path d="M 20,${tempWarningHighY} l -5,-5 l 10,0 z" fill="#ff9f43"/>
                <path d="M 20,${tempCriticalHighY} l -5,-5 l 10,0 z" fill="#ff4757"/>
                <!-- Low warning/critical markers -->
                <path d="M 20,${tempWarningLowY} l -5,5 l 10,0 z" fill="#ff9f43"/>
                <path d="M 20,${tempCriticalLowY} l -5,5 l 10,0 z" fill="#ff4757"/>
            </g>

            <!-- Humidity Bar -->
            <g transform="translate(100,25)">
                <rect class="bar-bg" x="0" y="0" width="40" height="200" rx="3"
                      fill="#f0f0f0" stroke="#666" stroke-width="2"/>
                <rect class="bar-fill" x="2" y="${200 - humidHeight}" width="36" height="${humidHeight}" rx="2"
                      fill="${humidColor}"/>
                <!-- High warning/critical markers -->
                <path d="M 20,${humidWarningHighY} l -5,-5 l 10,0 z" fill="#ff9f43"/>
                <path d="M 20,${humidCriticalHighY} l -5,-5 l 10,0 z" fill="#ff4757"/>
                <!-- Low warning/critical markers -->
                <path d="M 20,${humidWarningLowY} l -5,5 l 10,0 z" fill="#ff9f43"/>
                <path d="M 20,${humidCriticalLowY} l -5,5 l 10,0 z" fill="#ff4757"/>
            </g>

            <!-- Door Time Bar -->
            <g transform="translate(165,25)">
                <rect class="bar-bg" x="0" y="0" width="40" height="200" rx="3"
                      fill="#f0f0f0" stroke="#666" stroke-width="2"/>
                <rect class="bar-fill" x="2" y="${200 - doorHeight}" width="36" height="${doorHeight}" rx="2"
                      fill="${doorColor}"/>
                <path d="M 20,${doorTimeY} l -5,-5 l 10,0 z" fill="#ff4757"/>
            </g>
        </g>

        <!-- Reading Circles -->
        <g class="readings" transform="translate(0,285)">
            <!-- Temperature -->
            <g transform="translate(55,0)">
                <circle r="18" fill="${tempColor}" stroke="#666" stroke-width="1"/>
                <text y="5" text-anchor="middle" fill="white" 
                      font-family="Arial" font-weight="bold">${state.temperature}°C</text>
            </g>

            <!-- Humidity -->
            <g transform="translate(120,0)">
                <circle r="18" fill="${humidColor}" stroke="#666" stroke-width="1"/>
                <text y="5" text-anchor="middle" fill="white" 
                      font-family="Arial" font-weight="bold">${state.humidity}%</text>
            </g>

            <!-- Door Status -->
            <g transform="translate(185,0)">
                <circle r="18" fill="${doorColor}" stroke="#666" stroke-width="1"/>
                <text y="5" text-anchor="middle" fill="white" 
                      font-family="Arial" font-weight="bold" font-size="smaller">${state.doorStatus}</text>
            </g>
        </g>
    </svg>`;
};

const calculateBarHeight = (value: number, min: number, max: number): number => {
    const percentage = (value - min) / (max - min);
    return Math.max(0, Math.min(1, percentage)) * 200; // 200 is the bar height
};

const getStatusColor = (value: number, warningHigh: number, warningLow: number, criticalHigh: number, criticalLow: number): string => {
    if (value >= criticalHigh || value <= criticalLow) return '#ff4757';  // Red for critical
    if (value >= warningHigh || value <= warningLow) return '#ff9f43';    // Orange for warning
    return '#26de81';  // Green for normal
};

const getGradientsAndFilters = (): string => `
    <linearGradient id="metallic" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" style="stop-color:#8a8a8a" />
        <stop offset="50%" style="stop-color:#666666" />
        <stop offset="100%" style="stop-color:#8a8a8a" />
    </linearGradient>

    <linearGradient id="panelGradient" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" style="stop-color:#e0e0e0" />
        <stop offset="100%" style="stop-color:#d0d0d0" />
    </linearGradient>

    <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">
        <feGaussianBlur in="SourceAlpha" stdDeviation="2" />
        <feOffset dx="1" dy="1" />
        <feComposite in2="SourceAlpha" operator="arithmetic" k2="-1" k3="1" />
        <feComposite in2="SourceGraphic" operator="over" />
    </filter>
`;
