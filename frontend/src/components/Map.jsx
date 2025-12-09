import React from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

const Map = ({ earthquakes }) => {
    const getMagColor = (mag) => {
        if (mag < 3) return '#10b981'; // success
        if (mag < 5) return '#f59e0b'; // warning
        if (mag < 7) return '#ef4444'; // danger
        return '#7f1d1d'; // dark red
    };

    return (
        <div className="w-full h-full rounded-2xl overflow-hidden shadow-2xl border border-slate-700/50 bg-surface relative z-10">
            <MapContainer
                center={[20, 0]}
                zoom={2}
                scrollWheelZoom={true}
                style={{ height: "100%", width: "100%", zIndex: 0 }}
                className="z-0"
            >
                <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
                    url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                />

                {earthquakes.map((eq, idx) => (
                    <CircleMarker
                        key={idx}
                        center={[eq.latitude, eq.longitude]}
                        radius={Math.pow(eq.mag, 1.5)} // Exponential size
                        pathOptions={{
                            color: getMagColor(eq.mag),
                            fillColor: getMagColor(eq.mag),
                            fillOpacity: 0.6,
                            weight: 1
                        }}
                    >
                        <Popup className="custom-popup">
                            <div className="p-2 min-w-[150px]">
                                <h3 className="font-bold text-slate-900 text-lg mb-1">M {eq.mag.toFixed(1)}</h3>
                                <div className="text-slate-600 text-sm space-y-1">
                                    <p>Depth: <span className="font-medium">{eq.depth} km</span></p>
                                    <p className="text-xs text-slate-500">{new Date(eq.time).toLocaleString()}</p>
                                </div>
                            </div>
                        </Popup>
                    </CircleMarker>
                ))}
            </MapContainer>
        </div>
    );
};

export default Map;
