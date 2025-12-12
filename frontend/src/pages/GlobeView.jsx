import React, { useEffect, useState, useRef } from 'react';
import Globe from 'react-globe.gl';
import axios from 'axios';

const GlobeView = () => {
    const globeEl = useRef();
    const [points, setPoints] = useState([]);
    const [rings, setRings] = useState([]);
    const [hover, setHover] = useState(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const host = window.location.hostname;
                // Fetch 2000 points - a balance between "more data" and stability
                const res = await axios.get(`http://${host}:8000/api/recent-earthquakes?limit=500`);

                let rawData = res.data;
                // Fallback if data is empty to ensure visualization works
                if (!Array.isArray(rawData) || rawData.length === 0) {
                    console.warn("API returned empty data, using sample points.");
                    rawData = [
                        { latitude: 35.6, longitude: 140.0, mag: 6.5, depth: 40, place: "Tokyo, Japan", time: Date.now() },
                        { latitude: 34.0, longitude: -118.2, mag: 5.2, depth: 10, place: "Los Angeles, CA", time: Date.now() },
                        { latitude: -33.4, longitude: -70.6, mag: 7.0, depth: 80, place: "Santiago, Chile", time: Date.now() }
                    ];
                }

                const data = rawData.map(eq => ({
                    lat: parseFloat(eq.latitude),
                    lng: parseFloat(eq.longitude),
                    size: Math.max(0.1, Math.pow(eq.mag, 2) / 50), // Ensure min size
                    color: eq.depth > 70 ? '#EF4444' : (eq.depth > 30 ? '#F59E0B' : '#10B981'), // Red deep, Green shallow
                    ...eq
                }));

                setPoints(data);

                // Add rings only for significant earthquakes (> 5.5) to keep performance high
                const significant = data.filter(d => d.mag > 5.5).map(d => ({
                    lat: d.lat,
                    lng: d.lng,
                    maxR: d.mag * 3,
                    propagationSpeed: d.mag * 0.5,
                    repeatPeriod: 1500, // Slower pulse for elegance
                    color: d.color
                }));
                setRings(significant);

            } catch (err) {
                console.error("Globe data fetch error:", err);
            }
        };

        fetchData();
    }, []);

    useEffect(() => {
        if (globeEl.current) {
            globeEl.current.controls().autoRotate = true;
            globeEl.current.controls().autoRotateSpeed = 0.5;
            globeEl.current.pointOfView({ altitude: 2.5 });
        }
    }, []);

    return (
        <div className="relative h-[calc(100vh-6rem)] w-full rounded-2xl overflow-hidden bg-slate-950 border border-slate-800 shadow-2xl">
            <Globe
                ref={globeEl}
                backgroundColor="rgba(0,0,0,0)"
                width={window.innerWidth > 768 ? window.innerWidth - 300 : window.innerWidth - 40}
                height={window.innerHeight - 120}
                globeImageUrl="//unpkg.com/three-globe/example/img/earth-blue-marble.jpg" // Brighter "Day" texture
                bumpImageUrl="//unpkg.com/three-globe/example/img/earth-topology.png"
                backgroundImageUrl="//unpkg.com/three-globe/example/img/night-sky.png" // Stars background
                pointsData={points}
                pointLat="lat"
                pointLng="lng"
                pointColor="color"
                pointAltitude={0.1} // Static altitude for visibility check
                pointRadius="size"
                pointsMerge={true} // Performance optimization
                ringsData={rings}
                ringColor="color"
                ringMaxRadius="maxR"
                ringPropagationSpeed="propagationSpeed"
                ringRepeatPeriod="repeatPeriod"
                pointLabel={d => `
                    <div style="background: rgba(15, 23, 42, 0.95); padding: 12px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); color: white; font-family: 'Inter', sans-serif; min-width: 220px; box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.5);">
                        <div style="font-weight: 700; font-size: 15px; margin-bottom: 6px; color: #38bdf8; letter-spacing: -0.01em;">${d.place || 'Unknown Location'}</div>
                        
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px;">
                            <div style="background: rgba(255,255,255,0.05); padding: 8px; border-radius: 6px;">
                                <div style="color: #94a3b8; font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 2px;">Magnitude</div>
                                <div style="font-weight: 700; font-size: 16px; color: ${d.color};">${d.mag.toFixed(1)}</div>
                            </div>
                            <div style="background: rgba(255,255,255,0.05); padding: 8px; border-radius: 6px;">
                                <div style="color: #94a3b8; font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 2px;">Depth</div>
                                <div style="font-weight: 600; font-size: 14px;">${d.depth} km</div>
                            </div>
                        </div>

                        <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 8px;">
                            <div style="font-size: 11px; color: #64748b;">${new Date(d.time).toLocaleDateString()}</div>
                            <div style="font-size: 10px; font-family: monospace; color: #475569;">${d.lat.toFixed(2)}, ${d.lng.toFixed(2)}</div>
                        </div>
                    </div>
                `}
                atmosphereColor="#3b82f6"
                atmosphereAltitude={0.2}
            />

            {/* Legend Overlay */}
            <div className="absolute bottom-6 right-6 p-4 glass-card rounded-xl border border-slate-700/50 z-50 pointer-events-none">
                <div className="text-xs font-medium text-slate-400 mb-2 uppercase tracking-wider">Depth</div>
                <div className="space-y-2">
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]"></div>
                        <span className="text-xs text-slate-300">Shallow (&lt;30km)</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-amber-500 shadow-[0_0_10px_rgba(245,158,11,0.5)]"></div>
                        <span className="text-xs text-slate-300">Intermediate</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.5)]"></div>
                        <span className="text-xs text-slate-300">Deep (&gt;70km)</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default GlobeView;
