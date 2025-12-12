import React, { useEffect, useState, useRef } from 'react';
import { Activity, Radio } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const LiveMonitor = () => {
    const [events, setEvents] = useState([]);
    const [graphData, setGraphData] = useState([]);
    const [status, setStatus] = useState("disconnected");
    const [threshold, setThreshold] = useState(5.0);
    const ws = useRef(null);
    const MAX_GRAPH_POINTS = 50;

    useEffect(() => {
        // Connect to WebSocket using dynamic hostname
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.hostname;
        const wsUrl = `${protocol}//${host}:8000/ws/seismic`;

        ws.current = new WebSocket(wsUrl);

        ws.current.onopen = () => {
            console.log("WS Connected");
            setStatus("connected");
        };
        ws.current.onerror = (e) => console.error("WS Error:", e);
        ws.current.onclose = () => setStatus("disconnected");
        ws.current.onmessage = (event) => {
            const data = JSON.parse(event.data);

            // Update Event List
            setEvents(prev => [data, ...prev].slice(0, 5)); // Keep last 5 for list

            // Update Graph Data
            setGraphData(prev => {
                const newData = [...prev, {
                    time: new Date(data.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }),
                    mag: data.mag,
                    alert: data.alert
                }];
                return newData.slice(-MAX_GRAPH_POINTS); // Keep last N points
            });
        };

        return () => {
            if (ws.current) ws.current.close();
        };
    }, []);

    const updateThreshold = async (val) => {
        setThreshold(val);
        try {
            const host = window.location.hostname;
            await fetch(`http://${host}:8000/api/alerts/config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ threshold: parseFloat(val) })
            });
        } catch (e) {
            console.error("Failed to update alert config", e);
        }
    };

    return (
        <div className="glass-card rounded-2xl p-6 overflow-hidden relative">
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <Radio className={`w-5 h-5 ${status === 'connected' ? 'text-red-500 animate-pulse' : 'text-slate-500'}`} />
                    <h3 className="text-lg font-bold text-white">Live Seismic Feed</h3>
                </div>
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2 text-xs text-slate-400">
                        <span>Alert Threshold:</span>
                        <input
                            type="number"
                            value={threshold}
                            onChange={(e) => updateThreshold(e.target.value)}
                            className="bg-slate-900 border border-slate-700 rounded px-2 py-1 w-16 text-white"
                        />
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Live Graph Section */}
                <div className="lg:col-span-2 h-[200px] w-full bg-slate-900/50 rounded-xl border border-slate-700/50 p-2">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={graphData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
                            <XAxis
                                dataKey="time"
                                stroke="#64748b"
                                fontSize={10}
                                tickLine={false}
                                axisLine={false}
                                interval="preserveStartEnd"
                            />
                            <YAxis
                                domain={[0, 10]}
                                stroke="#64748b"
                                fontSize={10}
                                tickLine={false}
                                axisLine={false}
                                width={20}
                            />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc' }}
                                itemStyle={{ color: '#38bdf8' }}
                            />
                            <Line
                                type="monotone"
                                dataKey="mag"
                                stroke="#38bdf8"
                                strokeWidth={2}
                                dot={false}
                                activeDot={{ r: 4, fill: '#38bdf8' }}
                                isAnimationActive={false} // Disable animation for smoother realtime updates
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>

                {/* Event List Section */}
                <div className="space-y-3 max-h-[200px] overflow-y-auto pr-1 custom-scrollbar">
                    {events.length === 0 ? (
                        <div className="text-center py-8 text-slate-500 text-sm">
                            Waiting for signal...
                        </div>
                    ) : (
                        events.map((event, idx) => (
                            <div key={idx} className={`flex items-center justify-between p-3 rounded-lg border animate-fade-in-up ${event.alert ? 'bg-red-500/10 border-red-500/50' : 'bg-slate-800/50 border-slate-700/50'}`}>
                                <div className="flex items-center gap-3">
                                    <div className={`w-2 h-2 rounded-full ${event.mag > 5 ? 'bg-red-500' : 'bg-green-500'}`}></div>
                                    <div>
                                        <p className="text-sm font-bold text-white">Mag {event.mag}</p>
                                        <p className="text-[10px] text-slate-400">{event.depth}km</p>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <p className="text-[10px] text-slate-500 font-mono">
                                        {new Date(event.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                                    </p>
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </div>

        </div>
    );
};

export default LiveMonitor;
