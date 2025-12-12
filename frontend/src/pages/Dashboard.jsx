import React, { useEffect, useState } from 'react';
import axios from 'axios';
import AnalyticsCharts from '../components/AnalyticsCharts';
import Map from '../components/Map';
import LiveMonitor from '../components/LiveMonitor';
import { Activity, Radio, AlertTriangle, Layers, CloudRain } from 'lucide-react';
import { clsx } from 'clsx';

const StatCard = ({ label, value, subtext, icon: Icon, color }) => (
    <div className="glass-card rounded-2xl p-6 group">
        <div className="flex items-start justify-between mb-4">
            <div>
                <p className="text-slate-400 text-sm font-medium mb-1 font-display tracking-wide uppercase text-[10px]">{label}</p>
                <h3 className="text-3xl font-bold text-white font-display tracking-tight group-hover:scale-105 transition-transform duration-300 origin-left">{value}</h3>
            </div>
            <div className={clsx("p-3 rounded-xl bg-opacity-10 backdrop-blur-sm", `bg-${color} text-${color}`)}>
                <Icon className="w-6 h-6" />
            </div>
        </div>
        {subtext && <p className="text-xs text-slate-500 font-mono">{subtext}</p>}
    </div>
);

const Dashboard = () => {
    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(true);

    const [weather, setWeather] = useState(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [earthquakeRes, weatherRes] = await Promise.all([
                    axios.get('http://localhost:8000/api/recent-earthquakes?limit=-1'), // Fetch ALL data for analytics
                    axios.get('http://localhost:8000/api/weather') // Fetches default location weather
                ]);
                setData(earthquakeRes.data);
                setWeather(weatherRes.data);
            } catch (err) {
                console.error("Failed to fetch data", err);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, []);

    // Compute stats
    const totalEvents = data.length;
    const maxMag = data.length ? Math.max(...data.map(d => d.mag)).toFixed(1) : 0;
    const avgDepth = data.length ? (data.reduce((a, b) => a + b.depth, 0) / data.length).toFixed(1) : 0;
    const significant = data.filter(d => d.mag >= 5.0).length;

    return (
        <div className="space-y-6 animate-fade-in">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-3xl font-bold text-white mb-2">Global Overview</h2>
                    <p className="text-slate-400">Real-time seismic activity monitoring system</p>
                </div>
                <div className="px-4 py-2 bg-blue-500/10 text-blue-400 rounded-lg text-sm font-medium border border-blue-500/20">
                    Last Updated: {new Date().toLocaleTimeString()}
                </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <StatCard
                    label="Total Events"
                    value={totalEvents}
                    subtext="Last 30 days"
                    icon={Radio}
                    color="blue-500"
                />
                <StatCard
                    label="Max Magnitude"
                    value={maxMag}
                    subtext={maxMag > 6 ? "High Risk" : "Normal"}
                    icon={AlertTriangle}
                    color="red-500"
                />
                <StatCard
                    label="Avg Depth"
                    value={`${avgDepth} km`}
                    subtext="Global Average"
                    icon={Layers}
                    color="emerald-500"
                />
                <StatCard
                    label="Significant"
                    value={significant}
                    subtext="Magnitude > 5.0"
                    icon={Activity}
                    color="amber-500"
                />
            </div>

            {/* Weather & Additional Info */}
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
                <StatCard
                    label="Local Conditions"
                    value={weather && weather.units ? `${weather.temperature}${weather.units.temperature_2m}` : (weather?.error ? "Unavailable" : "Loading...")}
                    subtext={weather && weather.units ? `Wind: ${weather.wind_speed} ${weather.units.wind_speed_10m}` : "Fetching..."}
                    icon={CloudRain}
                    color="sky-500"
                />
                <div className="lg:col-span-2">
                    <LiveMonitor />
                </div>
            </div>

            {/* Analytics Section */}
            <div className="mb-6">
                <h3 className="text-xl font-bold text-white mb-4">Seismic Analytics</h3>
                <AnalyticsCharts data={data} />
            </div>

            {/* Map Section */}
            <div className="h-[600px] w-full rounded-2xl overflow-hidden relative">
                {loading ? (
                    <div className="absolute inset-0 bg-surface flex items-center justify-center">
                        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
                    </div>
                ) : (
                    <Map earthquakes={data} />
                )}
            </div>
        </div>
    );
};

export default Dashboard;
