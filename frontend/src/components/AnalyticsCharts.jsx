import React, { useMemo } from 'react';
import {
    AreaChart,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    BarChart,
    Bar,
    ScatterChart,
    Scatter,
    ZAxis
} from 'recharts';
import { clsx } from 'clsx';
import { TrendingUp, Layers, Activity } from 'lucide-react';

const ChartCard = ({ title, icon: Icon, children, className }) => (
    <div className={clsx("glass-card rounded-2xl p-6 border border-slate-700/50 bg-slate-900/50 backdrop-blur-xl", className)}>
        <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-lg bg-primary/10 text-primary">
                <Icon size={20} />
            </div>
            <h3 className="text-lg font-semibold text-white font-display tracking-tight">{title}</h3>
        </div>
        <div className="h-[300px] w-full">
            {children}
        </div>
    </div>
);

const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
        return (
            <div className="bg-slate-900/90 border border-slate-700 p-3 rounded-xl shadow-xl backdrop-blur-md">
                <p className="text-slate-300 text-sm font-medium mb-1">{label}</p>
                {payload.map((entry, index) => (
                    <p key={index} className="text-sm font-bold" style={{ color: entry.color }}>
                        {entry.name}: {entry.value}
                    </p>
                ))}
            </div>
        );
    }
    return null;
};

const AnalyticsCharts = ({ data }) => {
    // 1. Magnitude Distribution (Histogram-like)
    const magDistribution = useMemo(() => {
        if (!data.length) return [];
        const bins = {};
        for (let i = 0; i <= 9; i += 0.5) {
            bins[i] = 0;
        }
        data.forEach(d => {
            const bin = Math.floor(d.mag * 2) / 2; // Round to nearest 0.5
            if (bins[bin] !== undefined) bins[bin]++;
        });
        return Object.keys(bins).map(bin => ({
            range: `M${bin}-${Number(bin) + 0.5}`,
            count: bins[bin]
        }));
    }, [data]);

    // 2. Depth vs Magnitude (Scatter)
    const depthVsMag = useMemo(() => {
        return data
            .filter((_, i) => i % 5 === 0) // Downsample for performance if needed
            .map(d => ({
                depth: d.depth,
                mag: d.mag,
                place: d.place
            }));
    }, [data]);

    // 3. Time Series (Daily Counts) - simplified for "Recent" data usually
    const timeSeries = useMemo(() => {
        if (!data.length) return [];
        const counts = {};
        data.forEach(d => {
            // Assuming data is sorted or we sort it. ISO string "YYYY-MM-DD..."
            const date = new Date(d.time).toLocaleDateString();
            counts[date] = (counts[date] || 0) + 1;
        });
        // Convert to array and take last 14 days or so
        return Object.keys(counts).map(date => ({
            date,
            events: counts[date]
        })).slice(-14);
    }, [data]);

    if (!data || data.length === 0) return null;

    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-fade-in">

            {/* Magnitude Distribution */}
            <ChartCard title="Magnitude Distribution" icon={Activity}>
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={magDistribution}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} vertical={false} />
                        <XAxis
                            dataKey="range"
                            stroke="#94a3b8"
                            fontSize={12}
                            tickLine={false}
                            axisLine={false}
                        />
                        <YAxis
                            stroke="#94a3b8"
                            fontSize={12}
                            tickLine={false}
                            axisLine={false}
                        />
                        <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.05)' }} />
                        <Bar
                            dataKey="count"
                            name="Earthquakes"
                            fill="#3b82f6"
                            radius={[4, 4, 0, 0]}
                            animationDuration={1500}
                        />
                    </BarChart>
                </ResponsiveContainer>
            </ChartCard>

            {/* Depth vs Magnitude */}
            <ChartCard title="Depth Analysis" icon={Layers}>
                <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                        <XAxis
                            type="number"
                            dataKey="depth"
                            name="Depth"
                            unit="km"
                            stroke="#94a3b8"
                            fontSize={12}
                            tickLine={false}
                            axisLine={false}
                        />
                        <YAxis
                            type="number"
                            dataKey="mag"
                            name="Magnitude"
                            stroke="#94a3b8"
                            fontSize={12}
                            tickLine={false}
                            axisLine={false}
                        />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} content={({ active, payload }) => {
                            if (active && payload && payload.length) {
                                const d = payload[0].payload;
                                return (
                                    <div className="bg-slate-900/90 border border-slate-700 p-3 rounded-xl shadow-xl backdrop-blur-md">
                                        <p className="text-slate-300 font-medium mb-1">{d.place}</p>
                                        <p className="text-sm text-emerald-400">Depth: {d.depth} km</p>
                                        <p className="text-sm text-amber-400">Mag: {d.mag}</p>
                                    </div>
                                )
                            }
                            return null;
                        }} />
                        <Scatter name="Events" data={depthVsMag} fill="#10b981" fillOpacity={0.6} line={false} />
                    </ScatterChart>
                </ResponsiveContainer>
            </ChartCard>

            {/* Timeline Analysis - Full Width */}
            <ChartCard title="Seismic Activity Timeline" icon={TrendingUp} className="lg:col-span-2">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={timeSeries}>
                        <defs>
                            <linearGradient id="colorEvents" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} vertical={false} />
                        <XAxis
                            dataKey="date"
                            stroke="#94a3b8"
                            fontSize={12}
                            tickLine={false}
                            axisLine={false}
                        />
                        <YAxis
                            stroke="#94a3b8"
                            fontSize={12}
                            tickLine={false}
                            axisLine={false}
                        />
                        <Tooltip content={<CustomTooltip />} />
                        <Area
                            type="monotone"
                            dataKey="events"
                            name="Events"
                            stroke="#f59e0b"
                            fillOpacity={1}
                            fill="url(#colorEvents)"
                            strokeWidth={2}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </ChartCard>

        </div>
    );
};

export default AnalyticsCharts;
