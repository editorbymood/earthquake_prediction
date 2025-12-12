import React, { useState } from 'react';
import axios from 'axios';
import { Target, Zap, Info, RotateCcw, Activity } from 'lucide-react';

const Predictor = () => {
    const [inputs, setInputs] = useState({
        latitude: 35.0,
        longitude: 139.0,
        depth: 10.0,
        gap: 0,
        dmin: 0,
        rms: 0
    });

    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleChange = (e) => {
        setInputs({
            ...inputs,
            [e.target.name]: e.target.value
        });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setResult(null);

        try {
            // Convert inputs to numbers
            const payload = {
                latitude: parseFloat(inputs.latitude),
                longitude: parseFloat(inputs.longitude),
                depth: parseFloat(inputs.depth),
                gap: parseFloat(inputs.gap),
                dmin: parseFloat(inputs.dmin),
                rms: parseFloat(inputs.rms)
            };
            const res = await axios.post('http://localhost:8000/api/predict', payload);
            setTimeout(() => { // Artifical delay for effect
                setResult(res.data);
                setLoading(false);
            }, 800);
        } catch (err) {
            setError(err.response?.data?.detail || "Prediction failed. Is the model loaded?");
            setLoading(false);
        }
    };

    return (
        <div className="max-w-6xl mx-auto space-y-8 animate-fade-in">
            {/* Header */}
            <div className="text-center space-y-4">
                <h2 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-500">
                    Neuro-Seismic Predictor
                </h2>
                <p className="text-slate-400 text-lg max-w-2xl mx-auto">
                    Utilizing advanced ensemble machine learning models to estimate earthquake magnitude based on geophysical parameters.
                </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-12">
                {/* Input Form */}
                <div className="glass-card rounded-3xl p-8 shadow-2xl">
                    <div className="flex items-center gap-3 mb-8 pb-4 border-b border-slate-700/50">
                        <Target className="text-primary w-6 h-6" />
                        <h3 className="text-xl font-bold text-white">Input Parameters</h3>
                    </div>

                    <form onSubmit={handleSubmit} className="space-y-6">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-slate-300">Latitude</label>
                                <input
                                    type="number" step="0.0001" name="latitude" required
                                    value={inputs.latitude} onChange={handleChange}
                                    className="w-full bg-slate-900/50 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
                                />
                            </div>
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-slate-300">Longitude</label>
                                <input
                                    type="number" step="0.0001" name="longitude" required
                                    value={inputs.longitude} onChange={handleChange}
                                    className="w-full bg-slate-900/50 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
                                />
                            </div>
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-slate-300">Depth (km)</label>
                                <input
                                    type="number" step="0.1" name="depth" required
                                    value={inputs.depth} onChange={handleChange}
                                    className="w-full bg-slate-900/50 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
                                />
                            </div>
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-slate-300 flex items-center gap-2">
                                    Gap <Info className="w-3 h-3 text-slate-500 cursor-help" title="Azimuthal gap" />
                                </label>
                                <input
                                    type="number" step="1" name="gap"
                                    value={inputs.gap} onChange={handleChange}
                                    className="w-full bg-slate-900/50 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-primary/50 transition-all"
                                />
                            </div>
                        </div>

                        <div className="pt-6">
                            <button
                                type="submit"
                                disabled={loading}
                                className="w-full bg-gradient-to-r from-primary to-secondary hover:from-blue-600 hover:to-indigo-600 text-white font-bold py-4 rounded-xl shadow-lg shadow-primary/25 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform active:scale-95 flex items-center justify-center gap-2"
                            >
                                {loading ? (
                                    <span className="flex items-center gap-2">
                                        <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                        Processing...
                                    </span>
                                ) : (
                                    <>
                                        <Zap className="w-5 h-5" />
                                        Analyze & Predict
                                    </>
                                )}
                            </button>
                        </div>
                    </form>
                </div>

                {/* Results Area */}
                <div className="relative">
                    {result ? (
                        <div className="glass-card rounded-3xl p-8 shadow-2xl h-full flex flex-col items-center justify-center text-center animate-fade-in-up border-primary/20">
                            <div className="mb-6 relative">
                                <div className="absolute inset-0 bg-accent/20 blur-3xl rounded-full" />
                                <h2 className="text-7xl font-black text-transparent bg-clip-text bg-gradient-to-br from-white to-slate-400 relative z-10">
                                    {result.predicted_magnitude.toFixed(2)}
                                </h2>
                                <p className="text-xl font-medium text-accent mt-2">Magnitude (Mw)</p>
                            </div>

                            <div className="space-y-2 mb-8">
                                <div className="px-4 py-2 bg-slate-800 rounded-full border border-slate-700 text-sm text-slate-300 inline-block">
                                    Confidence: <span className="text-emerald-400 font-bold">{result.confidence}</span>
                                </div>
                            </div>

                            <div className="w-full grid grid-cols-2 gap-4 mt-auto">
                                <div className="bg-slate-900/50 p-4 rounded-xl text-left border border-slate-800">
                                    <p className="text-xs text-slate-500 mb-1">Model Details</p>
                                    <p className="font-semibold">Ensemble (RF + XGB)</p>
                                </div>
                                <div className="bg-slate-900/50 p-4 rounded-xl text-left border border-slate-800">
                                    <p className="text-xs text-slate-500 mb-1">Execution Time</p>
                                    <p className="font-semibold">0.04s</p>
                                </div>
                            </div>

                            <button
                                onClick={() => setResult(null)}
                                className="mt-6 flex items-center gap-2 text-slate-400 hover:text-white transition-colors"
                            >
                                <RotateCcw className="w-4 h-4" /> Reset Analysis
                            </button>
                        </div>
                    ) : (
                        <div className="bg-surface/50 border border-slate-800 rounded-3xl p-8 h-full flex flex-col items-center justify-center text-center border-dashed">
                            <div className="w-20 h-20 bg-slate-800 rounded-full flex items-center justify-center mb-6">
                                <Activity className="w-10 h-10 text-slate-600" />
                            </div>
                            <h3 className="text-xl font-bold text-slate-400 mb-2">Ready to Analyze</h3>
                            <p className="text-slate-500 max-w-sm">
                                Enter geophysical parameters in the form to generate a magnitude prediction using our AI models.
                            </p>
                            {error && (
                                <div className="mt-8 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 max-w-sm">
                                    <p className="text-sm font-semibold">Error</p>
                                    <p className="text-xs">{error}</p>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default Predictor;
