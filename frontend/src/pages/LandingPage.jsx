import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Activity, Globe, Zap, ArrowRight, Shield } from 'lucide-react';

const LandingPage = () => {
    const navigate = useNavigate();

    return (
        <div className="min-h-screen bg-background text-white font-sans overflow-hidden relative selection:bg-primary/30">
            {/* Background Gradients */}
            <div className="absolute top-0 left-0 w-full h-full overflow-hidden z-0 pointer-events-none">
                <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] bg-primary/20 rounded-full blur-[120px] animate-pulse-slow" />
                <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-secondary/20 rounded-full blur-[120px] animate-pulse-slow delay-1000" />
            </div>

            {/* Navbar */}
            <nav className="relative z-10 flex items-center justify-between px-8 py-6 max-w-7xl mx-auto">
                <div className="flex items-center gap-2">
                    <div className="p-2 bg-gradient-to-br from-primary to-secondary rounded-lg shadow-lg shadow-primary/20">
                        <Activity className="w-6 h-6 text-white" />
                    </div>
                    <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
                        SeismicAI
                    </span>
                </div>
                <button 
                    onClick={() => navigate('/dashboard')}
                    className="px-6 py-2 rounded-full border border-slate-700 bg-surface/50 backdrop-blur-md hover:bg-surface/80 transition-all duration-300 text-sm font-medium"
                >
                    Launch App
                </button>
            </nav>

            {/* Hero Section */}
            <main className="relative z-10 max-w-7xl mx-auto px-6 pt-20 pb-16 text-center lg:text-left lg:flex lg:items-center lg:justify-between lg:gap-12">
                <div className="lg:w-1/2 space-y-8">
                    <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-surface/40 border border-slate-700/50 backdrop-blur-sm">
                        <span className="flex h-2 w-2 rounded-full bg-success animate-pulse"></span>
                        <span className="text-xs font-medium text-slate-300">Live Earthquake Monitoring</span>
                    </div>
                    
                    <h1 className="text-5xl lg:text-7xl font-bold tracking-tight leading-tight">
                        Predicting the <br/>
                        <span className="bg-clip-text text-transparent bg-gradient-to-r from-primary via-accent to-secondary">
                            Unpredictable
                        </span>
                    </h1>
                    
                    <p className="text-lg text-slate-400 max-w-xl mx-auto lg:mx-0 leading-relaxed">
                        Leveraging advanced ensemble learning and deep neural networks to analyze seismic patterns and forecast earthquake probability with unprecedented accuracy.
                    </p>

                    <div className="flex flex-col sm:flex-row items-center gap-4 justify-center lg:justify-start">
                        <button 
                            onClick={() => navigate('/dashboard')}
                            className="group relative px-8 py-4 bg-gradient-to-r from-primary to-secondary rounded-xl text-white font-semibold shadow-lg shadow-primary/25 hover:shadow-primary/40 transition-all duration-300 overflow-hidden"
                        >
                            <span className="relative z-10 flex items-center gap-2">
                                Access Dashboard <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                            </span>
                        </button>
                        <button 
                            onClick={() => window.open('https://github.com', '_blank')}
                            className="px-8 py-4 rounded-xl bg-surface border border-slate-700 text-slate-300 hover:text-white hover:bg-surface/80 transition-all duration-300"
                        >
                            View Documentation
                        </button>
                    </div>

                    <div className="pt-8 grid grid-cols-3 gap-6 border-t border-slate-800/50">
                        <div>
                            <div className="text-2xl font-bold text-white">94%</div>
                            <div className="text-sm text-slate-500">Model Accuracy</div>
                        </div>
                        <div>
                            <div className="text-2xl font-bold text-white">2.5M+</div>
                            <div className="text-sm text-slate-500">Data Points</div>
                        </div>
                        <div>
                            <div className="text-2xl font-bold text-white">24/7</div>
                            <div className="text-sm text-slate-500">Real-time Analysis</div>
                        </div>
                    </div>
                </div>

                {/* Right Content - Visual/Grid */}
                <div className="hidden lg:block lg:w-1/2 relative">
                    <div className="relative z-10 bg-surface/30 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-6 shadow-2xl">
                        <div className="grid grid-cols-2 gap-4">
                            <FeatureCard 
                                icon={<Globe className="w-6 h-6 text-accent" />} 
                                title="Global Mapping" 
                                desc="Interactive 3D visualization of seismic belts."
                            />
                            <FeatureCard 
                                icon={<Zap className="w-6 h-6 text-warning" />} 
                                title="Real-time Alerts" 
                                desc="Instant notifications for high-risk anomalies."
                            />
                            <FeatureCard 
                                icon={<Activity className="w-6 h-6 text-primary" />} 
                                title="Deep Analysis" 
                                desc="Multi-layered neural network processing."
                            />
                            <FeatureCard 
                                icon={<Shield className="w-6 h-6 text-success" />} 
                                title="Risk Assessment" 
                                desc="Localized safety scores and predictions."
                            />
                        </div>
                        
                        {/* Fake Chart or Data Viz */}
                        <div className="mt-4 p-4 bg-background/50 rounded-xl border border-slate-700/30">
                            <div className="flex items-center justify-between mb-2">
                                <div className="text-xs font-semibold text-slate-400">Seismic Activity Log</div>
                                <div className="text-xs text-primary">Live Feed</div>
                            </div>
                            <div className="space-y-2">
                                {[0.8, 0.4, 0.6, 0.9, 0.3].map((val, i) => (
                                    <div key={i} className="flex items-center gap-2">
                                        <div className="w-full h-1.5 bg-slate-700 rounded-full overflow-hidden">
                                            <div 
                                                className="h-full bg-gradient-to-r from-primary to-accent rounded-full animate-pulse" 
                                                style={{ width: `${val * 100}%` }}
                                            />
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Decorative Elements */}
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[120%] h-[120%] bg-gradient-to-tr from-primary/10 to-secondary/10 rounded-full blur-3xl -z-10" />
                </div>
            </main>
        </div>
    );
};

const FeatureCard = ({ icon, title, desc }) => (
    <div className="p-4 rounded-xl bg-slate-800/40 border border-slate-700/30 hover:bg-slate-800/60 transition-colors">
        <div className="mb-3 p-2 bg-slate-900/50 w-fit rounded-lg">{icon}</div>
        <h3 className="font-semibold text-white text-sm mb-1">{title}</h3>
        <p className="text-xs text-slate-400 leading-relaxed">{desc}</p>
    </div>
);

export default LandingPage;
