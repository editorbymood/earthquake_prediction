import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Radio, Activity, Settings, Menu, Globe } from 'lucide-react';
import { clsx } from 'clsx';

const Sidebar = () => {
    const navItems = [
        { icon: LayoutDashboard, label: 'Dashboard', path: '/dashboard' },
        { icon: Globe, label: '3D Globe', path: '/globe' },
        { icon: Radio, label: 'Live Map', path: '/map' },
        { icon: Activity, label: 'Predictor', path: '/predictor' },
        // { icon: BarChart3, label: 'Analytics', path: '/analytics' },
    ];

    return (
        <aside className="fixed left-0 top-0 h-screen w-64 bg-slate-900/50 backdrop-blur-xl border-r border-white/5 hidden md:flex flex-col z-50">
            <div className="p-6 flex items-center gap-3">
                <div className="w-10 h-10 bg-gradient-to-tr from-primary to-accent rounded-xl flex items-center justify-center shadow-lg shadow-primary/20">
                    <Activity className="text-white w-6 h-6" />
                </div>
                <div>
                    <h1 className="text-2xl font-display font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
                        SeismicAI
                    </h1>
                    <p className="text-[10px] text-slate-500 font-mono tracking-wider">PREDICTION SYSTEM</p>
                </div>
            </div>

            <nav className="flex-1 px-4 py-6 space-y-2">
                {navItems.map((item) => (
                    <NavLink
                        key={item.path}
                        to={item.path}
                        className={({ isActive }) =>
                            clsx(
                                "flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 group relative overflow-hidden",
                                isActive
                                    ? "bg-primary/10 text-white shadow-inner"
                                    : "text-slate-400 hover:bg-white/5 hover:text-white"
                            )
                        }
                    >
                        {({ isActive }) => (
                            <>
                                {isActive && (
                                    <div className="absolute left-0 top-0 bottom-0 w-1 bg-primary rounded-r-full" />
                                )}
                                <item.icon className={clsx("w-5 h-5 transition-transform group-hover:scale-110", isActive && "text-primary")} />
                                <span className="font-medium">{item.label}</span>
                            </>
                        )}
                    </NavLink>
                ))}
            </nav>

            <div className="p-4 border-t border-white/5">
                <div className="p-4 rounded-xl bg-gradient-to-br from-slate-800/50 to-slate-900/50 border border-white/5 backdrop-blur-md">
                    <div className="flex items-center gap-3 mb-2">
                        <div className="w-2 h-2 rounded-full bg-success animate-pulse"></div>
                        <span className="text-xs font-semibold text-emerald-400">System Online</span>
                    </div>
                    <p className="text-xs text-slate-500">v2.1.0 • Stable</p>
                </div>
            </div>
        </aside>
    );
};

export default Sidebar;
