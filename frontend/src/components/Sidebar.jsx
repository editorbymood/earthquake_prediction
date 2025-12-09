import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Radio, Activity, Settings, Menu } from 'lucide-react';
import { clsx } from 'clsx';

const Sidebar = () => {
    const navItems = [
        { icon: LayoutDashboard, label: 'Dashboard', path: '/dashboard' },
        { icon: Radio, label: 'Live Map', path: '/map' },
        { icon: Activity, label: 'Predictor', path: '/predictor' },
        // { icon: BarChart3, label: 'Analytics', path: '/analytics' },
    ];

    return (
        <aside className="fixed left-0 top-0 h-screen w-64 bg-surface border-r border-slate-700/50 hidden md:flex flex-col z-50">
            <div className="p-6 flex items-center gap-3">
                <div className="w-8 h-8 bg-gradient-to-tr from-accent to-primary rounded-lg flex items-center justify-center shadow-lg shadow-accent/20">
                    <Activity className="text-white w-5 h-5" />
                </div>
                <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
                    SeismicAI
                </h1>
            </div>

            <nav className="flex-1 px-4 py-6 space-y-2">
                {navItems.map((item) => (
                    <NavLink
                        key={item.path}
                        to={item.path}
                        className={({ isActive }) =>
                            clsx(
                                "flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 group",
                                isActive
                                    ? "bg-primary/10 text-primary shadow-sm ring-1 ring-primary/20"
                                    : "text-slate-400 hover:bg-slate-800/50 hover:text-white"
                            )
                        }
                    >
                        <item.icon className="w-5 h-5 transition-transform group-hover:scale-110" />
                        <span className="font-medium">{item.label}</span>
                        {/* Active Indicator */}
                        <div className="ml-auto w-1 h-1 rounded-full bg-primary opacity-0 transition-opacity" />
                    </NavLink>
                ))}
            </nav>

            <div className="p-4 border-t border-slate-700/50">
                <div className="p-4 rounded-xl bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700/50">
                    <div className="flex items-center gap-3 mb-2">
                        <div className="w-2 h-2 rounded-full bg-success animate-pulse"></div>
                        <span className="text-xs font-semibold text-emerald-400">System Online</span>
                    </div>
                    <p className="text-xs text-slate-500">v2.0.0 • Connected</p>
                </div>
            </div>
        </aside>
    );
};

export default Sidebar;
