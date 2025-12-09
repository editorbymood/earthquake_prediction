import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Predictor from './pages/Predictor';
import MapStart from './components/Map'; // We might use a full page map too

const AppLayout = ({ children }) => (
    <div className="flex min-h-screen bg-background text-white font-sans selection:bg-primary/30">
        <Sidebar />
        <main className="flex-1 md:ml-64 p-8 overflow-x-hidden">
            {children}
        </main>
    </div>
);

import LandingPage from './pages/LandingPage';

function App() {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<LandingPage />} />
                <Route path="/dashboard" element={
                    <AppLayout>
                        <Dashboard />
                    </AppLayout>
                } />
                <Route path="/predictor" element={
                    <AppLayout>
                        <Predictor />
                    </AppLayout>
                } />
                <Route path="/map" element={
                    <AppLayout>
                        <div className="h-[calc(100vh-4rem)] rounded-2xl overflow-hidden border border-slate-700/50">
                            <DashboardMapWrapper />
                        </div>
                    </AppLayout>
                } />
            </Routes>
        </Router>
    );
}

// Simple wrapper to fetch data for the standalone map page
const DashboardMapWrapper = () => {
    const [data, setData] = React.useState([]);
    React.useEffect(() => {
        // Fetch logic
        import('axios').then(axios => {
            axios.get('http://localhost:8000/api/recent-earthquakes')
                .then(res => setData(res.data))
                .catch(console.error);
        })
    }, []);
    // Lazy load map to avoid issues if not loaded
    const Map = React.lazy(() => import('./components/Map'));
    return (
        <React.Suspense fallback={<div className="text-center p-10">Loading Map...</div>}>
            <Map earthquakes={data} />
        </React.Suspense>
    )
}

export default App;
