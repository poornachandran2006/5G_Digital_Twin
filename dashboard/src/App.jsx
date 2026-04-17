import { useState } from 'react';
import { ThemeProvider } from './context/ThemeContext';
import LandingPage from './components/LandingPage';
import { SimProvider } from './context/SimContext';
import { useSimSocket } from './hooks/useSimSocket';
import Sidebar from './components/layout/Sidebar';
import TopBar from './components/layout/TopBar';
import OverviewPanel from './components/panels/OverviewPanel';
import NetworkMapPanel from './components/panels/NetworkMapPanel';
import KPIPanel from './components/panels/KPIPanel';
import PredictionPanel from './components/panels/PredictionPanel';
import RLAgentPanel from './components/panels/RLAgentPanel';
import ShapPanel from './components/panels/ShapPanel';
import AnomalyPanel from './components/panels/AnomalyPanel';
import ABTestPanel from './components/panels/ABTestPanel';

const PANELS = {
  overview: OverviewPanel,
  map: NetworkMapPanel,
  kpi: KPIPanel,
  predictions: PredictionPanel,
  shap: ShapPanel,
  anomaly: AnomalyPanel,
  abtest: ABTestPanel,
  rl: RLAgentPanel,
};

function DashboardInner() {
  const [active, setActive] = useState('overview');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  useSimSocket();
  const Panel = PANELS[active];

  return (
    <div
      className="flex h-screen overflow-hidden"
      style={{ background: 'var(--bg-primary)', color: 'var(--text-primary)' }}
    >
      <Sidebar
        active={active}
        onNavigate={setActive}
        isOpen={sidebarOpen}
        setSidebarOpen={setSidebarOpen}
      />
      <div className="flex flex-col flex-1 overflow-hidden">
        <TopBar onMenuClick={() => setSidebarOpen(prev => !prev)} />
        <main
          className="flex-1 overflow-auto p-5"
          style={{ background: 'var(--bg-primary)' }}
        >
          {/* Key on active so panel re-mounts + re-animates on tab change */}
          <div key={active} className="animate-fade-in-up">
            <Panel />
          </div>
        </main>
      </div>
    </div>
  );
}

function Dashboard() {
  return (
    <SimProvider>
      <DashboardInner />
    </SimProvider>
  );
}

export default function App() {
  const [entered, setEntered] = useState(false);

  return (
    <ThemeProvider>
      {!entered
        ? <LandingPage onEnter={() => setEntered(true)} />
        : <Dashboard />
      }
    </ThemeProvider>
  );
}