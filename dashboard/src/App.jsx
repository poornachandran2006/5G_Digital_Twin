import { useState } from 'react';
import { SimProvider } from './context/SimContext';
import { useSimSocket } from './hooks/useSimSocket';
import Sidebar from './components/layout/Sidebar';
import TopBar from './components/layout/TopBar';
import OverviewPanel from './components/panels/OverviewPanel';
import NetworkMapPanel from './components/panels/NetworkMapPanel';
import KPIPanel from './components/panels/KPIPanel';
import PredictionPanel from './components/panels/PredictionPanel';
import RLAgentPanel from './components/panels/RLAgentPanel';
import ShapPanel from "./components/panels/ShapPanel";

const PANELS = {
  overview: OverviewPanel,
  map: NetworkMapPanel,
  kpi: KPIPanel,
  predictions: PredictionPanel,
  shap: ShapPanel,
  rl: RLAgentPanel,
};

function DashboardInner() {
  const [active, setActive] = useState('overview');
  useSimSocket();
  const Panel = PANELS[active];
  return (
    <div className="flex h-screen bg-[#0a0f1e] text-white overflow-hidden">
      <Sidebar active={active} onNavigate={setActive} />
      <div className="flex flex-col flex-1 overflow-hidden">
        <TopBar />
        <main className="flex-1 overflow-auto p-4">
          <Panel />
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <SimProvider>
      <DashboardInner />
    </SimProvider>
  );
}
