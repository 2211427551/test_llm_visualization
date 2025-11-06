import MainLayout from './layouts/MainLayout'
import ThreeColumnLayout from './layouts/ThreeColumnLayout'
import LeftPanel from './components/LeftPanel'
import CenterPanel from './components/CenterPanel'
import RightPanel from './components/RightPanel'

import { VisualizationProvider } from './hooks/useVisualizationState'

const App = () => {
  return (
    <VisualizationProvider>
      <MainLayout>
        <div className="h-[calc(100vh-12rem)]">
          <ThreeColumnLayout
            leftPanel={<LeftPanel />}
            centerPanel={<CenterPanel />}
            rightPanel={<RightPanel />}
          />
        </div>
      </MainLayout>
    </VisualizationProvider>
  )
}

export default App
