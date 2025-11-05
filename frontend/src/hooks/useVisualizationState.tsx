import {
  createContext,
  type PropsWithChildren,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react'
import visualizationSteps from '../data/visualizationData'
import type { LayerVisualizationData, StepVisualizationData } from '../types/visualization'

interface VisualizationContextValue {
  steps: StepVisualizationData[]
  stepCount: number
  currentStepIndex: number
  currentStep: StepVisualizationData
  currentStepId: string
  selectedLayerId: string | null
  selectedLayer: LayerVisualizationData | null
  canGoToPrevious: boolean
  canGoToNext: boolean
  goToPreviousStep: () => void
  goToNextStep: () => void
  setStepByIndex: (index: number) => void
  setStepById: (stepId: string) => void
  selectLayer: (layerId: string) => void
  isLayerSelected: (layerId: string) => boolean
}

const VisualizationContext = createContext<VisualizationContextValue | undefined>(undefined)

const clampIndex = (index: number, upperBound: number) => Math.max(0, Math.min(index, upperBound))

export const VisualizationProvider = ({ children }: PropsWithChildren) => {
  const [currentStepIndex, setCurrentStepIndex] = useState(0)
  const [selectedLayerId, setSelectedLayerId] = useState<string | null>(() => {
    const firstLayer = visualizationSteps[0]?.layers[0]
    return firstLayer?.id ?? null
  })

  const steps = visualizationSteps
  const stepCount = steps.length
  const upperBound = Math.max(0, stepCount - 1)

  const currentStep = steps[currentStepIndex] ?? steps[0]
  const currentStepId = currentStep?.id ?? ''

  const selectedLayer = useMemo(() => {
    if (!currentStep) {
      return null
    }

    if (selectedLayerId) {
      const match = currentStep.layers.find((layer) => layer.id === selectedLayerId)
      if (match) {
        return match
      }
    }

    return currentStep.layers[0] ?? null
  }, [currentStep, selectedLayerId])

  useEffect(() => {
    if (!currentStep) {
      return
    }

    if (!selectedLayerId || !currentStep.layers.some((layer) => layer.id === selectedLayerId)) {
      const fallbackLayerId = currentStep.layers[0]?.id ?? null
      if (fallbackLayerId !== selectedLayerId) {
        setSelectedLayerId(fallbackLayerId)
      }
    }
  }, [currentStep, selectedLayerId])

  const setStepByIndex = useCallback(
    (index: number) => {
      setCurrentStepIndex((prev) => {
        const nextIndex = clampIndex(index, upperBound)
        if (nextIndex === prev) {
          return prev
        }
        return nextIndex
      })
    },
    [upperBound],
  )

  const setStepById = useCallback(
    (stepId: string) => {
      const index = steps.findIndex((step) => step.id === stepId)
      if (index >= 0) {
        setStepByIndex(index)
      }
    },
    [setStepByIndex, steps],
  )

  const goToPreviousStep = useCallback(() => {
    setStepByIndex(currentStepIndex - 1)
  }, [currentStepIndex, setStepByIndex])

  const goToNextStep = useCallback(() => {
    setStepByIndex(currentStepIndex + 1)
  }, [currentStepIndex, setStepByIndex])

  const selectLayer = useCallback((layerId: string) => {
    setSelectedLayerId((prev) => (prev === layerId ? prev : layerId))
  }, [])

  const isLayerSelected = useCallback(
    (layerId: string) => {
      return selectedLayerId === layerId
    },
    [selectedLayerId],
  )

  const value = useMemo<VisualizationContextValue>(() => {
    const canGoToPrevious = currentStepIndex > 0
    const canGoToNext = currentStepIndex < upperBound

    return {
      steps,
      stepCount,
      currentStepIndex,
      currentStep,
      currentStepId,
      selectedLayerId,
      selectedLayer: selectedLayer ?? null,
      canGoToPrevious,
      canGoToNext,
      goToPreviousStep,
      goToNextStep,
      setStepByIndex,
      setStepById,
      selectLayer,
      isLayerSelected,
    }
  }, [
    currentStep,
    currentStepId,
    currentStepIndex,
    goToNextStep,
    goToPreviousStep,
    isLayerSelected,
    selectedLayer,
    selectedLayerId,
    setStepById,
    setStepByIndex,
    stepCount,
    steps,
    upperBound,
  ])

  return <VisualizationContext.Provider value={value}>{children}</VisualizationContext.Provider>
}

export const useVisualizationState = () => {
  const context = useContext(VisualizationContext)
  if (!context) {
    throw new Error('useVisualizationState 必须在 VisualizationProvider 内部使用')
  }
  return context
}
