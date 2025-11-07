/**
 * API 服务模块
 * 
 * 功能：
 * - 封装所有后端 API 调用
 * - 统一错误处理
 * - 类型安全的请求/响应
 * - 超时和重试机制
 */

import type { CapturedVisualizationData, ForwardResult, InitializeResult, ModelConfig } from '../types/api'
import type { StepVisualizationData } from '../types/visualization'

// API 响应接口定义
interface InitializeApiResponse {
  success: boolean
  message: string
  config: ModelConfig
}

interface ForwardApiResponse {
  success: boolean
  message: string
  logits_shape: number[]
  sequence_length: number
  captured_data?: {
    steps: StepVisualizationData[]
    runtime: CapturedVisualizationData['runtime']
    tokenSequence: string[]
    modelSummary: CapturedVisualizationData['modelSummary']
  } | null
}

// 常量配置
const DEFAULT_ERROR_MESSAGE = '网络请求失败，请检查后端服务状态。'
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api/v1'
const REQUEST_TIMEOUT = 15000

const buildErrorFromResponse = async (response: Response) => {
  try {
    const data = await response.json()
    if (typeof data?.detail === 'string') {
      return new Error(data.detail)
    }
    if (typeof data?.message === 'string') {
      return new Error(data.message)
    }
  } catch (error) {
    if (process.env.NODE_ENV === 'development') {
      console.warn('响应解析失败', error)
    }
  }
  return new Error(DEFAULT_ERROR_MESSAGE)
}

const request = async <T>(path: string, options: RequestInit = {}): Promise<T> => {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT)

  try {
    const headers = new Headers(options.headers ?? {})
    if (options.body && !headers.has('Content-Type')) {
      headers.set('Content-Type', 'application/json')
    }

    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      headers,
      signal: controller.signal,
    })

    if (!response.ok) {
      throw await buildErrorFromResponse(response)
    }

    return (await response.json()) as T
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error('请求超时，请稍后重试。')
    }
    if (error instanceof Error) {
      throw error
    }
    throw new Error(DEFAULT_ERROR_MESSAGE)
  } finally {
    window.clearTimeout(timeout)
  }
}

const mapInitialize = (payload: InitializeApiResponse): InitializeResult => {
  const rawConfig = payload.config as ModelConfig & { initialized_at?: string }
  const normalizedConfig: ModelConfig = {
    ...rawConfig,
    initializedAt:
      typeof rawConfig.initializedAt === 'string'
        ? rawConfig.initializedAt
        : typeof rawConfig.initialized_at === 'string'
          ? rawConfig.initialized_at
          : '',
  }

  return {
    success: payload.success,
    message: payload.message,
    config: normalizedConfig,
  }
}

const mapForward = (payload: ForwardApiResponse): ForwardResult => {
  const capturedData = payload.captured_data
    ? {
        steps: payload.captured_data.steps,
        runtime: payload.captured_data.runtime,
        tokenSequence: payload.captured_data.tokenSequence,
        modelSummary: payload.captured_data.modelSummary,
      }
    : undefined

  return {
    success: payload.success,
    message: payload.message,
    logitsShape: payload.logits_shape,
    sequenceLength: payload.sequence_length,
    capturedData,
  }
}

export const initializeModel = async (): Promise<InitializeResult> => {
  const response = await request<InitializeApiResponse>('/initialize', { method: 'GET' })
  return mapInitialize(response)
}

export const forwardInference = async (payload: { text: string; captureData?: boolean }): Promise<ForwardResult> => {
  const response = await request<ForwardApiResponse>('/forward', {
    method: 'POST',
    body: JSON.stringify({
      text: payload.text,
      capture_data: payload.captureData ?? true,
    }),
  })
  return mapForward(response)
}
