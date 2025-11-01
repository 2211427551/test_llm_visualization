'use client';

import React, { useRef, useEffect, useState, Suspense } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Box, Sphere, Line } from '@react-three/drei';
import * as THREE from 'three';
import { useVisualization } from '@/contexts/VisualizationContext';
import { useAnimation } from '@/contexts/AnimationContext';
import { colors } from '@/lib/design-system';

// 3D Visualization Component
function Visualization3D() {
  const { state: vizState } = useVisualization();
  const { state: animState } = useAnimation();
  const meshRef = useRef<THREE.Mesh>(null);
  const { camera } = useThree();

  // Animation loop
  useFrame((state, delta) => {
    if (meshRef.current && animState.isPlaying) {
      meshRef.current.rotation.y += delta * 0.5;
    }
  });

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} />

      {/* Camera Controls */}
      <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />

      {/* Main visualization elements */}
      <group position={[0, 0, 0]}>
        {/* Transformer layers visualization */}
        {Array.from({ length: 6 }, (_, i) => (
          <group key={i} position={[0, i * 2 - 5, 0]}>
            <Box
              ref={i === 0 ? meshRef : undefined}
              args={[4, 1.5, 1]}
              position={[0, 0, 0]}
            >
              <meshStandardMaterial
                color={
                  vizState.modules.attention.isActive && i % 2 === 0
                    ? colors.modules.attention
                    : vizState.modules.moe.isActive && i % 2 === 1
                    ? colors.modules.moe
                    : colors.neutral[600]
                }
                opacity={0.8}
                transparent
              />
            </Box>
            
            {/* Layer label */}
            <Text
              position={[0, 0, 1]}
              fontSize={0.3}
              color="white"
              anchorX="center"
              anchorY="middle"
            >
              Layer {i + 1}
            </Text>
          </group>
        ))}

        {/* Input/Output tokens */}
        {vizState.modules.embedding.isActive && (
          <group position={[-6, 0, 0]}>
            {Array.from({ length: 5 }, (_, i) => (
              <Sphere
                key={i}
                args={[0.3, 16, 16]}
                position={[0, i * 2 - 4, 0]}
              >
                <meshStandardMaterial color={colors.modules.embedding} />
              </Sphere>
            ))}
            <Text
              position={[0, -6, 0]}
              fontSize={0.3}
              color="white"
              anchorX="center"
              anchorY="middle"
            >
              Input Tokens
            </Text>
          </group>
        )}

        {/* Output tokens */}
        {vizState.modules.output.isActive && (
          <group position={[6, 0, 0]}>
            {Array.from({ length: 5 }, (_, i) => (
              <Sphere
                key={i}
                args={[0.3, 16, 16]}
                position={[0, i * 2 - 4, 0]}
              >
                <meshStandardMaterial color={colors.modules.output} />
              </Sphere>
            ))}
            <Text
              position={[0, -6, 0]}
              fontSize={0.3}
              color="white"
              anchorX="center"
              anchorY="middle"
            >
              Output Tokens
            </Text>
          </group>
        )}

        {/* Connection lines */}
        {vizState.settings.showConnections && (
          <>
            {/* Input to first layer */}
            <Line
              points={[
                [-5, 0, 0],
                [-2, 0, 0],
              ]}
              color={colors.visualization.flow.forward}
              lineWidth={2}
            />
            
            {/* Between layers */}
            {Array.from({ length: 5 }, (_, i) => (
              <Line
                key={i}
                points={[
                  [2, i * 2 - 4, 0],
                  [2, (i + 1) * 2 - 4, 0],
                ]}
                color={colors.visualization.flow.attention}
                lineWidth={2}
              />
            ))}
            
            {/* Last layer to output */}
            <Line
              points={[
                [2, 4, 0],
                [5, 0, 0],
              ]}
              color={colors.visualization.flow.forward}
              lineWidth={2}
            />
          </>
        )}

        {/* Attention visualization */}
        {vizState.modules.attention.isActive && (
          <group position={[0, 0, 3]}>
            {Array.from({ length: 8 }, (_, i) => (
              <Box
                key={i}
                args={[0.5, 0.5, 0.5]}
                position={[
                  Math.cos((i / 8) * Math.PI * 2) * 2,
                  Math.sin((i / 8) * Math.PI * 2) * 2,
                  0,
                ]}
              >
                <meshStandardMaterial
                  color={colors.modules.attention}
                  opacity={0.6}
                  transparent
                />
              </Box>
            ))}
            <Text
              position={[0, -3, 0]}
              fontSize={0.3}
              color="white"
              anchorX="center"
              anchorY="middle"
            >
              Attention Heads
            </Text>
          </group>
        )}
      </group>
    </>
  );
}

// Loading fallback
function LoadingFallback() {
  return (
    <div className="w-full h-full flex items-center justify-center bg-slate-100 dark:bg-slate-800">
      <div className="text-center">
        <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
        <p className="text-slate-600 dark:text-slate-400">Loading 3D visualization...</p>
      </div>
    </div>
  );
}

export function VisualizationCanvas() {
  const { state: vizState } = useVisualization();
  const [isClient, setIsClient] = useState(false);

  // Prevent hydration issues
  useEffect(() => {
    // Use setTimeout to avoid calling setState synchronously in effect
    setTimeout(() => setIsClient(true), 0);
  }, []);

  if (!isClient) {
    return <LoadingFallback />;
  }

  // Render 2D fallback for 2D mode
  if (vizState.settings.viewMode === '2d') {
    return (
      <div className="w-full h-full flex items-center justify-center bg-slate-100 dark:bg-slate-800">
        <div className="text-center">
          <div className="text-6xl mb-4">📊</div>
          <h3 className="text-xl font-semibold text-slate-700 dark:text-slate-300 mb-2">
            2D Visualization Mode
          </h3>
          <p className="text-slate-600 dark:text-slate-400">
            2D visualization coming soon
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full relative">
      <Canvas
        camera={{ position: [10, 5, 10], fov: 50 }}
        className="w-full h-full"
      >
        <Suspense fallback={null}>
          <Visualization3D />
        </Suspense>
      </Canvas>
      
      {/* View mode indicator */}
      <div className="absolute top-4 left-4 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm rounded-lg px-3 py-2 text-sm">
        <span className="font-medium text-slate-700 dark:text-slate-300">
          {vizState.settings.viewMode === '3d' ? '3D View' : 'Mixed View'}
        </span>
      </div>
      
      {/* Current step indicator */}
      <div className="absolute top-4 right-4 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm rounded-lg px-3 py-2 text-sm">
        <span className="font-medium text-slate-700 dark:text-slate-300">
          Step {vizState.currentStep + 1}/{vizState.totalSteps || 0}
        </span>
      </div>
    </div>
  );
}