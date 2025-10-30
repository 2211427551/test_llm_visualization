'use client';

import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text } from '@react-three/drei';
import { useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { getHeatmapColor } from '@/lib/visualization/colors';

interface Matrix3DProps {
  data: number[][];
  title?: string;
  showValues?: boolean;
  interactive?: boolean;
  onCellClick?: (i: number, j: number, value: number) => void;
}

function MatrixMesh({ 
  data, 
  showValues, 
  onCellClick 
}: { 
  data: number[][]; 
  showValues?: boolean;
  onCellClick?: (i: number, j: number, value: number) => void;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const [hoveredCell, setHoveredCell] = useState<{ i: number; j: number } | null>(null);
  
  // Rotate slightly for better view
  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.2) * 0.1;
    }
  });
  
  const boxes = useMemo(() => {
    const rows = data.length;
    const cols = data[0]?.length || 0;
    const spacing = 1.2;
    const offsetX = (cols - 1) * spacing / 2;
    const offsetY = (rows - 1) * spacing / 2;
    
    return data.flatMap((row, i) =>
      row.map((value, j) => {
        const normalizedValue = Math.max(0, Math.min(1, value));
        const color = new THREE.Color(getHeatmapColor(normalizedValue));
        
        return {
          position: [
            j * spacing - offsetX,
            -i * spacing + offsetY,
            0
          ] as [number, number, number],
          color,
          value,
          coords: { i, j },
        };
      })
    );
  }, [data]);
  
  return (
    <group ref={groupRef}>
      {boxes.map((box, idx) => {
        const isHovered = hoveredCell?.i === box.coords.i || hoveredCell?.j === box.coords.j;
        const scale = isHovered ? 1.1 : 1;
        
        return (
          <group key={idx}>
            <mesh
              position={box.position}
              scale={scale}
              onClick={() => onCellClick?.(box.coords.i, box.coords.j, box.value)}
              onPointerOver={() => setHoveredCell(box.coords)}
              onPointerOut={() => setHoveredCell(null)}
            >
              <boxGeometry args={[1, 1, 0.2]} />
              <meshStandardMaterial 
                color={box.color} 
                metalness={0.3}
                roughness={0.4}
                emissive={isHovered ? box.color : new THREE.Color(0x000000)}
                emissiveIntensity={isHovered ? 0.3 : 0}
              />
            </mesh>
            
            {showValues && (
              <Text
                position={[box.position[0], box.position[1], box.position[2] + 0.15]}
                fontSize={0.3}
                color="white"
                anchorX="center"
                anchorY="middle"
              >
                {box.value.toFixed(2)}
              </Text>
            )}
          </group>
        );
      })}
    </group>
  );
}

export function Matrix3D({ 
  data, 
  title, 
  showValues = false, 
  interactive = true,
  onCellClick 
}: Matrix3DProps) {
  return (
    <div className="relative w-full h-[500px] bg-slate-900 rounded-xl overflow-hidden">
      {title && (
        <div className="absolute top-4 left-4 z-10 bg-black/50 backdrop-blur px-4 py-2 rounded-lg">
          <h3 className="text-white font-medium">{title}</h3>
        </div>
      )}
      
      <Canvas camera={{ position: [0, 0, 20], fov: 50 }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        <pointLight position={[-10, -10, 10]} intensity={0.5} />
        
        <MatrixMesh 
          data={data} 
          showValues={showValues}
          onCellClick={onCellClick}
        />
        
        {interactive && (
          <OrbitControls 
            enablePan={true} 
            enableZoom={true}
            enableRotate={true}
            maxDistance={50}
            minDistance={10}
          />
        )}
      </Canvas>
      
      <div className="absolute bottom-4 right-4 text-xs text-slate-400 bg-black/50 backdrop-blur px-3 py-1 rounded">
        {interactive ? 'Drag to rotate, scroll to zoom' : 'Auto-rotating'}
      </div>
    </div>
  );
}
