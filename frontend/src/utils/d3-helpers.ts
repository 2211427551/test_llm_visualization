import * as d3 from 'd3';

/**
 * D3 utility functions for data visualization
 */

export const createColorScale = (domain: number[], range: string[]) => {
  return d3.scaleLinear<string>().domain(domain).range(range);
};

export const createLinearScale = (domain: [number, number], range: [number, number]) => {
  return d3.scaleLinear().domain(domain).range(range);
};

export const createTimeScale = (domain: [Date, Date], range: [number, number]) => {
  return d3.scaleTime().domain(domain).range(range);
};

export const formatNumber = d3.format('.2f');
export const formatPercent = d3.format('.1%');

export const interpolateColor = d3.interpolateRgb;

export { d3 };
